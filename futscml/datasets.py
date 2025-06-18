from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import torch
from futscml import is_image, pil_loader, images_in_directory, subdirectories

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import check_integrity

# from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

class ImageDirectory(data.Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self._root = root
        self.root_path = os.path.expanduser(root)
        self.transform = transform
        self.files = sorted(os.listdir(self.root_path))
        def is_valid_file(path):
            return is_image(path) and not path.lower().startswith('.')
        self.files = list(filter(is_valid_file, self.files))
        self.loader = pil_loader
        print(f"{self.__class__.__name__}: Found {len(self)} images in {root}")
        
    def __getitem__(self, index):
        path = os.path.join(self.root_path, self.files[index])
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.files)

class DirectoryOfSubdirectories(data.Dataset):
    def __init__(self, root, transform):
        super().__init__()
        self.root = root
        self.subdirs = subdirectories(self.root)
        self.images = [images_in_directory(os.path.join(self.root, subdir)) for subdir in self.subdirs]
        self.total_images = sum([len(subdir) for subdir in self.images])
        self.transform = transform

        self.lut = np.empty(self.total_images, dtype=np.uint32)
        self.in_class_lut = np.empty(self.total_images, dtype=np.uint32)
        e = 0
        for clazz in range(len(self.subdirs)):
            self.lut[e:e+len(self.images[clazz])] = clazz
            self.in_class_lut[e:e+len(self.images[clazz])] = np.arange(len(self.images[clazz]))
            e += len(self.images[clazz])
        
    def __len__(self):
        return self.total_images
    
    def __getitem__(self, idx):
        clazz = int(self.lut[idx])
        path = os.path.join(self.root, self.subdirs[clazz], self.images[clazz][self.in_class_lut[idx]])
        tensor = self.transform(pil_loader(path))
        return tensor, clazz


class HardMiningSampler(data.Sampler):
    def __init__(self, dataset, history_per_term=10):
        super().__init__(dataset)
        self.dataset = dataset
        self.history_per_term = history_per_term
        self._loss_history = np.zeros(
            [len(self.dataset), history_per_term],
            dtype=np.float64
        )
        self._loss_counts = np.zeros([len(self.dataset)],
                                     dtype=np.int64)

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()

    def update_with_local_losses(self, indices, losses):
        for idx, loss in zip(indices, losses):
            loss = float(loss.detach().cpu().numpy())
            if self._loss_counts[idx] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[idx, :-1] = self._loss_history[idx, 1:]
                self._loss_history[idx, -1] = loss
            else:
                self._loss_history[idx, self._loss_counts[idx] % self._loss_history.shape[1]] = loss
                self._loss_counts[idx] += 1

    def weights(self):
        if not self._warmed_up():
            return np.ones([len(self.dataset)], dtype=np.float64) / len(self.dataset)
        weights = np.mean(self._loss_history ** 2, axis=-1)
        weights /= np.sum(weights)
        return weights

    def __iter__(self):
        while True:
            yield np.random.choice(len(self.dataset), p=self.weights())

    def __next__(self):
        # sample with replacement according to weights
        return np.random.choice(len(self.dataset), p=self.weights())

    # returns Index, Batch
    def __call__(self):
        return next(self)


class InfiniteDatasetSampler:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.enumerator = None
        self.generator = self._generator()

    def reset_enumerator(self):
        self.enumerator = enumerate(self.dataloader)

    def _generator(self):
        if self.enumerator is None: self.reset_enumerator()
        while True:
            try:
                yield next(self.enumerator)
            except StopIteration:
                self.reset_enumerator()
                yield next(self.enumerator)

    def __next__(self):
        return next(self.generator)

    # returns Index, Batch
    def __call__(self):
        return next(self)

class RestrictedCIFAR10(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, labels_include=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        if labels_include is None:
            self.labels_include = list(range(0, 10))
        else:
            self.labels_include = labels_include

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                labels_mask = ([(i in self.labels_include) for i in entry['labels']])
                data_filtered = entry['data'][labels_mask]
                labels_filtered = np.array(entry['labels'])[labels_mask]
                self.data.append(data_filtered)
                if 'labels' in entry:
                    self.targets.extend(labels_filtered)
                else:
                    raise Exception("Revisit")
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")



if __name__ == "__main__":
    labels = [2,5]
    test = RestrictedCIFAR10('data/', labels_include=labels)
    for p in test:
        assert(p[1] in labels)
    print(len(test))