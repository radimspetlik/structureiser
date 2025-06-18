from .futscml import guess_model_device
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np

def capture_layer_indices(sequential_model, capture_indices, x):
    feat = []
    if -1 in capture_indices: feat.append(x)
    for i in range(len(sequential_model)):
        x = sequential_model[i](x)
        if i in capture_indices: feat.append(x)
        if len(feat) == len(capture_indices): break
    return feat


def image_to_image_net_forward(model, dataset, batch_size=1, device='cuda:0', checkpoint=None, checkpoint_state_dict='state_dict'):
    '''
    If dataset supports .filename(idx), it will be returned alongside each sample
    '''
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    np.random.seed(0)

    if checkpoint is not None:
        cp = torch.load(checkpoint)
        model.load_state_dict(cp[checkpoint_state_dict])
    if device is None:
        device = guess_model_device(model)
    model = model.to(device)
    model.eval()
    
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=False)

    for i, b in enumerate(dataloader):
        if isinstance(b, list) or isinstance(b, tuple):
            b = b[0]
        inputs = b.to(device)
        result = model(inputs)
        for sample in range(result.shape[0]):
            if hasattr(dataset, 'filename') and callable(dataset.filename):
                yield result[sample], dataset.filename(i * batch_size + sample)
            else:
                yield result[sample]


# Example usecase
if __name__ == "__main__":
    class InferDataset(Dataset):
        def __init__(self, dataroot, xform):
            self.root = os.path.join(dataroot, 'input')
            self.frames = images_in_directory(self.root)
            self.tensors = []
            self.xform = xform
            for frame in self.frames:
                x = pil_loader(os.path.join(self.root, frame))
                self.tensors.append(self.xform(x))
        
        def __len__(self):
            return len(self.tensors)

        def __getitem__(self, idx):
            return self.tensors[idx]
        
        def filename(self, idx):
            return os.path.basename(self.frames[idx])

    parser = ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model = ImageToImageGenerator_JohnsonFutschik(norm_layer='instance_norm', use_bias=True, tanh=True)
    transform = ImageTensorConverter(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], resize='flex;8;max;512', drop_alpha=True)
    data = InferDataset(args.input_dir, transform)

    result_generator = image_to_image_net_forward(model, data, batch_size=2, checkpoint=args.checkpoint)
    for result, filename in result_generator:
        pil = transform(result)
        pil.save(os.path.join(args.output_dir, filename))