from argparse import ArgumentParser
from datetime import datetime

import yaml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
from einops import rearrange

from futscml import *
from train import ImageToImageGenerator_JohnsonFutschik

class InferDataset(Dataset):
    def __init__(self, dataroot, xform):
        self.root = dataroot
        self.frames = images_in_directory(self.root)
        self.xform = xform
        self.loaded = []
        for idx in range(len(self.frames)):
            x = pil_loader(os.path.join(self.root, self.frames[idx]))
            x = self.xform(x)
            self.loaded.append(x)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        x = self.loaded[idx]
        return x, self.frames[idx]

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('checkpoint_dir', help='Checkpoint directory', type=str)
    p.add_argument('checkpoint_filename', help='Network checkpoint', type=str)
    p.add_argument('output_dir', help='Output dir', type=str)
    p.add_argument('input_dir', help='Input directory to stylize', type=str)
    p.set_defaults(feature=False)
    argds = p.parse_args()

    if argds.checkpoint_dir.endswith(os.sep):
        argds.checkpoint_dir = argds.checkpoint_dir[:-1]

    _, experiment_name = os.path.split(argds.checkpoint_dir)
    conf_filepath = os.path.join(argds.checkpoint_dir, 'config.yml')
    with open(conf_filepath, 'r') as f:
        config = yaml.safe_load(f)

    config['input'] = argds.input_dir
    config['output'] = argds.output_dir
    config['checkpoint'] = os.path.join(argds.checkpoint_dir, argds.checkpoint_filename)

    model = ImageToImageGenerator_JohnsonFutschik(config=config, **config['model_params'])

    ckpt = torch.load(config['checkpoint'])['state_dict']
    model.load_state_dict(ckpt)

    model = model.to(config['device']).eval()

    transform = ImageTensorConverter(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                                     resize=f'flex;8;max;{config["resize"]}' if config["resize"] is not None else f'flex;8', drop_alpha=True)
    dataset = InferDataset(config['input'], transform)
    dataset = DataLoader(dataset, num_workers=0, batch_size=1)

    if not os.path.exists(config['output']):
        os.makedirs(config['output'])

    generated = []
    begin = datetime.now()
    with torch.no_grad():
        with torch.amp.autocast(dtype=torch.float16, device_type='cuda', enabled=False):
            pbar = tqdm(dataset)
            for batch in pbar:
                t, p = batch[0], batch[1]
                pbar.set_description("Processing: " + p[0])
                t = t.to(config['device'])
                if config['model_params']['input_channels'] == 4:
                    ones = torch.ones_like(t[:, :1], device=t.device)
                    t = torch.cat([ones, t], dim=1)
                r = model(t)
                r = torch.clip(r, -1, 1)
                n, c, h, w = r.shape
                for i in range(n):
                    # transform(r[i]).save(os.path.join(config['output'], p[i]))
                    generated.append(r[i:i+1] * 127.5 + 127.5)

    video = torch.cat(generated, dim=0).cpu()
    video_path = os.path.join(config['output'], 'output.mp4')
    fps = 24
    torchvision.io.write_video(video_path, rearrange(video, 'b c h w -> b h w c'), fps, options={'crf': '18'})

