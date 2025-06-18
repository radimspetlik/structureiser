import os
import numpy as np
import scipy.misc
import shutil
import torch
from collections import deque
import csv
from datetime import datetime

# no longer need tf for tensorboard
from torch.utils import tensorboard
from .futscml import dotdict, copy_file

class TensorboardLogger():
    def __init__(self, output_path, suffix=None, checkpoint_fmt='checkpoint_%03d.pth'):
        self.output_path = output_path
        self.suffix = '' if suffix is None else suffix

        self.created_at = datetime.now()
        self.dir_created = False
        self.summary_writer = None
        self.checkpoint_fmt = checkpoint_fmt

        self._init_summary()
    
    def __del__(self):
        self.flush()
        # seems to keep process alive
        # if self.summary_writer:
        #     self.summary_writer.close()

    def experiment_name(self):
        return self.created_at.strftime('%Y-%m-%d_%H-%M-%S') + self.suffix

    def location(self):
        return os.path.join(self.output_path, self.experiment_name())

    def _init_summary(self):
        self._init_dir()
        if self.summary_writer is not None: return
        print(f'Writing logs to {self.location()}.')
        self.summary_writer = tensorboard.SummaryWriter(self.location())

    def _init_dir(self):
        if self.dir_created: return
        if not os.path.exists(self.location()):
            os.makedirs(self.location())
        self.dir_created = True

    def log_scalar(self, tag, value, step):
        self._init_summary()
        self.summary_writer.add_scalar(tag=tag, scalar_value=value, global_step=step)
        
    def log_multiple_scalars(self, scalar_dict, step):
        self._init_summary()
        for k, v in scalar_dict.items():
            if v is None: continue
            self.log_scalar(k, v, step)

    def log_scalars_single_plot(self, tag, subtag_value_dict, step):
        self._init_summary()
        self.summary_writer.add_scalars(main_tag=tag, tag_scalar_dict=subtag_value_dict, global_step=step)

    def log_histogram(self, tag, values, step, bins='auto'):
        self._init_summary()
        self.summary_writer.add_histogram(tag=tag, values=values, global_step=step, bins=bins)

    def log_image(self, tag, image, step, format='CHW'):
        self._init_summary()
        self.summary_writer.add_image(tag=tag, img_tensor=image, global_step=step, dataformats=format)

    def log_multiple_images(self, tag, images, step, format='NCHW'):
        self._init_summary()
        self.summary_writer.add_images(tag=tag, img_tensor=image, global_step=step, dataformats=format)

    def log_figure(self, tag, figure, step):
        self._init_summary()
        self.summary_writer.add_figure(tag=tag, figure=figure, global_step=step, close=True)

    def log_text(self, tag, text, step):
        self._init_summary()
        self.summary_writer.add_text(tag=tag, text_string=text, global_step=step)
    
    # expects uint or [0, 1]
    def log_video(self, tag, video, step, fps=4):
        self._init_summary()
        self.summary_writer.add_video(tag=tag, vid_tensor=video, global_step=step, fps=fps)

    def log_audio(self, tag, audio, step, samplerate=44100):
        self._init_summary()
        self.summary_writer.add_audio(tag=tag, snd_tensor=audio, global_step=step, sample_rate=samplerate)    

    def log_checkpoint(self, state, epoch_tag):
        self._init_dir()
        where = os.path.join(self.location(), self.checkpoint_fmt % epoch_tag)
        torch.save(state, where)

    def _best_checkpoint_location(self, prefix=''):
        where = os.path.join(self.location(), prefix + "checkpoint_best.pth")
        return where

    def log_checkpoint_best(self, state, prefix=''):
        self._init_dir()
        where = self._best_checkpoint_location(prefix)
        torch.save(state, where)

    def log_mkdir(self, dirname):
        self._init_dir()
        os.makedirs(os.path.join(self.location(), dirname))

    def log_file(self, path: str, output_name=None, output_path=None):
        self._init_dir()
        where = os.path.join(self.location(), '' if output_path is None else output_path, os.path.basename(path) if output_name is None else output_name)
        if not os.path.isfile(path):
            print(f'File {path} does not exist.')
            return
        copy_file(path, where)

    def flush(self):
        if self.summary_writer:
            self.summary_writer.flush()


class FileLogger():
    def __init__(self, log_dir, log_suffix=None, checkpoint_fmt=None):
        self.log_dir = log_dir + (log_suffix if log_suffix is not None else '')
        if not os.path.exists(self.log_dir) or not os.path.isdir(self.log_dir):
            self.log_dir = os.path.join(self.log_dir, datetime.now().strftime('%d-%b-%Y-%H-%M-%S'))
            os.makedirs(self.log_dir)
        else:
            # if dir exists, create a new one beneath
            self.log_dir = os.path.join(self.log_dir, datetime.now().strftime('%d-%b-%Y-%H-%M-%S'))
            os.makedirs(self.log_dir)
        self.scalar_file = None
        self.scalar_file_csv = None
        self.checkpoint_fmt = checkpoint_fmt

    def log_scalar(self, tag, value, step):
        if self.scalar_file is None:
            self.scalar_file = open(os.path.join(self.log_dir, 'scalars.csv'), 'w+', newline='')
            self.scalar_file_csv = csv.writer(self.scalar_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        self.scalar_file_csv.writerow([step, tag, value])
        self.scalar_file.flush()

    def log_image(self, tag, images, step):
        dp = os.path.join(self.log_dir, str(step), str(tag))
        if not os.path.exists(dp):
            os.makedirs(dp)
        for i in range(len(images)):
            img = np.array(images[i])
            scipy.misc.toimage(np.squeeze(img), cmin=0, cmax=255).save(os.path.join(dp, f'image_{i:03d}.png'), format="png")

    def log_checkpoint(self, state, epoch_tag):
        # state is should have:
        # epoch, model, model.state_dict, optimizer.state_dict
        
        # save new checkpoint
        if self.checkpoint_fmt is None:
            self.checkpoint_fmt = 'checkpoint_epoch_%03d.pth'
        where = os.path.join(self.log_dir, self.checkpoint_fmt % epoch_tag)
        torch.save(state, where)

    def log_checkpoint_best(self, state):
        where = os.path.join(self.log_dir, "checkpoint_best.pth")
        torch.save(state, where)


class LossLogger:
    def __init__(self, running_over_last=100):
        self.dict = dict()
        self.deque_size = running_over_last
    
    def log(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.dict:
                self.dict[k] = deque(maxlen=self.deque_size)
            self.dict[k].append(v)

    def get(self):
        out = {}
        for k, v in self.dict.items():
            out[k] = np.mean(v)
        return out

    def stats_over_last_n(self, n=None):
        if n is None:
            n = self.deque_size
        # d_print = []
        dicts = dotdict()
        for (item, value) in self.dict.items():
            items_in = len(value)
            if items_in < n:
                n = items_in
            assert(len(list(value)[-n:]) == n)
            mean = sum(list(value)[-n:]) / float(n)
            dicts[item] = mean
            # d_print.append((item, f"{mean:.3f}"))
        # formatted = f"{[f'{item}: {val}' for (item, val) in d_print]}"
        return dicts
    
