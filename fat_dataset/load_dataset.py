import collections
import os
import io
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

Context = collections.namedtuple('Context', ['frames', 'cameras'])
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])


class Fat(Dataset):
    """
    batch_size: number of the training batch
    m_size: number of images/vps in each query
    total: number of data chunk in /fat-torch for training or testing
    By default, we have 200 sample in each chunk, such that we divide them
    into 20 batches and 10 query for each batch

    """
    def __init__(self, root_dir, phase):
        self.root_dir = root_dir
        self.phase = phase  # train or test
        self.start = 0
        self.count = 0

        self.total = len(os.listdir(os.path.join(self.root_dir, self.phase))) / 3

        if phase == "train":
            self.batch_size = 20
            self.m_size = 10
        else:
            self.batch_size = 1
            self.m_size = 1

        self.depth_imgs = None
        self.rgb_imgs = None
        self.viewpoints = None

    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir, self.phase))) / 3

    def __getitem__(self, idx):
        depth_path = os.path.join(self.root_dir, self.phase, "depth_{}".format(idx)+".npy")
        rgb_path = os.path.join(self.root_dir, self.phase, "rgb_{}".format(idx)+".npy")
        vp_path = os.path.join(self.root_dir, self.phase, "vp_{}".format(idx)+".npy")

        depth_data = np.load(depth_path)
        rgb_data = np.load(rgb_path)
        vp_data = np.load(vp_path)

        depth_imgs = torch.from_numpy(depth_data)
        rgb_imgs = torch.from_numpy(rgb_data)
        viewpoints = torch.from_numpy(vp_data)
        viewpoints = viewpoints.view(-1, 7)

        return depth_imgs, rgb_imgs, viewpoints

    def load_new(self):
        if self.start >= self.total:
            self.start = 0

        idx = self.start
        depth_path = os.path.join(self.root_dir, self.phase, "depth_{}".format(idx) + ".npy")
        rgb_path = os.path.join(self.root_dir, self.phase, "rgb_{}".format(idx) + ".npy")
        vp_path = os.path.join(self.root_dir, self.phase, "vp_{}".format(idx) + ".npy")

        self.start += 1

        depth_data = np.load(depth_path)
        rgb_data = np.load(rgb_path)
        vp_data = np.load(vp_path)

        self.depth_imgs = torch.from_numpy(depth_data).permute(0, 3, 1, 2)
        self.rgb_imgs = torch.from_numpy(rgb_data).permute(0, 3, 1, 2)
        self.viewpoints = torch.from_numpy(vp_data).view(-1, 7)

    def get_batch(self):
        depth_temp = None
        rgb_temp = None
        vp_temp = None
        for b in range(self.batch_size):
            ms = b * self.m_size
            me = (b + 1) * self.m_size
            d_temp = self.depth_imgs[ms:me].unsqueeze(0)
            r_temp = self.rgb_imgs[ms:me].unsqueeze(0)
            v_temp = self.viewpoints[ms:me].unsqueeze(0)
            if b == 0:
                depth_temp = d_temp
                rgb_temp = r_temp
                vp_temp = v_temp
            else:
                depth_temp = torch.cat([depth_temp, d_temp], 0)
                rgb_temp = torch.cat([rgb_temp, r_temp], 0)
                vp_temp = torch.cat([vp_temp, v_temp], 0)
        return depth_temp, rgb_temp, vp_temp
