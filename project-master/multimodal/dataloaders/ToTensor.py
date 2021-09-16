import torch
import numpy as np


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device=None):
        self.device = device

    def __call__(self, sample):


        # transpose flow into 2 x H x W
        for k in sample.keys():
            if k.startswith('flow'):
                sample[k] = sample[k].transpose((2, 0, 1))

        # convert numpy arrays to pytorch tensors
        new_dict = dict()
        for k, v in sample.items():
            if self.device is None:
                # torch.tensor(v, device = self.device, dtype = torch.float32)
                new_dict[k] = torch.FloatTensor(v)
            else:
                new_dict[k] = torch.from_numpy(v).float()

        return new_dict


class MyToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device="cpu"):
        self.device = device

    def __call__(self, sample):
      
        color_prev = sample["color_prev"].astype(float) / 255.0
        depth_prev = np.expand_dims(sample["depth_prev"].astype(float), axis=2)
        ft_prev = np.expand_dims(sample["ft_prev"].astype(float), axis=1)
        action = np.expand_dims(sample["action"].astype(float), axis=1).transpose((1, 0))

        color = sample["color"].astype(float) / 255.0
        depth = np.expand_dims(sample["depth"].astype(float), axis=2)
        contact = np.array(sample["contact"]).astype(float)

        # convert numpy arrays to pytorch tensors
        return {"color_prev": self.to_tensor(color_prev),
                "depth_prev": self.to_tensor(depth_prev),
                "ft_prev":    self.to_tensor(ft_prev),
                "action":     self.to_tensor(action),
                "color":      self.to_tensor(color),
                "depth":      self.to_tensor(depth),
                "contact":    self.to_tensor(contact)
                }

    @staticmethod
    def to_tensor(data):
        return torch.from_numpy(data).float()
