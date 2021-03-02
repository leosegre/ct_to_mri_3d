import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchio as tio
import numpy as np
import nibabel as nib
import torch.nn.functional as nnf
import torch
import matplotlib.pyplot as plt


def save3Dimage(img3d, img_shape, path):
        img3d = torch.squeeze(img3d, 0)
        img_shape = img3d.shape

        plt.subplot(2, 2, 1)
        plt.imshow(img3d[:, :, img_shape[2]//2], cmap="gray")
        # a1.set_aspect(ax_aspect)

        plt.subplot(2, 2, 2)
        plt.imshow(img3d[:, img_shape[1]//2, :], cmap="gray")
        # a2.set_aspect(sag_aspect)

        plt.subplot(2, 2, 3)
        plt.imshow(img3d[img_shape[0]//2, :, :].T, cmap="gray")
        # a3.set_aspect(cor_aspect)

        plt.savefig(path)

def save3Dimage_numpy(img3d, img_shape, path):

        plt.subplot(2, 2, 1)
        plt.imshow(img3d[:, :, img_shape[2]//2], cmap="gray")
        # a1.set_aspect(ax_aspect)

        plt.subplot(2, 2, 2)
        plt.imshow(img3d[:, img_shape[1]//2, :], cmap="gray")
        # a2.set_aspect(sag_aspect)

        plt.subplot(2, 2, 3)
        plt.imshow(img3d[img_shape[0]//2, :, :].T, cmap="gray")
        # a3.set_aspect(cor_aspect)

        plt.savefig(path)


class DataLoaderDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.
    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_ct = os.path.join(opt.dataroot, opt.phase + 'ct')  # create a path '/path/to/data/ct'
        self.dir_mr = os.path.join(opt.dataroot, opt.phase + 'mr')  # create a path '/path/to/data/mr'

        self.ct_paths = sorted(make_dataset(self.dir_ct, opt.max_dataset_size))   # load images from '/path/to/data/ct'
        self.mr_paths = sorted(make_dataset(self.dir_mr, opt.max_dataset_size))    # load images from '/path/to/data/mr'
        self.ct_size = len(self.ct_paths)  # get the size of dataset ct
        self.mr_size = len(self.mr_paths)  # get the size of dataset mr
        mrtoct = self.opt.direction == 'mrtoct'
        input_nc = self.opt.output_nc if mrtoct else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if mrtoct else self.opt.output_nc      # get the number of channels of output image
        self.transform_ct = get_transform(self.opt)
        self.transform_mr = get_transform(self.opt)


    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains ct, mr, ct_paths and mr_paths
            ct (tensor)       -- an image in the input domain
            mr (tensor)       -- its corresponding image in the target domain
            ct_paths (str)    -- image paths
            mr_paths (str)    -- image paths
        """
        ct_path = self.ct_paths[index % self.ct_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_mr = index % self.mr_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_mr = random.randint(0, self.mr_size - 1)
        mr_path = self.mr_paths[index_mr]
        num_of_slices = self.opt.batch_size
        ct_img_ = tio.ScalarImage(ct_path).data
        # ct_img_ = ct_img_[:,:,:,np.round(np.linspace(0, ct_img_.shape[3]-1, num_of_slices)).astype(int)]
        # ct_img = torch.unsqueeze(ct_img_, 0).to(dtype=torch.float32)
        # z_start = random.randint(0, ct_img.shape[4] - spacing * num_of_slices)
        # ct_img  = nnf.interpolate(ct_img[:,:,z_start:z_start + num_of_slices * spacing], size=(256, 256, num_of_slices), mode='trilinear', align_corners=False)
        # ct_img  = nnf.interpolate(ct_img, size=(256, 256, num_of_slices), mode='trilinear', align_corners=False)
        # ct_img = torch.squeeze(ct_img, 0)

        mr_img_ = tio.ScalarImage(mr_path).data
        # mr_img_ = mr_img_[:,:,:,np.round(np.linspace(0, mr_img_.shape[3]-1, num_of_slices)).astype(int)]
        # mr_img = torch.unsqueeze(mr_img_, 0).to(dtype=torch.float32)
        # z_start = random.randint(0, mr_img.shape[4] - spacing * num_of_slices)
        # mr_img = nnf.interpolate(mr_img[:,:,z_start:z_start + num_of_slices * spacing], size=(256, 256, num_of_slices), mode='trilinear', align_corners=False)
        # mr_img = nnf.interpolate(mr_img, size=(256, 256, num_of_slices), mode='trilinear', align_corners=False)
        # mr_img = torch.squeeze(mr_img, 0)
        # save3Dimage(ct_img_, ct_img_.shape, 'images_test/ct_before_transform.png')
        # save3Dimage(mr_img_, mr_img_.shape, 'images_test/mr_before_transform.png')

        # apply image transformation
        # ------------------------------------------------
        ct = self.transform_ct(ct_img_)
        mr = self.transform_mr(mr_img_)

        # ------------------------------------------------
        return {'ct': ct, 'mr': mr, 'ct_paths': ct_path, 'mr_paths': mr_path}

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.ct_size, self.mr_size)