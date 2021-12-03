import random

from augmentations import ElasticTransformations, RandomRotationWithMask

import cv2
import PIL
import torch
import numpy as np
import torchvision
import scipy.ndimage

from os import listdir
from os.path import isfile, join

path_train = r"/content/drive/My Drive/UNODEdatasets/images/images"
path_val = r"/content/drive/MyDrive/UNODEdatasets/images_val"
path_test = r"/content/drive/MyDrive/UNODEdatasets/images_test"

onlyfiles = [f for f in listdir(path_train) if isfile(join(path_train, f))]
onlyfiles_val = [f for f in listdir(path_val) if isfile(join(path_val, f))] 
onlyfiles_test = [f for f in listdir(path_test) if isfile(join(path_test, f))] 

## Lo siguiente solo se hace para el training set que por alguna razón tiene imágenes con otro formato
# quitamos las imágenes que tienen otro formato: 
strange_train_idx = ([2052,
                        2058,
                        2071,
                        2076,
                        2088,
                        2089])
strange_train_idx = strange_train_idx[::-1]
for idx in strange_train_idx:
    onlyfiles.pop(idx)

cv2.setNumThreads(0)

class GLaSDataLoader2(object):
    def __init__(self, patch_size, dataset_repeat=1, images=np.arange(0, 70), validation=False):
        self.image_fname1 = "/content/drive/My Drive/UNODEdatasets/images/images/"
        self.image_fname2 = "/content/drive/My Drive/UNODEdatasets/label/label/"
        self.images = images

        self.patch_size = patch_size
        self.repeat = dataset_repeat
        self.validation = validation

    def __getitem__(self, index):
        image, mask = self.index_to_filename(index)
        image, mask = self.open_and_resize(image, mask)
        mask = mask[...,0].squeeze()/255
        image, mask = torch.tensor(image).T/255, np.array([mask.T, mask.T])
        mask = torch.tensor(mask)
        return image, (mask > 0.5).float()

    def index_to_filename(self, index):
        """Helper function to retrieve filenames from index"""
        index_img = index // self.repeat
        index_img = self.images[index_img]
        #index_str = str(index_img.item() + 1)

        image = self.image_fname1 + onlyfiles[index_img]
        mask = self.image_fname2 + onlyfiles[index_img]
        return image, mask

    def open_and_resize(self, image, mask):
        """Helper function to pad smaller image to the correct size"""
        image = PIL.Image.open(image)
        mask = PIL.Image.open(mask)

        # ratio = (256 / 512)
        # new_size = (int(round(image.size[0] / ratio)),
        #             int(round(image.size[1] / ratio)))
        new_size = self.patch_size


        image = image.resize(new_size)
        mask = mask.resize(new_size)

        image = np.array(image)
        mask = np.array(mask)
        return image, mask

    def __len__(self):
        return len(self.images) * self.repeat
      
      
class GLaSDataLoader3(object):
    def __init__(self, patch_size, dataset_repeat=1, images=np.arange(0, 70), validation=False):
        self.image_fname1 = "/content/drive/MyDrive/UNODEdatasets/images_val/"
        self.image_fname2 = "/content/drive/MyDrive/UNODEdatasets/label_val/"
        self.images = images

        self.patch_size = patch_size
        self.repeat = dataset_repeat
        self.validation = validation

    def __getitem__(self, index):
        image, mask = self.index_to_filename(index)
        image, mask = self.open_and_resize(image, mask)
        mask = mask[...,0].squeeze()/255
        image, mask = torch.tensor(image).T/255, np.array([mask.T, mask.T])
        mask = torch.tensor(mask)
        return image, (mask >  0.05).float()

    def index_to_filename(self, index):
        """Helper function to retrieve filenames from index"""
        index_img = index // self.repeat
        index_img = self.images[index_img]
        #index_str = str(index_img.item() + 1)

        image = self.image_fname1 + onlyfiles_val[index_img]
        mask = self.image_fname2 + onlyfiles_val[index_img]
        return image, mask

    def open_and_resize(self, image, mask):
        """Helper function to pad smaller image to the correct size"""
        image = PIL.Image.open(image)
        mask = PIL.Image.open(mask)

        # ratio = (256 / 512)
        # new_size = (int(round(image.size[0] / ratio)),
        #             int(round(image.size[1] / ratio)))
        new_size = self.patch_size


        image = image.resize(new_size)
        mask = mask.resize(new_size)

        image = np.array(image, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)
        return image, mask

    def __len__(self):
        return len(self.images) * self.repeat

class GLaSDataLoader4(object):
    def __init__(self, patch_size, dataset_repeat=1, images=np.arange(0, 70), validation=False):
        self.image_fname1 = "/content/drive/MyDrive/UNODEdatasets/images_test/"
        self.image_fname2 = "/content/drive/MyDrive/UNODEdatasets/label_test/"
        self.images = images

        self.patch_size = patch_size
        self.repeat = dataset_repeat
        self.validation = validation

    def __getitem__(self, index):
        image, mask = self.index_to_filename(index)
        image, mask = self.open_and_resize(image, mask)
        mask = mask[...,0].squeeze()/255
        image, mask = torch.tensor(image).T/255, np.array([mask.T, mask.T])
        mask = torch.tensor(mask)
        return image, (mask > 0.05).float()

    def index_to_filename(self, index):
        """Helper function to retrieve filenames from index"""
        index_img = index // self.repeat
        index_img = self.images[index_img]
        #index_str = str(index_img.item() + 1)

        image = self.image_fname1 + onlyfiles_test[index_img]
        mask = self.image_fname2 + onlyfiles_test[index_img]
        return image, mask

    def open_and_resize(self, image, mask):
        """Helper function to pad smaller image to the correct size"""
        image = PIL.Image.open(image)
        mask = PIL.Image.open(mask)

        # ratio = (256 / 512)
        # new_size = (int(round(image.size[0] / ratio)),
        #             int(round(image.size[1] / ratio)))
        new_size = self.patch_size


        image = image.resize(new_size)
        mask = mask.resize(new_size)

        image = np.array(image, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)
        return image, mask

    def __len__(self):
        return len(self.images) * self.repeat

