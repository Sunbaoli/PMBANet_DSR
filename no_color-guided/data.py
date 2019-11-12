from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor

from dataset import DatasetFromFolderEval, DatasetFromFolder, DatasetFromFolder_slice

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform():
    return Compose([
        #CenterCrop(crop_size),
        #Resize(crop_size // upscale_factor),
        ToTensor(),
    ])

def target_transform():
    return Compose([
        #CenterCrop(crop_size),
        ToTensor(),
    ])
def target1_transform():
    return Compose([
        #CenterCrop(crop_size),
        ToTensor(),
    ])
def target2_transform():
    return Compose([
        #CenterCrop(crop_size),
        ToTensor(),
    ])
def target3_transform():
    return Compose([
        #CenterCrop(crop_size),
        ToTensor(),
    ])
def target4_transform():
    return Compose([
        #CenterCrop(crop_size),
        ToTensor(),
    ])
def target5_transform():
    return Compose([
        #CenterCrop(crop_size),
        ToTensor(),
    ])

def get_training_set(data_dir, dataset, hr, upscale_factor, patch_size, data_augmentation):
    hr_dir = join(data_dir, hr)
    lr_dir = join(data_dir, dataset)
    #crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(hr_dir, lr_dir,patch_size, upscale_factor, dataset, data_augmentation,
                             input_transform=input_transform(),
                             target_transform=target_transform())
def get_training_set_slice(data_dir, dataset, hr, hr1, hr2, hr3, hr4, hr5, upscale_factor, patch_size, data_augmentation):
    hr_dir = join(data_dir, hr)
    hr1_dir = join(data_dir, hr1)
    hr2_dir = join(data_dir, hr2)
    hr3_dir = join(data_dir, hr3)
    hr4_dir = join(data_dir, hr4)
    hr5_dir = join(data_dir, hr5)
    lr_dir = join(data_dir, dataset)
    #crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder_slice(lr_dir, hr_dir, hr1_dir, hr2_dir, hr3_dir, hr4_dir, hr5_dir, patch_size, upscale_factor, data_augmentation,
                             input_transform=input_transform(),
                             target_transform=target_transform(),
                             target1_transform=target1_transform(),
                             target2_transform=target2_transform(),
                             target3_transform=target3_transform(),
                             target4_transform=target4_transform(),
                             target5_transform=target5_transform())


def get_test_set(data_dir, dataset, hr, upscale_factor,patch_size):
    hr_dir = join(data_dir, hr)
    lr_dir = join(data_dir, dataset)
    #crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(hr_dir, lr_dir,patch_size, upscale_factor, dataset, data_augmentation=False,
                             input_transform=input_transform(),
                             target_transform=target_transform())

def get_eval_set(lr_dir):
    return DatasetFromFolderEval(lr_dir,
                             input_transform=input_transform(),
                             target_transform=target_transform())

