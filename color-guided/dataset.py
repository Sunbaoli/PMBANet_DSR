import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image
import random
from random import randrange

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):

    #img = Image.open(filepath).convert('RGB')
##############
    img = Image.open(filepath)
    #img = Image.open(filepath)   

    #y, _, _ = img.split()
    return img

def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):
    (c, ih, iw) = img_in.shape
    ####print('input:', ih, iw)
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)
    img_in = img_in[:, iy:iy + ip, ix:ix + ip]
    print('get_patch', img_tar.size(), ty, ty+tp, tx, tx+tp)
    img_tar = img_tar[:, ty:ty + tp, tx:tx + tp]
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}
    
    ####print('after', img_tar.size())
    
    return img_in, img_tar, info_patch

def augment(img_in,img_color, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        ####print('<-------------->', img_tar.size())
        img_in = torch.from_numpy(img_in.numpy()[:, :, ::-1].copy())
        img_color = torch.from_numpy(img_color.numpy()[:, :, ::-1].copy())
        img_tar = torch.from_numpy(img_tar.numpy()[:, :, ::-1].copy())
    
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = torch.from_numpy(img_in.numpy()[:, ::-1, :].copy())
            img_color = torch.from_numpy(img_color.numpy()[:, ::-1, :].copy())
            img_tar = torch.from_numpy(img_tar.numpy()[:, ::-1, :].copy())
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = torch.FloatTensor(np.transpose(img_in.numpy(),(0,2,1)))
            img_color = torch.FloatTensor(np.transpose(img_color.numpy(),(0,2,1)))
            img_tar = torch.FloatTensor(np.transpose(img_tar.numpy(),(0,2,1)))
            info_aug['trans'] = True

    return img_in,img_color,img_tar, info_aug
    
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, lr_dir, rgb_dir, patch_size, upscale_factor, dataset, data_augmentation, input_transform=None, input_rgb_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.lr_dir = lr_dir
        self.rgb_dir = rgb_dir
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.dataset = dataset
        self.input_transform = input_transform
        self.input_rgb_transform = input_rgb_transform
        self.target_transform = target_transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        target = load_img(self.image_filenames[index])

        # print(self.image_filenames[index], target.size)

        _, file = os.path.split(self.image_filenames[index])
        
        ##### print('<==============>', self.dataset)
        if self.dataset == 'DIV2K_train_LR_aug_x8':
            input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x8.png'))
        # elif self.dataset == 'DIV2K_train_LR_aug_x4':
        #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4.png'))
        # elif self.dataset == 'DIV2K_train_LR_aug_x2':
        #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x2.png'))
        # elif self.dataset == 'DIV2K_train_LR_difficult':
        #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4d.png'))
        # elif self.dataset == 'DIV2K_train_LR_mild':
        #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4m.png'))
        # elif self.dataset == 'DIV2K_train_LR_wild':
        #     set=randrange(1, 5)
        #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4w'+str(set)+'.png'))
        # #####
        # elif self.dataset == 'DIV2K_train_LR_bicubicX2/X2/':
        #     ##### print('--------', self.lr_dir, os.path.splitext(file)[0])
        #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x2.png'))
        #     #### print(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x2.png'))
            
        # elif self.dataset == 'DIV2K_train_LR_bicubicX4/X4/':
        #     ##### print('--------', self.lr_dir, os.path.splitext(file)[0])
        #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4.png'))
        #     #### print(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x2.png'))
            
        # elif self.dataset == 'DIV2K_train_LR_unknownX2/X2/':
        #     ##### print('--------', self.lr_dir, os.path.splitext(file)[0])
        #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x2.png'))
        #     #### print(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x2.png'))
        
        # elif self.dataset == 'DIV2K_train_LR_unknownX3/X3/':
        #     ##### print('--------', self.lr_dir, os.path.splitext(file)[0])
        #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x3.png'))
        #     #### print(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x2.png'))
                
        # elif self.dataset == 'DIV2K_train_LR_unknownX4/X4/':
        #     ##### print('--------', self.lr_dir, os.path.splitext(file)[0])
        #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4.png'))
        #     #### print(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x2.png'))
            
        # elif self.dataset == 'DIV2K_train_LR_bicubicX3/X3/':
        #     ##### print('--------', self.lr_dir, os.path.splitext(file)[0])
        #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x3.png'))
        #     #### print(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x2.png'))
        ######################
        elif self.dataset == 'depth_map/data/L_pic128_x2/':
            input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'.png'))
            #print(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'.png'))
        ######################
        #######***
        elif self.dataset == 'train_all_depth_x8/':
        	input_rgb = load_img(os.path.join(self.rgb_dir, os.path.splitext(file)[0]+'.png'))
        	input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'.png'))

        elif self.dataset == 'train_all_depth_x4/':
            input_rgb = load_img(os.path.join(self.rgb_dir, os.path.splitext(file)[0]+'.png'))
            input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'.png'))

        elif self.dataset == 'train_all_depth_x2/':
            input_rgb = load_img(os.path.join(self.rgb_dir, os.path.splitext(file)[0]+'.png'))
            input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'.png'))

        elif self.dataset == 'train_all_depth_x16/':
            input_rgb = load_img(os.path.join(self.rgb_dir, os.path.splitext(file)[0]+'.png'))
            input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'.png'))
        #######***
        
        if self.input_rgb_transform:
        	input_rgb = self.input_rgb_transform(input_rgb)

        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
            #print('target:', target.size())
        # print('target:', target.size())
        
        if self.data_augmentation:
            input,input_rgb,target, _ = augment(input,input_rgb,target)
        return input_rgb, input, target

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderEval(data.Dataset):
    def __init__(self, lr_dir,rgb_dir,input_transform=None,input_rgb_transform=None,target_transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.image_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.input_rgb_transform = input_rgb_transform
        self.target_transform = target_transform
        self.lr_dir = lr_dir
        self.rgb_dir = rgb_dir

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])
        
        if self.lr_dir == './data_x16/test_x16/':
            input_rgb = load_img(os.path.join(self.rgb_dir,os.path.splitext(file)[0]+'.png'))
        # elif self.dataset == 'DIV2K_train_LR_aug_x4':
        #     input = load_img(os.path.join(self.lr_dir,os.path.splitext(file)[0]+'x4.png'))

        if self.input_transform:
            input = self.input_transform(input)
        if self.input_rgb_transform:
        	input_rgb = self.input_rgb_transform(input_rgb)
        return input, input_rgb, file
      
    def __len__(self):
        return len(self.image_filenames)
