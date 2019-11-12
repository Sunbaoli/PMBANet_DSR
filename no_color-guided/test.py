from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_eval_set
from functools import reduce
import scipy.io as sio
import time
import cv2

from pmpanet_x2 import Net as PMBAX2
from pmpanet_x4 import Net as PMBAX4
from pmpanet_x8 import Net as PMBAX8
from pmpanet_x16 import Net as PMBAX16

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=4, type=float, help='number of gpu')
parser.add_argument('--gpu', type=int, default=0, help='the number of using gpu')
parser.add_argument('--input_dir', type=str, default='./data/')
parser.add_argument('--output', default='Results/', help='Location to save checkpoint models')
parser.add_argument('--test_dataset', type=str, default='testx8/')
parser.add_argument('--model_type', type=str, default='PMBAX8')
parser.add_argument('--model', default="./weights/rb2simple/train_all_depth_x8/Server2RB2Simplet_epoch_119.pth", help='sr pretrained base model')
opt = parser.parse_args()

gpus_list=range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
test_set = get_eval_set(os.path.join(opt.input_dir,opt.test_dataset))
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
if opt.model_type == 'PMBAX8':
    model = PMBAX8(num_channels=1, base_filter=64,  feat = 256, num_stages=3, scale_factor=opt.upscale_factor) ##For NTIRE2018
else:
    model = PMBAX8(num_channels=1, base_filter=64,  feat = 256, num_stages=5, scale_factor=opt.upscale_factor) ###D-DBPN
#####

if os.path.exists(opt.model):
    #model= torch.load(opt.model, map_location=lambda storage, loc: storage)
    model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
    print('Pre-trained SR model is loaded.<---------------------------->')

if cuda:
    model = model.cuda(gpus_list[opt.gpu])

def eval():
    model.eval()

    torch.set_grad_enabled(False)

    for batch in testing_data_loader:
        input, name = Variable(batch[0],volatile=True), batch[1]
          
        if cuda:
            input = input.cuda(gpus_list[opt.gpu])

        t0 = time.time()

        prediction = model(input)

        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
        save_img(prediction.cpu().data, name[0])

def save_img(img, img_name):

    save_img = img.squeeze().clamp(0, 1).numpy()

    save_dir=os.path.join(opt.output,opt.test_dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_fn = save_dir +'/'+ img_name
    cv2.imwrite(save_fn,save_img*255)
##Eval Start!!!!
eval()
