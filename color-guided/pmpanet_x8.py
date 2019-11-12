import os
import torch.nn as nn
import torch.optim as optim
from base_networks import *
from torchvision.transforms import *

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, scale_factor):
        super(Net, self).__init__()
        
        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2
        #### 
        elif scale_factor == 16:
          kernel = 20
          stride = 16
          padding = 2
        
        #Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)

        self.feat_color0 = ConvBlock(3, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat_color1 = ConvBlock(feat, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat_color2 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat_color3 = ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)

        #Multi-view prior
        self.m1 = MultiViewBlock1(64, 12, 8, 2)
        self.m2 = MultiViewBlock2(2*64, 12, 8, 2)
        self.m3 = MultiViewBlock3(3*64, 12, 8, 2)
        self.m4 = MultiViewBlock4(4*64, 12, 8, 2)
        #self.m5 = MultiViewBlock5(5*64, 8, 4, 2)

        #Channel_attention
        self.c1 = channel_attentionBlock(64)
        self.c2 = channel_attentionBlock(64)
        self.c3 = channel_attentionBlock(64)
        self.c4 = channel_attentionBlock(64)
        self.c5 = channel_attentionBlock(64)

        #Reconstruction 1
        self.r1_1 = FeedbackBlock1(64, base_filter, kernel, stride, padding)
        self.r1_2 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        self.r1_3 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        self.r1_4 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        self.r1_5 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)

        #Reconstruction 2
        self.r2_1 = FeedbackBlock1(2*64, base_filter, kernel, stride, padding)
        self.r2_2 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        self.r2_3 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        self.r2_4 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        self.r2_5 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)

        #Reconstruction 3
        self.r3_1 = FeedbackBlock1(3*64, base_filter, kernel, stride, padding)
        self.r3_2 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        self.r3_3 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        self.r3_4 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        self.r3_5 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)

        #Reconstruction 4
        # self.r4_1 = FeedbackBlock1(4*64, base_filter, kernel, stride, padding)
        # self.r4_2 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        # self.r4_3 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        # self.r4_4 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        # self.r4_5 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)

        #Reconstruction 5
        # self.r5_1 = FeedbackBlock1(5*64, base_filter, kernel, stride, padding)
        # self.r5_2 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        # self.r5_3 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        # self.r5_4 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        # self.r5_5 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)


        self.down2 = ConvBlock(base_filter, base_filter, kernel, stride, padding, activation='prelu', norm=None)
        self.down3 = ConvBlock(base_filter, base_filter, kernel, stride, padding, activation='prelu', norm=None)
        self.down4 = ConvBlock(base_filter, base_filter, kernel, stride, padding, activation='prelu', norm=None)
        self.down4 = ConvBlock(base_filter, base_filter, kernel, stride, padding, activation='prelu', norm=None)
        #Reconstruction

        self.output_conv1_1 = ConvBlock(4*base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.output_conv2_1 = ConvBlock(4*base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.output_conv3_1 = ConvBlock(4*base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.output_conv4_1 = ConvBlock(4*base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.output_conv5_1 = ConvBlock(4*base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.output_conv = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            
    def forward(self, rgb, depth):


        x = self.feat0(depth)
        x = self.feat1(x)

        c = self.feat_color0(rgb)
        c1 = self.feat_color1(c)
        c3 = self.feat_color3(c1)

 ############1
        mv1 = self.m1(x)
        rb1 = self.r1_1(x)
        rb2 = self.r1_2(rb1)
        rb3 = self.r1_3(rb2)
        rb4 = self.r1_4(rb3)
        concat_h = torch.cat((rb1, rb2, rb3, rb4),1)
        r1 = self.output_conv1_1(concat_h)
        
        d1 = mv1 + r1 + c3
        d1 = self.c1(d1)
##############2
        x2 = self.down2(d1)
        x2 = torch.cat((x, x2),1)
        mv2 = self.m2(x2)
        rb1 = self.r2_1(x2)
        rb2 = self.r2_2(rb1)
        rb3 = self.r2_3(rb2)
        rb4 = self.r2_4(rb3)
        concat_h = torch.cat((rb1, rb2, rb3, rb4),1)
        r2 = self.output_conv2_1(concat_h)

        d2 = mv2 + r2
        d2 = self.c2(d2)
##############3
        x3 = self.down3(d2)
        x3 = torch.cat((x2, x3),1)
        mv3 = self.m3(x3)
        rb1 = self.r3_1(x3)
        rb2 = self.r3_2(rb1)
        rb3 = self.r3_3(rb2)
        rb4 = self.r3_4(rb3)
        concat_h = torch.cat((rb1, rb2, rb3, rb4),1)
        r3 = self.output_conv3_1(concat_h)

        d3 = mv3 + r3
        d3 = self.c3(d3)
        d = self.output_conv(d3)
        return d