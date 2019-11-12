import torch
import math

class DenseBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation='relu', norm='batch'):
        super(DenseBlock, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

# class AvgpoolingBlock(torch.nn.Module):
#     def __init__(self, kernel_size=3, stride=1, padding=1, dilation, return_indices, bias=True, ceil_mode, activation='prelu', norm=None):
#         super(AvgpoolingBlock, self).__init__()
#         self.avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding, dilation=1, return_indices=False, ceil_mode=False)
#         #self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

#         self.norm = norm
#         if self.norm =='batch':
#             self.bn = torch.nn.BatchNorm2d(output_size)
#         elif self.norm == 'instance':
#             self.bn = torch.nn.InstanceNorm2d(output_size)

#         self.activation = activation
#         if self.activation == 'relu':
#             self.act = torch.nn.ReLU(True)
#         elif self.activation == 'prelu':
#             self.act = torch.nn.PReLU()
#         elif self.activation == 'lrelu':
#             self.act = torch.nn.LeakyReLU(0.2, True)
#         elif self.activation == 'tanh':
#             self.act = torch.nn.Tanh()
#         elif self.activation == 'sigmoid':
#             self.act = torch.nn.Sigmoid()

#     def forward(self, x):
#         if self.norm is not None:
#             out = self.bn(self.avgpool(x))
#         else:
#             out = self.avgpool(x)

#         if self.activation is not None:
#             return self.act(out)
#         else:
#             return out

# class MaxpoolingBlock(torch.nn.Module):
#     def __init__(self, kernel_size=4, stride=4, padding=0, bias=True, activation='prelu', norm=None):
#         super(MaxpoolingBlock, self).__init__()
#         self.avgpool = torch.nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, return_indices=False, ceil_mode=False)
#         #self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

#         self.norm = norm
#         if self.norm =='batch':
#             self.bn = torch.nn.BatchNorm2d(output_size)
#         elif self.norm == 'instance':
#             self.bn = torch.nn.InstanceNorm2d(output_size)

#         self.activation = activation
#         if self.activation == 'relu':
#             self.act = torch.nn.ReLU(True)
#         elif self.activation == 'prelu':
#             self.act = torch.nn.PReLU()
#         elif self.activation == 'lrelu':
#             self.act = torch.nn.LeakyReLU(0.2, True)
#         elif self.activation == 'tanh':
#             self.act = torch.nn.Tanh()
#         elif self.activation == 'sigmoid':
#             self.act = torch.nn.Sigmoid()

#     def forward(self, x):
#         if self.norm is not None:
#             out = self.bn(self.avgpool(x))
#         else:
#             out = self.avgpool(x)

#         if self.activation is not None:
#             return self.act(out)
#         else:
#             return out

class DilaConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, dilation=2, bias=True, activation='prelu', norm=None):
        super(DilaConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding,  dilation, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()


    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)
        return out

class LA_attentionBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(LA_attentionBlock, self).__init__()

        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)

        self.up_1 = DeconvBlock(num_filter, num_filter , kernel_size, stride, padding, activation='prelu', norm=None)

        self.act_1 = torch.nn.ReLU(True)       

    def forward(self, x):

        p1 = self.avgpool_1(x)
        #print(p1.shape, '111111111111')
        l0 = self.up_1(p1)
        #print(l0.shape, 'l0l0l0l0l0l0l0l0l0l0l0l0')
        act1 = self.act_1(x - l0)
        #print(act1.shape, 'acacacacacaacacaacac')
        out_la = x + 0.2*(act1*x)
        #print(out_la.shape, 'oooooooooooooooooooooooooooo')

        return out_la

class LA_attentionBlock2(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(LA_attentionBlock2, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)

        self.avgpool_1 = torch.nn.AvgPool2d(8, 4, 2)

        self.up_1 = DeconvBlock(num_filter, num_filter , kernel_size, stride, padding, activation='prelu', norm=None)

        self.act_1 = torch.nn.ReLU(True)       

    def forward(self, x):

        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        #print(p1.shape, '111111111111')
        l0 = self.up_1(p1)
        #print(l0.shape, 'l0l0l0l0l0l0l0l0l0l0l0l0')
        act1 = self.act_1(x - l0)
        #print(act1.shape, 'acacacacacaacacaacac')
        out_la = x + 0.2*(act1*x)
        #print(out_la.shape, 'oooooooooooooooooooooooooooo')

        return out_la
class GA_attentionBlock(torch.nn.Module):
    def __init__(self, num_filter):
        super(GA_attentionBlock, self).__init__()

        self.g_aver_pooling1 = torch.nn.AdaptiveAvgPool2d(1)

        self.fc1 = torch.nn.Linear(in_features=num_filter, out_features=round(num_filter / 16))

        self.act_1 = torch.nn.ReLU(True)

        self.fc2 = torch.nn.Linear(in_features=round(num_filter / 16), out_features=num_filter)

        self.act_2 = torch.nn.Sigmoid()

        # self.avgpool_1 = torch.nn.AvgPool2d(8, 4, 2)

        # self.up_1 = DeconvBlock(num_filter, num_filter , kernel_size, stride, padding, activation='prelu', norm=None)

        # self.act_1 = torch.nn.ReLU(True)       

    def forward(self, x): 

        x1 = self.g_aver_pooling1(x)
        x1 = x1.view(x1.size(0), -1)
        c1 = self.fc1(x1)
        act1 = self.act_1(c1)
        c2 = self.fc2(act1)
        act2 = self.act_2(c2)
        act2 = act2.view(act2.size(0), act2.size(1), 1, 1)

        y = x*act2

        return y


class GA_res_attentionBlock(torch.nn.Module):
    def __init__(self, num_filter):
        super(GA_res_attentionBlock, self).__init__()

        self.g_aver_pooling1 = torch.nn.AdaptiveAvgPool2d(1)

        self.fc1 = torch.nn.Linear(in_features=num_filter, out_features=round(num_filter / 16))

        self.act_1 = torch.nn.ReLU(True)

        self.fc2 = torch.nn.Linear(in_features=round(num_filter / 16), out_features=num_filter)

        self.act_2 = torch.nn.Sigmoid()

        # self.avgpool_1 = torch.nn.AvgPool2d(8, 4, 2)

        # self.up_1 = DeconvBlock(num_filter, num_filter , kernel_size, stride, padding, activation='prelu', norm=None)

        # self.act_1 = torch.nn.ReLU(True)       

    def forward(self, x): 

        x1 = self.g_aver_pooling1(x)
        x1 = x1.view(x1.size(0), -1)
        c1 = self.fc1(x1)
        act1 = self.act_1(c1)
        c2 = self.fc2(act1)
        act2 = self.act_2(c2)
        act2 = act2.view(act2.size(0), act2.size(1), 1, 1)

        y = x + x*act2

        return y




#####2019.07.22
class MultiViewBlock1(torch.nn.Module):
    def __init__(self, num_filter, kernel_size, stride, padding, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock1, self).__init__()


        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(2*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(3*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(4*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(5*64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4),1)
        #print(concat_p1.shape, '111111111111111111')
        x_prior1_2 = self.dilaconv1_2(concat_p1)
        #print(x_prior1_2.shape, 'cccccccccccccccccccccccccccccc')
        
        h_prior1 = self.direct_up1(x_prior1_2)
        #out = self.output_conv1(h_prior1)


        return h_prior1

class MultiViewBlock2(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock2, self).__init__()


        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(3*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(4*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(5*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(6*64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4),1)
        x_prior1_2 = self.dilaconv1_2(concat_p1)
       
        h_prior1 = self.direct_up1(x_prior1_2)
        #out = self.output_conv1(h_prior1)


        return h_prior1

class MultiViewBlock3(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock3, self).__init__()


        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(4*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(5*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(6*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(7*64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4),1)
        x_prior1_2 = self.dilaconv1_2(concat_p1)
       
        h_prior1 = self.direct_up1(x_prior1_2)


        return h_prior1
class MultiViewBlock4(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock4, self).__init__()


        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(5*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(6*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(7*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(8*64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4),1)
        x_prior1_2 = self.dilaconv1_2(concat_p1)
       
        h_prior1 = self.direct_up1(x_prior1_2)


        return h_prior1

class MultiViewBlock5(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock5, self).__init__()


        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(6*64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(7*64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(8*64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(9*64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        #self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4),1)
        x_prior1_2 = self.dilaconv1_2(concat_p1)
       
        h_prior1 = self.direct_up1(x_prior1_2)


        return h_prior1
        
class FeedbackBlock1(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(FeedbackBlock1, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter , 8, 4, 2, activation='prelu', norm=None)
        self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        
        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        act1 = self.act_1(x - l00)
        out_la = x + 0.1*(act1*x)

        h0 = self.up_conv1(out_la)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - out_la)
        return h1 + h0


class FeedbackBlock2(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(FeedbackBlock2, self).__init__()
        self.down1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter , 8, 4, 2, activation='prelu', norm=None)
        self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        
        x = self.down1(x)
        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        act1 = self.act_1(x - l00)
        out_la = x + 0.1*(act1*x)

        h0 = self.up_conv1(out_la)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - out_la)
        return h1 + h0


class FeedbackBlockv2_1(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(FeedbackBlockv2_1, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter , 8, 4, 2, activation='prelu', norm=None)
        self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.act_2 = torch.nn.ReLU(True)

    def forward(self, x):
        
        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        act1 = self.act_1(x - l00)
        out_la = x + 0.1*(act1*x)

        h0 = self.up_conv1(out_la)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - out_la)
        act2 = self.act_2(h1)

        h3 = h0 + 0.2*(act2*h0)

        return h3



class FeedbackBlockv2_2(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(FeedbackBlockv2_2, self).__init__()
        self.down1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter , 8, 4, 2, activation='prelu', norm=None)
        self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.act_2 = torch.nn.ReLU(True)

    def forward(self, x):
        
        x = self.down1(x)
        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        act1 = self.act_1(x - l00)
        out_la = x + 0.1*(act1*x)

        h0 = self.up_conv1(out_la)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - out_la)
        act2 = self.act_2(h1)

        h3 = h0 +0.2*(act2*h0)

        return h3
class channel_attentionBlock(torch.nn.Module):
    def __init__(self, num_filter):
        super(channel_attentionBlock, self).__init__()

        self.g_aver_pooling1 = torch.nn.AdaptiveAvgPool2d(1)

        self.fc1 = torch.nn.Linear(in_features=num_filter, out_features=round(num_filter / 16))

        self.act_1 = torch.nn.ReLU(True)

        self.fc2 = torch.nn.Linear(in_features=round(num_filter / 16), out_features=num_filter)

        self.act_2 = torch.nn.Sigmoid()

        # self.avgpool_1 = torch.nn.AvgPool2d(8, 4, 2)

        # self.up_1 = DeconvBlock(num_filter, num_filter , kernel_size, stride, padding, activation='prelu', norm=None)

        # self.act_1 = torch.nn.ReLU(True)       

    def forward(self, x): 

        x1 = self.g_aver_pooling1(x)
        x1 = x1.view(x1.size(0), -1)
        c1 = self.fc1(x1)
        act1 = self.act_1(c1)
        c2 = self.fc2(act1)
        act2 = self.act_2(c2)
        act2 = act2.view(act2.size(0), act2.size(1), 1, 1)

        y = x + x*act2

        return y

class PDPANETFeedbackBlock1(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(PDPANETFeedbackBlock1, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        
        x = self.conv1(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0

class PDPANETFeedbackBlock2(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(PDPANETFeedbackBlock2, self).__init__()
        self.down1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.conv1(x1)
        h0 = self.up_conv1(x2)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x2)
        return h1 + h0


class PDPANETAttentionBlock1(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(PDPANETAttentionBlock1, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter , 8, 4, 2, activation='prelu', norm=None)
        self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        
        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        act1 = self.act_1(x - l00)
        out_la = x + 0.1*(act1*x)

        h0 = self.up_conv1(out_la)

        return h0


class PDPANETAttentionBlock2(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(PDPANETAttentionBlock2, self).__init__()
        self.down1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter , 8, 4, 2, activation='prelu', norm=None)
        self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        
        x = self.down1(x)
        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        act1 = self.act_1(x - l00)
        out_la = x + 0.1*(act1*x)

        h0 = self.up_conv1(out_la)

        return h0


class RDBBlock(torch.nn.Module):
    def __init__(self, in_filter, num_filter, bias=True, activation='prelu', norm=None):
        super(RDBBlock, self).__init__()
        self.c1 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c2 = ConvBlock(2*in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c3 = ConvBlock(3*in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c4 = ConvBlock(4*in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c5 = ConvBlock(5*in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c6 = ConvBlock(6*in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.c7 = ConvBlock(7*in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
    def forward(self, x):
        
        x1 = self.c1(x)
        cat1 = torch.cat((x, x1), 1)

        x2 = self.c2(cat1)
        cat2 = torch.cat((cat1, x2), 1)

        x3 = self.c3(cat2)
        cat3 = torch.cat((cat2, x3), 1)

        x4 = self.c4(cat3)
        cat4 = torch.cat((cat3, x4), 1)
        
        x5 = self.c5(cat4)
        cat5 = torch.cat((cat4, x5), 1)

        x6 = self.c6(cat5)
        cat6 = torch.cat((cat5, x6), 1)

        x7 = self.c7(cat6)

        out = x + x7

        return out
class EncoderBlock(torch.nn.Module):
    def __init__(self, in_filter, out_filter, num_filter):
        super(EncoderBlock, self).__init__()

        self.conv1_1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv1_2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv1_3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.pool1 = torch.nn.AvgPool2d(2, 2, 0)
        self.conv1_4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')



        self.conv2_1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv2_2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv2_3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.pool2 = torch.nn.AvgPool2d(2, 2, 0)
        self.conv2_4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')

        self.conv3_1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv3_2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv3_3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.pool3 = torch.nn.AvgPool2d(2, 2, 0)
        self.conv3_4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')

        self.conv3_5 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')


        self.down1 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation='prelu', norm='batch')
        self.down2 = ConvBlock(num_filter, num_filter, 8, 4, 2, activation='prelu', norm='batch')
        self.down3 = ConvBlock(num_filter, num_filter, 6, 2, 2, activation='prelu', norm='batch')


        self.conv4_1 = ConvBlock(3*num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv4_2 = ConvBlock(num_filter, out_filter, 1, 1, 0, activation='prelu', norm='batch')
    

    def forward(self, x): 
 
        c1_1 = self.conv1_1(x)  #256
        c1_2 = self.conv1_2(c1_1)
        c1_3 = self.conv1_3(c1_2)
        s1 = torch.cat((c1_1, c1_3),1)
        c1_4 = self.conv1_4(s1)
        p1 = self.pool1(c1_4)# 128

        c2_1 = self.conv2_1(p1) #128
        c2_2 = self.conv2_2(c2_1)
        c2_3 = self.conv2_3(c2_2)
        s2 = torch.cat((p1, c2_3),1)
        c2_4 = self.conv2_4(s2)
        p2 = self.pool2(c2_4)#64

        c3_1 = self.conv3_1(p2)#64
        c3_2 = self.conv3_2(c3_1)
        c3_3 = self.conv3_3(c3_2)
        s3 = torch.cat((p2, c3_3), 1)
        c3_4 = self.conv3_4(s3)
        p3 = self.pool3(c3_4)#32
        c3_5 = self.conv3_5(p3)#32

        e1 = self.down1(c1_4)
        e2 = self.down2(c2_4)

        s4 = torch.cat((c3_5, e1, e2), 1)

        out1 = self.conv4_1(s4)
        out = self.conv4_2(out1)
        return out


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_filter, out_filter, num_filter):
        super(DecoderBlock, self).__init__()

        self.conv1_1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv1_2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv1_3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.up1 = DeconvBlock(num_filter, num_filter , 6, 2, 2, activation='prelu', norm=None)

        self.conv2_1 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv2_2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.up2 = DeconvBlock(num_filter, num_filter , 6, 2, 2, activation='prelu', norm=None)

        self.conv3_1 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv3_2 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.up3 = DeconvBlock(num_filter, num_filter , 6, 2, 2, activation='prelu', norm=None)

        self.conv4_1 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv4_2 = ConvBlock(num_filter, out_filter, 1, 1, 0, activation='prelu', norm='batch')
    

    def forward(self, x): 
 
        c1_1 = self.conv1_1(x)
        c1_2 = self.conv1_2(c1_1)
        c1_3 = self.conv1_3(c1_2)
        hr1 = self.up1(c1_3)



        c2_1 = self.conv2_1(hr1)
        c2_2 = self.conv2_2(c2_1)
        hr2 = self.up2(c2_2)

        c3_1 = self.conv3_1(hr2)
        c3_2 = self.conv3_2(c3_1)
        hr3 = self.up3(c3_2)


        out1 = self.conv4_1(hr3)
        out = self.conv4_2(out1) 
        return out

class GeneratorBlock(torch.nn.Module):
    def __init__(self, in_filter, out_filter, num_filter):
        super(GeneratorBlock, self).__init__()

        self.conv1_1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv1_2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv1_3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.pool1 = torch.nn.AvgPool2d(2, 2, 0)
        self.conv1_4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')



        self.conv2_1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv2_2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv2_3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.pool2 = torch.nn.AvgPool2d(2, 2, 0)
        self.conv2_4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')

        self.conv3_1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.conv3_2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv3_3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')
        self.pool3 = torch.nn.AvgPool2d(2, 2, 0)
        self.conv3_4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')

        self.conv3_5 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm='batch')


        self.deconv1 = DeconvBlock(num_filter, num_filter, 6, 2, 2, activation='prelu', norm='batch')
        self.deconv2 = DeconvBlock(2*num_filter, num_filter, 6, 2, 2, activation='prelu', norm='batch')
        self.deconv3 = DeconvBlock(2*num_filter, num_filter, 6, 2, 2, activation='prelu', norm='batch')
        self.deconv4 = DeconvBlock(num_filter, num_filter, 6, 2, 2, activation='prelu', norm='batch')

        self.conv4_1 = ConvBlock(2*num_filter, num_filter, 3, 1, 1, activation='prelu', norm='batch')
        self.conv4_2 = ConvBlock(num_filter, out_filter, 1, 1, 0, activation='prelu', norm='batch')
    

    def forward(self, x): 
 
        c1_1 = self.conv1_1(x)  #256
        c1_2 = self.conv1_2(c1_1)
        c1_3 = self.conv1_3(c1_2)
        s1 = torch.cat((c1_1, c1_3),1)
        c1_4 = self.conv1_4(s1)
        p1 = self.pool1(c1_4)# 128

        c2_1 = self.conv2_1(p1) #128
        c2_2 = self.conv2_2(c2_1)
        c2_3 = self.conv2_3(c2_2)
        s2 = torch.cat((p1, c2_3),1)
        c2_4 = self.conv2_4(s2)
        p2 = self.pool2(c2_4)#64

        c3_1 = self.conv3_1(p2)#64
        c3_2 = self.conv3_2(c3_1)
        c3_3 = self.conv3_3(c3_2)
        s3 = torch.cat((p2, c3_3), 1)
        c3_4 = self.conv3_4(s3)
        p3 = self.pool3(c3_4)#32
        c3_5 = self.conv3_5(p3)#32

        dc1 = self.deconv1(c3_5)#64
        u1 = torch.cat((dc1, c3_4), 1)#64
        dc2 = self.deconv2(u1)#128
        u2 = torch.cat((dc2, c2_4), 1)#128
        dc3 = self.deconv3(u2)#256
        u3 = torch.cat((dc3, c1_4), 1)#256

        out1 = self.conv4_1(u3)#256
        out = self.conv4_2(out1)#256

        return out


class RB2DBPN(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(RB2DBPN, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        #self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        #self.up_1 = DeconvBlock(num_filter, num_filter , 8, 4, 2, activation='prelu', norm=None)
        #self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        
        x1 = self.conv1(x)
        # p1 = self.avgpool_1(x)
        # l00 = self.up_1(p1)
        # act1 = self.act_1(x - l00)
        # out_la = x + 0.1*(act1*x)

        h0 = self.up_conv1(x1)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x1)
        return h1 + h0


class RB2DBPN2(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(RB2DBPN2, self).__init__()
        self.down1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        #self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        #self.up_1 = DeconvBlock(num_filter, num_filter , 8, 4, 2, activation='prelu', norm=None)
        #self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        
        x1 = self.down1(x)
        x2 = self.conv1(x1)
        #p1 = self.avgpool_1(x)
        #l00 = self.up_1(p1)
        #act1 = self.act_1(x - l00)
        #out_la = x + 0.1*(act1*x)

        h0 = self.up_conv1(x2)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x2)
        return h1 + h0

class SimpleLR2HR(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size, stride, padding, bias=True, activation='prelu', norm=None):
        super(SimpleLR2HR, self).__init__()
        #self.down1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv2 = ConvBlock(in_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.conv4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.conv5 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv6 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        #self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        #self.up_1 = DeconvBlock(num_filter, num_filter , 8, 4, 2, activation='prelu', norm=None)
        #self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.conv7 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation, norm=None)
        self.conv8 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation, norm=None)


    def forward(self, x):
        
        #x1 = self.down1(x)
        #print(x.shape, 'xxxxxxxxxxxx')
        x2 = self.conv2(x)
        x3 = self.conv3(x2)
        res1 = torch.cat((x2, x3), 1)
        x4 = self.conv4(res1)
        x5 = self.conv5(x4)
        res2 = torch.cat((x2, x5), 1)
        x6 = self.conv6(res2)

        h0 = self.up_conv1(x6)
        h1 = self.conv7(h0)
        h2 = self.conv8(h1)
        return h2

class SimpleLR2HR2(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size, stride, padding, bias=True, activation='prelu', norm=None):
        super(SimpleLR2HR2, self).__init__()
        self.down1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv2 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv3 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.conv4 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.conv5 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)
        self.conv6 = ConvBlock(2*num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        #self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        #self.up_1 = DeconvBlock(num_filter, num_filter , 8, 4, 2, activation='prelu', norm=None)
        #self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.conv7 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation, norm=None)
        self.conv8 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation, norm=None)


    def forward(self, x):
        
        x1 = self.down1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        res1 = torch.cat((x1, x3), 1)
        x4 = self.conv4(res1)
        x5 = self.conv5(x4)
        res2 = torch.cat((x1, x5), 1)
        x6 = self.conv6(res2)

        h0 = self.up_conv1(x6)
        h1 = self.conv7(h0)
        h2 = self.conv8(h1)
        return h2

class MultiViewBlock_no_dense(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size, stride, padding, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock_no_dense, self).__init__()


        self.dilaconv1 = DilaConvBlock(in_filter, num_filter, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(num_filter, num_filter, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(num_filter, num_filter, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(num_filter, num_filter, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(num_filter, num_filter, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)  

        self.output_conv1 = ConvBlock(num_filter, num_filter, 3, 1, 1, activation='prelu', norm=None)      

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        #concat1 = torch.cat((x, x_prior1),1)
        x_prior2 = self.dilaconv2(x_prior1)
        #concat2 = torch.cat((concat1, x_prior2),1)
        x_prior3 = self.dilaconv3(x_prior2)
        #concat3 = torch.cat((concat2, x_prior3),1)
        x_prior4 = self.dilaconv4(x_prior3)
        #concat_p1 = torch.cat((concat3, x_prior4),1)
        #print(concat_p1.shape, '111111111111111111')
        x_prior1_2 = self.dilaconv1_2(x_prior4)
        #print(x_prior1_2.shape, 'cccccccccccccccccccccccccccccc')
        
        h_prior1 = self.direct_up1(x_prior1_2)
        out = self.output_conv1(h_prior1)
        return out
