# TEED: is a Tiny but Efficient Edge Detection, it comes from the LDC-B3
# with a Slightly modification
# LDC parameters:
# 155665
# TED > 58K

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.AF.Fsmish import smish as Fsmish
from utils.AF.Xsmish import Smish
from utils.img_processing import count_parameters
import cv2
from CoTnet import CoTAttention  # 引入CoT注意力机制

#修改，部分卷积层
from torch import Tensor
class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.forward = self.forward_split_cat


    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x

def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class CoFusion(nn.Module):
    # from LDC

    def __init__(self, in_ch, out_ch):
        super(CoFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3,
                               stride=1, padding=1) # before 64
        self.conv3= nn.Conv2d(32, out_ch, kernel_size=3,
                               stride=1, padding=1)# before 64  instead of 32
        self.relu = nn.Tanh()
        self.norm_layer1 = nn.GroupNorm(4, 32) # before 64

    def forward(self, x):
        # fusecat = torch.cat(x, dim=1)
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = F.softmax(self.conv3(attn), dim=1)
        return ((x * attn).sum(1)).unsqueeze(1)


class CoFusion2(nn.Module):
        # TEDv14-3
    def __init__(self, in_ch, out_ch):
        super(CoFusion2, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3,
                               stride=1, padding=1) # before 64
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=3,
        #                        stride=1, padding=1)# before 64
        self.conv3 = nn.Conv2d(32, out_ch, kernel_size=3,
                               stride=1, padding=1)# before 64  instead of 32
        self.ReLU= nn.Tanh()


    def forward(self, x):
        # fusecat = torch.cat(x, dim=1)
        attn = self.conv1(self.ReLU(x))
        attn = self.conv3(self.ReLU(attn)) # before , )dim=1)

        # return ((fusecat * attn).sum(1)).unsqueeze(1)
        return ((x * attn).sum(1)).unsqueeze(1)


class DoubleFusion(nn.Module):
    # TED fusion before the final edge map prediction
    def __init__(self, in_ch, out_ch):
        print("通道数")
        print(in_ch)

        super(DoubleFusion, self).__init__()
        #修改
        self.DWconv1 = nn.Conv2d(in_ch, in_ch*8, kernel_size=3,
                              stride=1, padding=1, groups=in_ch) # before 64
        # self.PTconv1 = Partial_conv3(in_ch,3)#修改，部分卷积
        #self.DWconv1 = nn.Conv2d(in_ch, in_ch*12, kernel_size=3,
        #                      stride=1, padding=1, groups=in_ch) # before 64
        self.PSconv1 = nn.PixelShuffle(1)
        self.PTconv2 = Partial_conv3(in_ch,3)#修改，部分卷积
        self.DWconv2 = nn.Conv2d(24, 24*1, kernel_size=3,
                              stride=1, padding=1,groups=24)# before 64  instead of 32

        self.AF= nn.Tanh()#XAF() #nn.Tanh()# XAF() #   # Smish()#


    def forward(self, x):
        # fusecat = torch.cat(x, dim=1)

        # 修改
        attn = self.PSconv1(self.DWconv1(self.AF(x))) # #TEED best res TEDv14 [8, 32, 352, 352]

        # attn = self.PSconv1(self.PTconv1(self.AF(x)))#修改，部分卷积
        
        attn2 = self.PSconv1(self.DWconv2(self.AF(attn))) # #TEED best res TEDv14[8, 3, 352, 352]
        
        # attn2 = self.PSconv1(self.PTconv2(self.AF(attn)))#修改，部分卷积

        return Fsmish(((attn2 +attn).sum(1)).unsqueeze(1)) #TED best res


class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()

        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=3, stride=1, padding=2, bias=True)),
        self.add_module('smish1', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=3, stride=1, bias=True))
    def forward(self, x):
        x1, x2 = x

        new_features = super(_DenseLayer, self).forward(Fsmish(x1))  # F.relu()

        return 0.5 * (new_features + x2), x2


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
            input_features = out_features


class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        #修改16 -》32
        self.constant_features = 32
        #self.constant_features = 16

        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads=[0,0,1,3,7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]  # kernel_size-1
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.Tanh())
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x):
        return self.features(x)


class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, use_ac=False):
        super(SingleConvBlock, self).__init__()
        # self.use_bn = use_bs
        self.use_ac=use_ac
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride,
                              bias=True)
        if self.use_ac:
            self.smish = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        if self.use_ac:
            return self.smish(x)
        else:
            return x

class DoubleConvBlock(nn.Module):
    def __init__(self, in_features, mid_features,
                 out_features=None,
                 stride=1,
                 use_act=True):
        super(DoubleConvBlock, self).__init__()

        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features,
                               3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.smish= nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.smish(x)
        x = self.conv2(x)
        if self.use_act:
            x = self.smish(x)
        return x

#修改cnn块，不使用smish,使用其它激活函数
class ReluConvBlock(nn.Module):
    def __init__(self, in_features, mid_features,
                 out_features=None,
                 stride=1,
                 use_act=True):
        super(ReluConvBlock, self).__init__()

        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features,
                               3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.Relu= nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.Relu(x)
        x = self.conv2(x)
        if self.use_act:
            x = self.Relu(x)
        return x
        

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y









#修改主体网络结构
class TED(nn.Module):
    """ Definition of  Tiny and Efficient Edge Detector """
    

    def __init__(self):
        super(TED, self).__init__()
        self.block_lxt = DoubleConvBlock(3, 8, 16)#修改
        # self.block_lxt_se = SEBlock(16)  # 修改1，添加使用注意力的SE块
        # self.block_lxt_se = CBAMBlock(16)  # 修改2，用CBAMBlock替换
        # self.block_lxt = ReluConvBlock(3, 8, 16)#修改，消融实验用，更换激活函数

        #self.block_1 = DoubleConvBlock(3, 16, 16, stride=2,)

        #修改 partial卷积（原）
        self.block_1 = DoubleConvBlock(16, 16, stride=2, )#修改

        self.block_2 = DoubleConvBlock(16, 32, use_act=False)

        self.dblock_3 = _DenseBlock(1, 32, 48) # [128,256,100,100] before (2, 32, 64)


        #修改CoT注意力
        self.coT_attn = CoTAttention(dim=48)  # 在block 3后引入CoT Attention

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # left skip connections, figure in the paper
        self.side_1 = SingleConvBlock(16, 32, 2)
        #修改
        self.side_lxt = SingleConvBlock(16, 16, 2)

        self.pre_dense_3 = SingleConvBlock(32, 48, 1)  # before (32, 64, 1)

        # USNet
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(48, 2) # (32, 64, 1)

        ##修改
        self.block_cat = DoubleFusion(3,3)# EDMFF fusion
        #self.block_cat = DoubleFusion(2, 3)  # TEED fusion

        self.apply(weight_init)

    def slice(self, tensor, slice_shape):
        t_shape = tensor.shape
        img_h, img_w = slice_shape
        if img_w!=t_shape[-1] or img_h!=t_shape[2]:
            new_tensor = F.interpolate(
                tensor, size=(img_h, img_w), mode='nearest',align_corners=False)

        else:
            new_tensor=tensor
        # tensor[..., :height, :width]
        return new_tensor
    def resize_input(self,tensor):
        t_shape = tensor.shape
        if t_shape[2] % 8 != 0 or t_shape[3] % 8 != 0:
            img_w= ((t_shape[3]// 8) + 1) * 8
            img_h = ((t_shape[2] // 8) + 1) * 8
            new_tensor = F.interpolate(
                tensor, size=(img_h, img_w), mode='nearest', align_corners=False)
        else:
            new_tensor = tensor
        return new_tensor

    # Based on BDCN Implementation @ https://github.com/pkuCactus/BDCN
    def crop_bdcn(data1, h, w, crop_h, crop_w):
        _, _, h1, w1 = data1.size()
        assert (h <= h1 and w <= w1)
        data = data1[:, :, crop_h:crop_h + h, crop_w:crop_w + w]
        return data


    def forward(self, x, single_test=False):
        assert x.ndim == 4, x.shape
        
        ##修改3 多尺度双边滤波，修改4 先高斯再双边滤波，修改5 先高斯再双边滤波，自适应参数
        ## 转换张量为numpy数组
        #x_np = x.cpu().numpy().transpose((0, 2, 3, 1))
        ## 对每张图片应用多尺度双边滤波
        #for i in range(x_np.shape[0]):
        ##    x_np[i] = multi_scale_bilateral_filter(x_np[i])
        ##    x_np[i] = combined_filter(x_np[i])
        #    x_np[i] = combined_adaptive_filter(x_np[i])
        ## 转换回张量
        #x = torch.from_numpy(x_np.transpose((0, 3, 1, 2))).to(x.device)
        
        
        ##修改6 转换张量为numpy数组
        #x_np = x.cpu().numpy().transpose((0, 2, 3, 1))
        ## 对每张图片应用细节保持滤波
        #for i in range(x_np.shape[0]):
            ##x_np[i] = detail_preserving_filter(x_np[i])
            ##x_np[i] = multi_scale_detail_preserving_filter(x_np[i])
            #x_np[i] = enhanced_multi_scale_filter(x_np[i])
        ## 转换回张量
        #x = torch.from_numpy(x_np.transpose((0, 3, 1, 2))).to(x.device)
        
        
        
         # supose the image size is 352x352

        #print(x.shape)#[8, 3, 300, 300]

        # 修改
        block_lxt = self.block_lxt(x)  # [8,16,300,300]
        
        ##修改9
        #block_lxt = self.enhanced_danet(block_lxt)
        #block_lxt = self.adaptive_conv(block_lxt)
        
        ##修改8
        #block_lxt = self.enhanced_cbam(block_lxt)
        #block_lxt = self.adaptive_conv(block_lxt)
        
        # block_lxt = self.block_lxt_se(block_lxt)  # 修改1，添加使用注意力的SE块
        # block_lxt = adaptive_bilateral_filter(block_lxt)# 修改2 自适应双边滤波
        
        block_lxt_side = self.side_lxt(block_lxt)  # 16 [8,32,150,150]

        # print("--------8*8 block--------")
        # print(block_lxt.shape)
        # print(block_lxt_side.shape)
        # print("--------8*8 block--------")

        # 修改Block 1
        block_1 = self.block_1(block_lxt) # [8,16,150,150]
        #print(block_1.shape)
        
        block_1_side = self.side_1(block_1) # 16 [8,32,88,88]
        block_1_add = block_lxt_side + block_1
        #print(block_1_add.shape)
        # # Block 1
        # block_1 = self.block_1(x) # [8,16,176,176]
        # block_1_side = self.side_1(block_1) # 16 [8,32,88,88]


        # 修改的Block 2
        block_2 = self.block_2(block_1_add) # 32 # [8,32,176,176]
        # block_2 = self.block_2_se(block_2)  # 修改1，添加使用注意力的SE块
        block_2_down = self.maxpool(block_2) # [8,32,88,88]
        block_2_add = block_2_down + block_1_side # [8,32,75,75]

        # # Block 2
        # block_2 = self.block_2(block_1) # 32 # [8,32,176,176]
        # block_2_down = self.maxpool(block_2) # [8,32,88,88]
        # block_2_add = block_2_down + block_1_side # [8,32,88,88]

        # print("----------block2----------")
        # print(block_2.shape)
        # print(block_2_down.shape)
        # print(block_2_add.shape)
        # print("----------block2----------")

        # # Block 3
        # block_3_pre_dense = self.pre_dense_3(block_2_down) # [8,64,88,88] block 3 L connection
        # block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense]) # [8,64,88,88]

        # 修改Block 3
        block_3_pre_dense = self.pre_dense_3(block_2) # [8,64,88,88] block 3 L connection
        block_3_pre = self.maxpool(block_3_pre_dense)
        # print(block_3_pre_dense.shape)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre]) # [8,64,88,88]
        ## 修改 CoT注意力
        #block_3 = self.coT_attn(block_3)  # 应用CoT Attention

        # print("----------block3----------")
        # print(block_3_pre_dense.shape)
        # print(block_3.shape)
        # print("----------block3----------")

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)

        results = [out_1, out_2, out_3]

        # concatenate multiscale outputs

        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        # print("--------block_cat----------")
        # print(block_cat.shape)
        # print("--------block_cat----------")
        
        
        block_cat = self.block_cat(block_cat)  # Bx1xHxW DoubleFusion

        results.append(block_cat)
        return results


if __name__ == '__main__':
    batch_size = 8
    img_height = 352
    img_width = 352

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda"
    input = torch.rand(batch_size, 3, img_height, img_width).to(device)
    # target = torch.rand(batch_size, 1, img_height, img_width).to(device)
    print(f"input shape: {input.shape}")
    model = TED().to(device)
    output = model(input)
    print(f"output shapes: {[t.shape for t in output]}")

    # for i in range(20000):
    #     print(i)
    #     output = model(input)
    #     loss = nn.MSELoss()(output[-1], target)
    #     loss.backward()
