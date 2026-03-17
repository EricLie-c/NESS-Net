import torch
import torch.nn as nn
import torch.nn.functional as F
from CBAM import CBAMLayer
from .SwinT import SwinTransformer

SEED = 42
torch.manual_seed(SEED)  
torch.cuda.manual_seed(SEED)          
torch.cuda.manual_seed_all(SEED)      
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False     

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels):
        super(SeparableConv2d, self).__init__()
        self.depthwise1 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, bias=True, groups=in_channels)
        self.depthwise2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True, groups=in_channels)
        self.relu = nn.PReLU(in_channels)
        self.depthwise3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True, groups=in_channels)
        self.pointwise1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x):
        x_identity = x.clone()
        x = self.depthwise1(x) + x
        x = self.depthwise2(x) + x
        x = self.relu(x)
        x = self.depthwise3(x) + x
        x = x + x_identity
        x = self.pointwise1(x)
        
        return x

class SampleFusedModule(nn.Module):
    def __init__(self, channel=1024, mode='up'):
        super(SampleFusedModule, self).__init__()
        self.mlp_5x5_1 = nn.Conv2d(channel, channel, 5, padding=2, groups=channel)
        self.mlp_3x3_2 = nn.Conv2d(channel, channel, 3, padding=1, groups=channel)
        self.mlp_3x3_3 = nn.Conv2d(channel, channel, 3, padding=1, groups=channel)
        self.mlp_1x1 = nn.Conv2d(channel, channel, 1)
        
        self.mlp1_5x5_1 = nn.Conv2d(channel*2, channel*2, 5, padding=2, groups=channel*2)
        self.mlp1_3x3_2 = nn.Conv2d(channel*2, channel*2, 3, padding=1, groups=channel*2)
        self.mlp1_3x3_3 = nn.Conv2d(channel*2, channel*2, 3, padding=1, groups=channel*2)
        self.mlp1_1x1 = nn.Conv2d(channel*2, channel, 1)
        
        self.mlp2_5x5_1 = nn.Conv2d(channel, channel, 5, padding=2, groups=channel)
        self.mlp2_3x3_2 = nn.Conv2d(channel, channel, 3, padding=1, groups=channel)
        self.mlp2_3x3_3 = nn.Conv2d(channel, channel, 3, padding=1, groups=channel)
        self.mlp2_1x1 = nn.Conv2d(channel, channel*2, 1)
             
        self.attn1 = CrossAttentionFusion(channel, channel)
        self.attn2 = CrossAttentionFusion(channel*2, channel*2)
        self.attn3 = CrossAttentionFusion(channel, channel)

        self.mode = mode
    def forward(self, x1, x2):
        if self.mode == 'up':
            x1 = F.interpolate(x1, x2.size()[2:], mode='bilinear', align_corners=False)
            x_identity = x1.clone()
            x1 = self.mlp1_5x5_1(x1) + x1
            x1 = self.mlp1_3x3_2(x1) + x1
            x1 = self.mlp1_3x3_3(x1) + x1 + x_identity
            x1 = self.mlp1_1x1(x1)
            x = self.attn1(x2, x1)
        elif self.mode == 'down':
            x2 = F.interpolate(x2, x1.size()[2:], mode='bilinear', align_corners=True)
            x_identity = x2.clone()
            x2 = self.mlp2_5x5_1(x2) + x2
            x2 = self.mlp2_3x3_2(x2) + x2
            x2 = self.mlp2_3x3_3(x2) + x2 + x_identity
            x2 = self.mlp2_1x1(x2)
            x = self.attn2(x1, x2)
        else:
            x_identity = x1.clone()
            x1 = self.mlp_5x5_1(x1) + x1
            x1 = self.mlp_3x3_2(x1) + x1
            x1 = self.mlp_3x3_3(x1) + x1 + x_identity
            x1 = self.mlp_1x1(x1)
            x = self.attn3(x2, x1)
        return x
        
# CAFM
class CrossAttentionFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossAttentionFusion, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, groups=in_channels)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, groups=in_channels)
        
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, groups=in_channels)
        
        self.sigmoid = nn.Sigmoid()

    def channel_shuffle(self, x, groups):
        B, C, H, W = x.size()
        
        assert C % groups == 0, "The number of channels must be divisible by the number of groups."
        
        x = x.view(B, groups, C // groups, H, W)  
        x = x.permute(0, 2, 1, 3, 4).contiguous() 
        return x.view(B, C, H, W)                 

    def forward(self, F_rgb, F_depth):
        B, C, H, W = F_rgb.shape
        # Query, Key, Value
        Query = self.query_conv(F_rgb)

        Key = self.key_conv(F_depth)

        Value = self.value_conv(F_depth)

        attn1 = Query * Key
        Query = self.channel_shuffle(Query, groups=4)
        Key = self.channel_shuffle(Key, groups=4)
        attn2 = Query * Key

        blocks1 = attn1.view(B, C, H//4, 4, W//4, 4)
        blocks1 = blocks1.permute(0,1,2,4,3,5)

        blocks2 = attn2.view(B, C, H//4, 4, W//4, 4)
        blocks2 = blocks2.permute(0,1,2,4,3,5)

        x_low_res1 = blocks1.mean(dim=(4,5))
        x_low_res2 = blocks2.mean(dim=(4,5))

        x_low_res_expanded1 = x_low_res1.unsqueeze(-1).unsqueeze(-1)  # (B,C,H/2,W/2,1,1)
        x_low_res_expanded1 = x_low_res_expanded1.expand(-1,-1,-1,-1,4,4)

        x_low_res_expanded2 = x_low_res2.unsqueeze(-1).unsqueeze(-1)  # (B,C,H/2,W/2,1,1)
        x_low_res_expanded2 = x_low_res_expanded2.expand(-1,-1,-1,-1,4,4)

        attn = x_low_res_expanded2.permute(0,1,2,4,3,5).reshape(B,C,H,W) + x_low_res_expanded1.permute(0,1,2,4,3,5).reshape(B,C,H,W)

        channel_attn = torch.mean(attn, dim=1, keepdim=True)
        channel_attn = nn.Softmax(dim=1)(channel_attn)
        Attention = self.sigmoid(attn * channel_attn)

        # Weighted Value
        F_fused = Attention * Value

        return F_rgb + F_fused

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.PReLU(channel // reduction),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.GELU(), res_scale=1
    ):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.fc2 = nn.Linear(channel // reduction, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, (1, 1))  
        y = y.view(b, c)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y  


class EdgeConstructModule(nn.Module):
    def __init__(self, in_fea=[64, 128, 256, 512, 512], mid_fea=32):
        super(EdgeConstructModule, self).__init__()
        self.relu = nn.PReLU(mid_fea)
        
        self.conv1 =  nn.Conv2d(in_fea[0], mid_fea, 1)
        self.conv2 =  nn.Conv2d(in_fea[1], mid_fea, 1)
        self.conv3 =  nn.Conv2d(in_fea[2], mid_fea, 1)
        self.conv4 =  nn.Conv2d(in_fea[3], mid_fea, 1)
        self.conv5 =  nn.Conv2d(in_fea[4], mid_fea, 1)
        
        self.conv5_1 = nn.Conv2d(in_fea[0], in_fea[0], 5, padding=2, groups=in_fea[0])
        self.conv5_2 = nn.Conv2d(in_fea[1], in_fea[1], 5, padding=2, groups=in_fea[1])
        self.conv5_3 = nn.Conv2d(in_fea[2], in_fea[2], 5, padding=2, groups=in_fea[2])
        self.conv5_4 = nn.Conv2d(in_fea[3], in_fea[3], 5, padding=2, groups=in_fea[3])
        self.conv5_5 = nn.Conv2d(in_fea[4], in_fea[4], 5, padding=2, groups=in_fea[4])

        self.classifer = nn.Sequential(
            nn.Conv2d(mid_fea, mid_fea, 3, padding=1),
            nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        )
        self.se_block = SEBlock(mid_fea)
        self.rcab = RCAB(mid_fea)

        self.conv_after_up_1 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)

        self.conv_after_up_2 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)

        self.conv_after_up_3 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)

        self.conv_after_up_4 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)

        self.conv_after_up_5 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)

    def forward(self, x1, x2, x3, x4, x5):
        
        edge1_fea = self.conv5_1(x1) + x1
        edge1 = self.relu(self.conv1(edge1_fea))
        
        edge2_fea = self.conv5_2(x2) + x2
        edge2 = self.relu(self.conv2(edge2_fea))
        
        edge3_fea = self.conv5_3(x3) + x3
        edge3 = self.relu(self.conv3(edge3_fea))
        
        edge4_fea = self.conv5_4(x4) + x4
        edge4 = self.relu(self.conv4(edge4_fea))
        
        edge5_fea = self.conv5_5(x5) + x5
        edge5 = self.relu(self.conv5(edge5_fea))

        edge5 = self.conv_after_up_5(edge5) + edge5
        edge0 = edge5 + edge4
        
        edge0 = F.interpolate(edge0, edge3.size()[2:], mode='bilinear', align_corners=False)
        edge0 = self.conv_after_up_4(edge0) + edge0    
        edge0 = edge0 + edge3
        
        edge0 = F.interpolate(edge0, edge2.size()[2:], mode='bilinear', align_corners=False)
        edge0 = self.conv_after_up_3(edge0) + edge0    
        edge0 = edge0 + edge2
        
        edge0 = F.interpolate(edge0, edge1.size()[2:], mode='bilinear', align_corners=False)
        edge0 = self.conv_after_up_2(edge0) + edge0    
        edge0 = edge0 + edge1      

        edge0 = F.interpolate(edge0, scale_factor=2, mode='bilinear', align_corners=False)
        edge0 = self.conv_after_up_1(edge0) + edge0    
        
        edge0 = self.se_block(edge0) + edge0
        edge = self.rcab(edge0)
        
        edge = self.classifer(edge) + edge
        return edge

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''
    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            # self.output_stride = 16
            pass
            # rates = [int(6 * (x.size(2)/256)) , 12, 18]  # 根据输入尺寸动态调整
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.PReLU(reduction_dim)
        ))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3, dilation=r, padding=r, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.PReLU(reduction_dim)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveMaxPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.PReLU(reduction_dim)
        )
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            nn.PReLU(reduction_dim)
        )
        self.se_block = SEBlock(reduction_dim)

    def forward(self, x, edge):
        x_size = x.size()
        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:], mode='bilinear', align_corners=False)
        out = img_features

        edge_features = F.interpolate(edge, x_size[2:], mode='bilinear', align_corners=False)
        edge_features = self.se_block(self.edge_conv(edge_features) + edge_features)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

# Cross-Modality Muti-Scale Attention-Weighted Module
class CMAM(nn.Module):
    def __init__(self, channel, out_channel=None, reduction=16):
        super(CMAM, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(3)
        self.max_pool1 = nn.AdaptiveMaxPool2d(3)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(5)
        self.max_pool2 = nn.AdaptiveMaxPool2d(5)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(1)
        self.max_pool4 = nn.AdaptiveMaxPool2d(1)
        self.channel = channel
        if out_channel is not None:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(channel*2, channel*2, 5, padding=2, bias=True, groups=channel*2),
                nn.Conv2d(channel*2, channel*2, 3, padding=1, bias=True, groups=channel*2),
                nn.GELU(),
                nn.Conv2d(channel*2, channel*2, 3, padding=1, bias=True, groups=channel*2),
            )
            self.after_conv_layers = nn.Conv2d(channel*2, out_channel, 1, bias=True)
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(channel*2, channel*2, 5, padding=2, bias=True, groups=channel*2),
                nn.Conv2d(channel*2, channel*2, 3, padding=1, bias=True, groups=channel*2),
                nn.GELU(),
                nn.Conv2d(channel*2, channel*2, 3, padding=1, bias=True, groups=channel*2),
            )
            self.after_conv_layers = nn.Conv2d(channel*2, channel*2, 1, bias=True)
        
        self.convy1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, bias=True, groups=channel),
            nn.Sigmoid()
        )
        self.convy2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, bias=True, groups=channel),
            nn.Sigmoid()
        )
        self.convz1 = nn.Sequential(
            nn.Conv2d(channel, channel, 5, bias=True, groups=channel),
            nn.Sigmoid()
        )
        self.convz2 = nn.Sequential(
            nn.Conv2d(channel, channel, 5, bias=True, groups=channel),
            nn.Sigmoid()
        )
        self.convv1 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=True),
            nn.Sigmoid()
        )
        self.convv2 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=True),
            nn.Sigmoid()
        )
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        self.mlp1 = nn.Conv2d(channel*2, channel, 1, bias=False)
        self.mlp2 = nn.Conv2d(channel*2, channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        self.conv1 = nn.Conv2d(channel*2, channel, 1)
        self.conv2 = nn.Conv2d(channel*2, channel, 1)
        
        self.cross_attn1 = CrossAttentionFusion(channel, channel)
        self.cross_attn2 = CrossAttentionFusion(channel, channel)
        self.weight1 = nn.Parameter(torch.ones(3))  # learnable params
        self.weight2 = nn.Parameter(torch.ones(3))

    def channel_shuffle(self, x: torch.Tensor, groups: int):
        B, C, H, W = x.size()
        
        assert C % groups == 0, "The number of channels must be divisible by the number of groups."
        
        x = x.view(B, groups, C // groups, H, W) 
        x = x.permute(0, 2, 1, 3, 4).contiguous() 
        return x.view(B, C, H, W)                 


    def forward(self, x1, x2):

        max_out1 = self.mlp1(self.max_pool(x1))
        avg_out1 = self.mlp1(self.avg_pool(x1))
        channel_out1 = self.sigmoid(max_out1 + avg_out1)
        x1 = self.conv1(x1)
        x1 = channel_out1 * x1 + x1
        
        max_out2 = self.mlp2(self.max_pool(x2))
        avg_out2 = self.mlp2(self.avg_pool(x2))
        channel_out2 = self.sigmoid(max_out2 + avg_out2)
        x2 = self.conv2(x2)
        x2 = channel_out2 * x2 + x2
        
        # Muti-Scale
        y_x1 = self.avg_pool1(x1) + self.max_pool1(x1)
        y_x2 = self.avg_pool1(x2) + self.max_pool1(x2)
        
        y1 = self.convy1(y_x1)
        y2 = self.convy2(y_x2)

        z_x1 = self.avg_pool2(x1) + self.max_pool2(x1)
        z_x2 = self.avg_pool2(x2) + self.max_pool2(x2)

        z1 = self.convz1(z_x1)
        z2 = self.convz2(z_x2)
        
        
        v_x1 = self.avg_pool4(x1) + self.max_pool4(x1)
        v_x2 = self.avg_pool4(x2) + self.max_pool4(x2)
        
        v1 = self.convv1(v_x1)
        v2 = self.convv2(v_x2)
        
        x = torch.cat((x1, x2), dim=1)

        # Attention-Weighted
        fused1 = (self.weight1[0]*y1 + self.weight1[1]*z1 + self.weight1[2]*v1) / self.weight1.sum()
        fused2 = (self.weight2[0]*y2 + self.weight2[1]*z2 + self.weight2[2]*v2) / self.weight2.sum()
        
        x_fir_half = x[:, :self.channel] * fused1
        x_sec_half = x[:, self.channel:] * fused2
        
        # Cross-Attention
        x_fir_half = self.cross_attn1(x_fir_half, x_sec_half)
        x_sec_half = self.cross_attn2(x_sec_half, x_fir_half)

        x_fir_half = self.channel_shuffle(x_fir_half, groups=4)
        x_sec_half = self.channel_shuffle(x_sec_half, groups=4)
        
        x = torch.cat((x_fir_half, x_sec_half), dim=1)
        ret = self.conv_layers(x) + x
        ret = self.after_conv_layers(x)
        return ret

class SelfGatedPredUnit(nn.Module):
    def __init__(self):
        super(SelfGatedPredUnit, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=True, groups=64)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv6 = nn.Conv2d(64, 1, kernel_size=1, bias=True)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.conv4(x0)
        x0 = self.sigmoid(x0)
        x = x0 * x + x
        
        x = self.conv5(x) + x
        x = self.conv6(x)
        return x
        
class CBAMConvBlock(nn.Module):
    def __init__(self):
        super(CBAMConvBlock, self).__init__()
        self.cbam = CBAMLayer(1024)      
        self.conv7x7_1 = nn.Conv2d(1024, 1024, 5, padding=2, groups=1024)
        self.conv7x7_2 = nn.Conv2d(1024, 1024, 3, padding=1, groups=1024)
        self.conv7x7_3 = nn.Conv2d(1024, 1024, 3, padding=1, groups=1024)
        self.conv1x1_1 = nn.Conv2d(1024, 512, 1)
        self.prelu = nn.PReLU(512)
        self.conv1x1_2 = nn.Conv2d(512, 1024, 1)
    def forward(self, x):
        
        x_identity = x.clone()
        x = self.conv7x7_1(x) + x
        x = self.conv7x7_2(x) + x
        x = self.conv7x7_3(x) + x
        x = self.conv1x1_1(x)
        x = self.prelu(x)
        x = self.conv1x1_2(x) + x_identity
        x = self.cbam(x)
        
        return x

        
class NESS_Net(nn.Module):
    def __init__(self, channel=32, general_stage=True):
        super(NESS_Net, self).__init__()
        self.relu = nn.PReLU(channel*2)
        self.edge_layer = EdgeConstructModule([128, 256, 512, 1024, 1024], mid_fea=96)
        self.edge_layer_depth = EdgeConstructModule([128, 256, 512, 1024, 1024], mid_fea=96)
        self.aspp = _AtrousSpatialPyramidPoolingModule(512, 32, output_stride=16)
        self.aspp_depth = _AtrousSpatialPyramidPoolingModule(512, 32, output_stride=16)
        
        self.aspp_2 = _AtrousSpatialPyramidPoolingModule(256, 32, output_stride=16)
        self.aspp_2_depth = _AtrousSpatialPyramidPoolingModule(256, 32, output_stride=16)

        self.sal_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=True, groups=channel)
        self.edge_conv = nn.Sequential( 
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(1, channel, kernel_size=1, bias=True)
        )
        self.edge_conv_depth = nn.Sequential( 
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(1, channel, kernel_size=1, bias=True)
        )
        self.rcab_sal_edge = RCAB(channel*2)
        self.after_aspp_conv5 = CMAM(channel*3, out_channel=channel)
        self.rcab_5 = RCAB(channel)
        self.after_aspp_conv2 = CMAM(channel*3, out_channel=channel)
        self.rcab_2 = RCAB(channel)
        self.sigmoid = nn.Sigmoid()
        
        self.fuse_canny_edge = SeparableConv2d(97)
        self.fuse_canny_edge_depth = SeparableConv2d(97)

        self.selfGatedPredUnit = SelfGatedPredUnit()
        
        self.swin_rgb = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        
        self.swin_depth = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
               
        self.cmam = CMAM(channel)  

        self.general_stage = general_stage
        
        self.cbam_rgb = CBAMConvBlock()
        self.cbam_depth = CBAMConvBlock()
        self.cmam_before_seg = CMAM(channel//2)
        self.pool_before_fuse_edges = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_before_fuse_edges_depth = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cascade1_d = SampleFusedModule(channel=1024, mode='none')
        self.cascade1_r = SampleFusedModule(channel=1024, mode='none')
        self.cascade2_d = SampleFusedModule(channel=512, mode='up')
        self.cascade2_r = SampleFusedModule(channel=512, mode='up')
        self.cascade3_d = SampleFusedModule(channel=128, mode='down')
        self.cascade3_r = SampleFusedModule(channel=128, mode='down')
        self.conv_sal_edge = nn.Conv2d(channel*2, channel*2, kernel_size=3, padding=1, bias=True, groups=channel*2)
        self.conv_init = nn.Conv2d(channel, 1, 1)
        self.conv_before_sal_edge = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=True, groups=channel)
        self.conv_before_feat_fuse = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=True, groups=channel)
        
    def forward(self, x, depth, image_edges, depth_edges):
        size = image_edges.size()[2:]
        rgb_list = self.swin_rgb(x)
        x1 = rgb_list[0]     
        x2 = rgb_list[1]
        x3 = rgb_list[2]
        x4 = rgb_list[3]
        x5 = x4.clone()
        x5 = self.cbam_rgb(x5)
        
        edge_map = self.edge_layer(x1, x2, x3, x4, x5)
        
        depth_list = self.swin_depth(depth)
        d1 = depth_list[0]
        d2 = depth_list[1]
        d3 = depth_list[2]
        d4 = depth_list[3]
        d5 = d4.clone()
        d5 = self.cbam_depth(d5)
        
        edge_map_depth = self.edge_layer_depth(d1, d2, d3, d4, d5)
        
        image_edges = self.pool_before_fuse_edges(image_edges)
        cat = torch.cat((edge_map, image_edges), dim=1)
        edge_fuse = self.fuse_canny_edge(cat) 

        depth_edges = self.pool_before_fuse_edges_depth(depth_edges)
        cat_depth = torch.cat((edge_map_depth, depth_edges), 1)
        edge_fuse_depth = self.fuse_canny_edge_depth(cat_depth) 

        d5 = self.cascade1_d(d5, d4)
        d5 = self.cascade2_d(d5, d3)
        
        x5 = self.cascade1_r(x5, x4)
        x5 = self.cascade2_r(x5, x3)

        d5 = self.aspp_depth(d5, edge_fuse_depth)
        x5 = self.aspp(x5, edge_fuse)
        
        d2 = self.cascade3_d(d2, d1)
        x2 = self.cascade3_r(x2, x1)
        
        d2 = self.aspp_2_depth(d2, edge_fuse_depth)
        x2 = self.aspp_2(x2, edge_fuse)
        
        x_conv5 = self.after_aspp_conv5(x5, d5)
        x_conv2 = self.after_aspp_conv2(x2, d2)
        
        x_conv5_up = F.interpolate(x_conv5, x2.size()[2:], mode='bilinear', align_corners=False)
        x_conv5_up = self.conv_before_feat_fuse(x_conv5_up) + x_conv5_up
        
        sal_init = self.cmam_before_seg(x_conv5_up, x_conv2)

        sal_init = F.interpolate(sal_init, scale_factor=2, mode='bilinear', align_corners=False)     
        sal_init = self.sal_conv(sal_init) + sal_init

        edge_feature = self.edge_conv(edge_fuse)

        sal_init = F.interpolate(sal_init, edge_feature.size()[2:], align_corners=False, mode='bilinear')
        sal_init = self.conv_before_sal_edge(sal_init) + sal_init

        sal_edge_feature = torch.cat((sal_init,edge_feature), 1)

        edge_feature_depth = self.edge_conv_depth(edge_fuse_depth)

        sal_edge_feature_depth = torch.cat((sal_init, edge_feature_depth), 1)
        
        sal_init = F.interpolate(sal_init, size, align_corners=False, mode='bilinear')
        sal_init = self.conv_init(sal_init)
        
        sal_edge_feature = self.cmam(sal_edge_feature, sal_edge_feature_depth)
        
        sal_edge_feature = F.interpolate(sal_edge_feature, size, mode='bilinear', align_corners=False)
        sal_edge_feature = self.conv_sal_edge(sal_edge_feature) + sal_edge_feature

        sal_edge_feature = self.rcab_sal_edge(sal_edge_feature)
        if self.general_stage:
            sal_ref = self.selfGatedPredUnit(self.relu(sal_edge_feature))

        else:
            sal_ref = self.selfGatedPredUnit(self.relu(F.dropout(sal_edge_feature, p=0.25)))
                      
        
        return sal_init, edge_fuse, sal_ref, edge_fuse_depth, image_edges, depth_edges
    
    def load_pre(self, pre_model):
        self.swin_rgb.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        self.swin_depth.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")
