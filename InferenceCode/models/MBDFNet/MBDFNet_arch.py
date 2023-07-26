from thop import profile
from basicsr.utils.registry import ARCH_REGISTRY
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .idynamic import IDynamicDWConv
import numbers

class DWBlock(nn.Module):
    def __init__(self, dim, window_size, dynamic=False, inhomogeneous=False, heads=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.dynamic = dynamic
        self.inhomogeneous = inhomogeneous
        self.heads = heads

        # pw-linear
        self.conv0 = nn.Conv2d(dim, dim, 1, bias=False)

        if dynamic and inhomogeneous:
            print(window_size, heads)
            self.conv = IDynamicDWConv(dim, window_size, heads)
        else:
            self.conv = nn.Conv2d(dim, dim, kernel_size=window_size, stride=1, padding=window_size // 2, groups=dim)

        # pw-linear
        self.conv2 = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv0(x)
        x = self.conv(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # x = self.conv0(x)
        flops += N * self.dim * self.dim
        # x = self.conv(x)
        if self.dynamic and not self.inhomogeneous:
            flops += (
                        N * self.dim + self.dim * self.dim / 4 + self.dim / 4 * self.dim * self.window_size * self.window_size)
        elif self.dynamic and self.inhomogeneous:
            flops += (
                        N * self.dim * self.dim / 4 + N * self.dim / 4 * self.dim / self.heads * self.window_size * self.window_size)
        flops += N * self.dim * self.window_size * self.window_size
        #  x = self.conv2(x)
        flops += N * self.dim * self.dim
        #  batchnorm + relu
        flops += 8 * self.dim * N
        return flops


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            Conv(3, out_plane // 4, kernel_size=3, stride=1, padding=1, act=True),
            Conv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, padding=0, act=True),
            Conv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, padding=1, act=True),
            Conv(out_plane // 2, out_plane - 3, kernel_size=1, stride=1, padding=0, act=True)
        )

        self.conv = Conv(out_plane, out_plane, kernel_size=1, stride=1, padding=0, act=True)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        # self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        # self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class GDFN(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False):
        super(GDFN, self).__init__()

        self.project_in = nn.Conv2d(in_channel, in_channel * 4, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(in_channel * 4, in_channel * 4, kernel_size=3, stride=1, padding=1,
                                groups=in_channel * 4, bias=bias)

        self.project_out = nn.Conv2d(in_channel * 2, out_channel, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class SpatialGatingUnit(nn.Module):
    def __init__(
        self,
        dim,
        patch_size=16,
        act = nn.Identity(),
        heads = 1,
        init_eps = 1e-3,
    ):
        super().__init__()
        dim_out = dim // 2
        self.heads = heads
        self.norm = nn.LayerNorm(dim_out)
        self.dim = dim_out
        self.act = act

        self.attn2conv = DWBlock(dim_out, 7, dynamic=True, inhomogeneous=True, heads=dim_out)

    def forward(self, x,x_cnn_dw):

        n, h = x.shape[1], self.heads

        res, gate = x.chunk(2, dim = -1)

        # x_cnn_dw: (512 24 1 1) gate: (2 256 64 24)

        x_cnn_dw = rearrange(x_cnn_dw, 'b c ph pw -> b (ph pw) c')
        gate = rearrange(gate, 'b n p c -> (b n) p c')

        gate = gate * x_cnn_dw

        res = rearrange(res, 'b n p c -> (b n) p c')
        gate = self.norm(gate)
        gate = self.attn2conv(rearrange(gate, 'b (h w) n -> b h w n', h=8, w=8))
        gate = rearrange(gate, 'b h w n -> b (h w) n')


        return self.act(gate) * res


class gmlp_mix(nn.Module):
    def __init__(self, dim, patch_size=8, head=1):
        super(gmlp_mix, self).__init__()


        self.to_hidden = nn.Conv2d(dim, dim , kernel_size=1)
        self.x_cnn = nn.Sequential(nn.Conv2d(dim, dim , kernel_size=1),
                                   LayerNorm(dim , LayerNorm_type='WithBias'),
                                   nn.Conv2d(dim , dim , kernel_size=3, stride=1, padding=1, groups=dim),
                                   nn.GELU()
                                   )
        self.act = nn.GELU()
        self.SGU = SpatialGatingUnit(dim,patch_size=patch_size,heads=head)
        self.channel_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            LayerNorm(dim //8, LayerNorm_type='WithBias'),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 2, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 16, kernel_size=1),
            LayerNorm(dim //16, LayerNorm_type='WithBias'),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )

        self.projection = nn.Conv2d(dim , dim//2, kernel_size=1)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)


        self.patch_size = patch_size

    def forward(self, x):

        b, c, h, w = x.shape

        x = self.to_hidden(x)
        x = self.act(x)
        x = rearrange(x, 'b c (h patch1) (w patch2) ->b (h w)  (patch1 patch2) c', patch1=self.patch_size,
                  patch2=self.patch_size)

        x_cnn = self.x_cnn(rearrange(x, 'b n (ph pw) c -> (b n) c ph pw', ph=self.patch_size, pw=self.patch_size))

        x_cnn_dw = self.channel_interaction(F.adaptive_avg_pool2d(x_cnn, output_size=1))
        x_cnn_dw = F.sigmoid(x_cnn_dw)

        x = self.SGU(x,x_cnn_dw)

        x_cnn = rearrange(x_cnn, '(b h w) c ph pw ->b c (h ph) (w pw)',b=b, h=h//self.patch_size, w=w//self.patch_size,
                          ph=self.patch_size, pw=self.patch_size)

        x = rearrange(x, '(b h w)  (patch1 patch2) c ->b c (h patch1) (w patch2) ', b = b,h=h//self.patch_size,
                      w = w//self.patch_size,patch1=self.patch_size,patch2=self.patch_size)

        x_att = x.clone() #c=dim//2
        x = self.spatial_interaction(x)
        x = F.sigmoid(x)
        # x_cnn: (256 48 128 128) x: (1 1 128 128)
        x = self.projection(x_cnn) * x

        out = torch.cat((x,x_att),dim=1)


        out = self.project_out(out)

        return out

class Atb(nn.Module):
    def __init__(self, n_feat):
        super(Atb, self).__init__()
        self.n_feat = n_feat
        self.conv1 = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, y):
        # fusion propagation
        feat_fusion = torch.cat([x, y], dim=1)  # b 128 256 256
        feat_fusion = self.lrelu(self.conv1(feat_fusion))  # b 128 256 256
        feat_prop1, feat_prop2 = torch.split(feat_fusion, self.n_feat, dim=1)
        feat_prop1 = feat_prop1 * torch.sigmoid(self.conv2(feat_prop1))
        feat_prop2 = feat_prop2 * torch.sigmoid(self.conv3(feat_prop2))
        x = feat_prop1 + feat_prop2
        return x


class Fuse(nn.Module):
    def __init__(self, n_feat, scale_factor):
        super(Fuse, self).__init__()
        self.n_feat = n_feat
        self.scale_factor = scale_factor
        self.atb = Atb(n_feat=self.n_feat)
        self.norm1 = LayerNorm(self.n_feat, LayerNorm_type='WithBias')
        self.norm = LayerNorm(self.n_feat, LayerNorm_type='WithBias')
        self.att_channel = GDFN(n_feat, n_feat)
        self.attn = gmlp_mix(n_feat, patch_size=8)

    def forward(self, enc, dnc):
        dnc = F.interpolate(dnc, scale_factor=1 / self.scale_factor, mode='bilinear')
        # fusion propagation

        x = self.atb(enc, dnc)
        x = x + self.attn(self.norm1(x))
        x = x + self.att_channel(self.norm(x))
        return x


class Conv(nn.Module):
    def __init__(self, input_channels, n_feats, kernel_size, stride=1, padding=0, bias=True, bn=False, act=False):
        super(Conv, self).__init__()
        m = []
        m.append(nn.Conv2d(input_channels, n_feats, kernel_size, stride, padding, bias=bias))
        if bn: m.append(nn.BatchNorm2d(n_feats))
        if act: m.append(nn.GELU())
        self.body = nn.Sequential(*m)

    def forward(self, input):
        return self.body(input)


class ResBlock(nn.Module):
    def __init__(self, n_feat, kernel_size, padding=0, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(Conv(n_feat, n_feat, kernel_size, padding=padding, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class SRN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_resblock=1, n_feat=32, kernel_size=5, ispre=False,
                 scale_factor=0.5):
        super(SRN, self).__init__()
        print("Creating SRN_SVLRM Net")

        self.ispre = ispre
        self.scale_factor = scale_factor

        if self.ispre:
            InBlock = [nn.Sequential(
                nn.Conv2d(in_channels * 2, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.ReLU(inplace=True)
            )]
            InBlock.extend([ResBlock(n_feat=n_feat, kernel_size=kernel_size, padding=kernel_size // 2)
                            for _ in range(n_resblock)])
            self.fuse1 = Fuse(n_feat=n_feat, scale_factor=self.scale_factor)
            self.fuse2 = Fuse(n_feat=n_feat * 2, scale_factor=self.scale_factor)
        else:
            InBlock = [nn.Sequential(
                nn.Conv2d(in_channels, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.ReLU(inplace=True)
            )]
            InBlock.extend([ResBlock(n_feat=n_feat, kernel_size=kernel_size, padding=kernel_size // 2)
                            for _ in range(n_resblock)])

        # encoder1
        Encoder_first = [nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.ReLU(inplace=True)
        )]
        Encoder_first.extend([ResBlock(n_feat=n_feat * 2, kernel_size=kernel_size, padding=kernel_size // 2)
                              for _ in range(n_resblock)])
        # encoder2
        Encoder_second = [nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.ReLU(inplace=True)
        )]
        Encoder_second.extend([ResBlock(n_feat=n_feat * 4, kernel_size=kernel_size, padding=kernel_size // 2)
                               for _ in range(n_resblock)])

        # decoder2
        Decoder_second = [ResBlock(n_feat=n_feat * 4, kernel_size=kernel_size, padding=kernel_size // 2)
                          for _ in range(n_resblock)]
        Decoder_second.append(nn.Sequential(
            nn.ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        ))
        # decoder1
        Decoder_first = [ResBlock(n_feat=n_feat * 2, kernel_size=kernel_size, padding=kernel_size // 2)
                         for _ in range(n_resblock)]
        Decoder_first.append(nn.Sequential(
            nn.ConvTranspose2d(n_feat * 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        ))

        OutBlock = [ResBlock(n_feat=n_feat, kernel_size=kernel_size, padding=kernel_size // 2)
                    for _ in range(n_resblock)]

        OutBlock_Post = [
            nn.Conv2d(n_feat, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)]

        self.inBlock = nn.Sequential(*InBlock)
        self.encoder_first = nn.Sequential(*Encoder_first)
        self.encoder_second = nn.Sequential(*Encoder_second)
        self.decoder_second = nn.Sequential(*Decoder_second)
        self.decoder_first = nn.Sequential(*Decoder_first)
        self.outBlock = nn.Sequential(*OutBlock)
        self.outBlock_post = nn.Sequential(*OutBlock_Post)

    def forward(self, x, pre_x=None, pre_dnc1=None, pre_dnc2=None):

        if self.ispre == False:
            first_scale_inblock = self.inBlock(x)
            first_scale_encoder_first = self.encoder_first(first_scale_inblock)
            first_scale_encoder_second = self.encoder_second(first_scale_encoder_first)
            first_scale_decoder_second = self.decoder_second(first_scale_encoder_second)
            first_scale_decoder_first = self.decoder_first(first_scale_decoder_second + first_scale_encoder_first)
            first_scale_outBlock = self.outBlock(first_scale_decoder_first + first_scale_inblock)

            recons = self.outBlock_post(first_scale_outBlock)
            return recons, first_scale_decoder_second, first_scale_outBlock

        else:
            first_scale_inblock = self.inBlock(torch.cat([x, pre_x], dim=1))
            tmp01 = self.fuse1(first_scale_inblock, pre_dnc2)
            first_scale_encoder_first = self.encoder_first(tmp01)
            tmp02 = self.fuse2(first_scale_encoder_first, pre_dnc1)
            first_scale_encoder_second = self.encoder_second(tmp02)
            # fuse2

            first_scale_decoder_second = self.decoder_second(first_scale_encoder_second)
            # fuse1

            first_scale_decoder_first = self.decoder_first(first_scale_decoder_second + first_scale_encoder_first)
            first_scale_outBlock = self.outBlock(first_scale_decoder_first + first_scale_inblock)

            recons = self.outBlock_post(first_scale_outBlock)
            return recons, first_scale_decoder_second, first_scale_outBlock



class MBDFNet(nn.Module):
    def __init__(self, scale_fact):
        super(MBDFNet, self).__init__()
        self.scale_fact = scale_fact
        self.srn1 = SRN(in_channels=3, out_channels=3, n_resblock=6, n_feat=16, kernel_size=3, ispre=False)
        self.srn2 = SRN(in_channels=3, out_channels=3, n_resblock=3, n_feat=16, kernel_size=3, ispre=True,
                        scale_factor=self.scale_fact)
        self.srn3 = SRN(in_channels=3, out_channels=3, n_resblock=3, n_feat=16, kernel_size=3, ispre=True,
                        scale_factor=self.scale_fact)
        self.feat_extract = SCM(out_plane=16)
        self.atb = Atb(n_feat=16)
        # stage3 = [ResBlock(n_feat=16 * 2, kernel_size=3, padding=1)
        #           for _ in range(3)]
        # self.stage3 = nn.Sequential(*stage3)

        de_smooth_net = [nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)]
        de_smooth_net.extend(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
                             for _ in range(6))

        self.de_smooth_net = nn.Sequential(*de_smooth_net)
        self.output_net = nn.Conv2d(16, 3, 3, 1, 1)


    def spatial_padding(self, lrs, pad):
        """ Apply spatial pdding.
        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).
            pad (int)
        Returns:
            Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).
        """
        b, c, h, w = lrs.size()

        pad_h = (pad - h % pad) % pad
        pad_w = (pad - w % pad) % pad

        # padding
        lrs = lrs.view(-1, c, h, w)
        lrs = F.pad(lrs, [0, pad_w, 0, pad_h], mode='reflect')

        return lrs.view(b, c, h + pad_h, w + pad_w)

    def forward(self, x):

        _, _, H_in, W_in = x.size()
        x = self.spatial_padding(x, pad=64)

        x_scale1 = x
        x_scale2 = F.interpolate(x_scale1, scale_factor=0.75, mode='bicubic')
        x_scale4 = F.interpolate(x_scale2, scale_factor=0.75, mode='bicubic')
        x_scale8 = F.interpolate(x_scale4, scale_factor=0.75, mode='bicubic')


        # scale8
        x_scale8_out, dnc1, dnc2 = self.srn1(x_scale8, None, None, None)
        x_scale8_up = F.interpolate(x_scale8_out, scale_factor=1 / 0.75, mode='bilinear')

        x_scale8_out += x_scale8

        # scale4
        x_scale4_out, dnc1, dnc2 = self.srn2(x_scale4, x_scale8_up, dnc1, dnc2)
        x_scale4_up = F.interpolate(x_scale4_out, scale_factor=1 / 0.75, mode='bilinear')

        x_scale4_out += x_scale4

        # scale2
        x_scale2_out, dnc1, dnc2 = self.srn3(x_scale2, x_scale4_up, dnc1, dnc2)
        up = F.interpolate(dnc2, scale_factor=1 / 0.75, mode='bilinear')

        x_scale2_out += x_scale2

        # scale1
        x_scale1_out = self.feat_extract(x_scale1)
        x_scale1_out = self.atb(x_scale1_out, up)
        x_scale1_out = self.output_net(self.de_smooth_net(x_scale1_out)) + x_scale1

        # return x_scale1_out[:, :, :H_in, :W_in], x_scale2_out[:, :, :int(H_in*0.75), :int(W_in*0.75)], x_scale4_out[:, :, :H_in*(0.75**2), :W_in*(0.75**2)], x_scale8_out[:, :, :H_in*(0.75**3), :W_in*(0.75**3)]
        return x_scale1_out[:, :, :H_in, :W_in], x_scale2_out[:, :, :int(H_in*0.75), :int(W_in*0.75)], x_scale4_out[:, :, :int(H_in*0.75**2), :int(W_in*0.75**2)], x_scale8_out[:, :, :int(H_in*0.75**3), :int(W_in*0.75**3)]

if __name__ == '__main__':
    model = MBDFNet(0.75).cuda()

    input = x = torch.ones(1, 3, 256, 256).cuda()
    out = model(input)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(name)
    print(f'out_1_size: {out[0].size()}; out_2_size: {out[1].size()}; out_3_size: {out[2].size()}; out_4_size: {out[3].size()}')
    flops, params = profile(model, inputs=(input,))
    print(flops / 1e9, params / 1e6)
