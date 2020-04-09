from model_utils import Conv2dTime, ODEBlock, get_nonlinearity

import torch
import torch.nn as nn



class ConvODEFunc(nn.Module):
    def __init__(self, nf, time_dependent=False, non_linearity='relu'):
        """
        Block for ConvODEUNet

        Args:
            nf (int): number of filters for the conv layers
            time_dependent (bool): whether to concat the time as a feature map before the convs
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
        """
        super(ConvODEFunc, self).__init__()
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations

        if time_dependent:
            self.norm1 = nn.InstanceNorm2d(nf)
            self.conv1 = Conv2dTime(nf, nf, kernel_size=3, stride=1, padding=1)
            self.norm2 = nn.InstanceNorm2d(nf)
            self.conv2 = Conv2dTime(nf, nf, kernel_size=3, stride=1, padding=1)
        else:
            self.norm1 = nn.InstanceNorm2d(nf)
            self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
            self.norm2 = nn.InstanceNorm2d(nf)
            self.conv2 = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=1)

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, t, x):
        self.nfe += 1
        if self.time_dependent:
            out = self.norm1(x)
            out = self.conv1(t, x)
            out = self.non_linearity(out)
            out = self.norm2(out)
            out = self.conv2(t, out)
            out = self.non_linearity(out)
        else:
            out = self.norm1(x)
            out = self.conv1(out)
            out = self.non_linearity(out)
            out = self.norm2(out)
            out = self.conv2(out)
            out = self.non_linearity(out)
        return out

class ConvODEUNet(nn.Module):
    def __init__(self, num_filters, output_dim=1, time_dependent=False,
                 non_linearity='softplus', tol=1e-3, adjoint=False):
        """
        ConvODEUNet (U-Node in paper)

        Args:
            num_filters (int): number of filters for first conv layer
            output_dim (int): how many feature maps the network outputs
            time_dependent (bool): whether to concat the time as a feature map before the convs
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
            tol (float): tolerance to be used for ODE solver
            adjoint (bool): whether to use the adjoint method to calculate the gradients
        """
        super(ConvODEUNet, self).__init__()
        nf = num_filters

        self.input_1x1 = nn.Conv2d(3, nf, 1, 1)

        ode_down1 = ConvODEFunc(nf, time_dependent, non_linearity)
        self.odeblock_down1 = ODEBlock(ode_down1, tol=tol, adjoint=adjoint)
        self.conv_down1_2 = nn.Conv2d(nf, nf*2, 1, 1)

        ode_down2 = ConvODEFunc(nf*2, time_dependent, non_linearity)
        self.odeblock_down2 = ODEBlock(ode_down2, tol=tol, adjoint=adjoint)
        self.conv_down2_3 = nn.Conv2d(nf*2, nf*4, 1, 1)

        ode_down3 = ConvODEFunc(nf*4, time_dependent, non_linearity)
        self.odeblock_down3 = ODEBlock(ode_down3, tol=tol, adjoint=adjoint)
        self.conv_down3_4 = nn.Conv2d(nf*4, nf*8, 1, 1)

        ode_down4 = ConvODEFunc(nf*8, time_dependent, non_linearity)
        self.odeblock_down4 = ODEBlock(ode_down4,  tol=tol, adjoint=adjoint)
        self.conv_down4_embed = nn.Conv2d(nf*8, nf*16, 1, 1)

        ode_embed = ConvODEFunc(nf*16, time_dependent, non_linearity)
        self.odeblock_embedding = ODEBlock(ode_embed,  tol=tol, adjoint=adjoint)

        self.conv_up_embed_1 = nn.Conv2d(nf*16+nf*8, nf*8, 1, 1)
        ode_up1 = ConvODEFunc(nf*8, time_dependent, non_linearity)
        self.odeblock_up1 = ODEBlock(ode_up1, tol=tol, adjoint=adjoint)

        self.conv_up1_2 = nn.Conv2d(nf*8+nf*4, nf*4, 1, 1)
        ode_up2 = ConvODEFunc(nf*4, time_dependent, non_linearity)
        self.odeblock_up2 = ODEBlock(ode_up2, tol=tol, adjoint=adjoint)

        self.conv_up2_3 = nn.Conv2d(nf*4+nf*2, nf*2, 1, 1)
        ode_up3 = ConvODEFunc(nf*2, time_dependent, non_linearity)
        self.odeblock_up3 = ODEBlock(ode_up3, tol=tol, adjoint=adjoint)

        self.conv_up3_4 = nn.Conv2d(nf*2+nf, nf, 1, 1)
        ode_up4 = ConvODEFunc(nf, time_dependent, non_linearity)
        self.odeblock_up4 = ODEBlock(ode_up4, tol=tol, adjoint=adjoint)

        self.classifier = nn.Conv2d(nf, output_dim, 1)

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x, return_features=False):
        x = self.non_linearity(self.input_1x1(x))

        features1 = self.odeblock_down1(x)  # 512
        x = self.non_linearity(self.conv_down1_2(features1))
        x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        features2 = self.odeblock_down2(x)  # 256
        x = self.non_linearity(self.conv_down2_3(features2))
        x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        features3 = self.odeblock_down3(x)  # 128
        x = self.non_linearity(self.conv_down3_4(features3))
        x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        features4 = self.odeblock_down4(x)  # 64
        x = self.non_linearity(self.conv_down4_embed(features4))
        x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        x = self.odeblock_embedding(x)  # 32

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((x, features4), dim=1)
        x = self.non_linearity(self.conv_up_embed_1(x))
        x = self.odeblock_up1(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((x, features3), dim=1)
        x = self.non_linearity(self.conv_up1_2(x))
        x = self.odeblock_up2(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((x, features2), dim=1)
        x = self.non_linearity(self.conv_up2_3(x))
        x = self.odeblock_up3(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((x, features1), dim=1)
        x = self.non_linearity(self.conv_up3_4(x))
        x = self.odeblock_up4(x)

        pred = self.classifier(x)
        return pred

class ConvResFunc(nn.Module):
    def __init__(self, num_filters, non_linearity='relu'):
        """
        Block for ConvResUNet

        Args:
            num_filters (int): number of filters for the conv layers
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
        """
        super(ConvResFunc, self).__init__()

        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm2d(2, num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x):
        out = self.norm(x)
        out = self.conv1(x)
        out = self.non_linearity(out)
        out = self.norm(out)
        out = self.conv2(out)
        out = self.non_linearity(out)
        out = x + out
        return out

class ConvResUNet(nn.Module):
    def __init__(self, num_filters, output_dim=1, non_linearity='softplus'):
        """
        ConvResUNet (U-Node in paper)

        Args:
            num_filters (int): number of filters for first conv layer
            output_dim (int): how many feature maps the network outputs
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
        """
        super(ConvResUNet, self).__init__()
        self.output_dim = output_dim

        self.input_1x1 = nn.Conv2d(3, num_filters, 1, 1)

        self.block_down1 = ConvResFunc(num_filters, non_linearity)
        self.conv_down1_2 = nn.Conv2d(num_filters, num_filters*2, 1, 1)
        self.block_down2 = ConvResFunc(num_filters*2, non_linearity)
        self.conv_down2_3 = nn.Conv2d(num_filters*2, num_filters*4, 1, 1)
        self.block_down3 = ConvResFunc(num_filters*4, non_linearity)
        self.conv_down3_4 = nn.Conv2d(num_filters*4, num_filters*8, 1, 1)
        self.block_down4 = ConvResFunc(num_filters*8, non_linearity)
        self.conv_down4_embed = nn.Conv2d(num_filters*8, num_filters*16, 1, 1)

        self.block_embedding = ConvResFunc(num_filters*16, non_linearity)

        self.conv_up_embed_1 = nn.Conv2d(num_filters*16+num_filters*8, num_filters*8, 1, 1)
        self.block_up1 = ConvResFunc(num_filters*8, non_linearity)
        self.conv_up1_2 = nn.Conv2d(num_filters*8+num_filters*4, num_filters*4, 1, 1)
        self.block_up2 = ConvResFunc(num_filters*4, non_linearity)
        self.conv_up2_3 = nn.Conv2d(num_filters*4+num_filters*2, num_filters*2, 1, 1)
        self.block_up3 = ConvResFunc(num_filters*2, non_linearity)
        self.conv_up3_4 = nn.Conv2d(num_filters*2+num_filters, num_filters, 1, 1)
        self.block_up4 = ConvResFunc(num_filters, non_linearity)

        self.classifier = nn.Conv2d(num_filters, self.output_dim, 1)

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x, return_features=False):
        x = self.non_linearity(self.input_1x1(x))

        features1 = self.block_down1(x)  # 512
        x = self.non_linearity(self.conv_down1_2(x))
        x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        features2 = self.block_down2(x)  # 256
        x = self.non_linearity(self.conv_down2_3(x))
        x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        features3 = self.block_down3(x)  # 128
        x = self.non_linearity(self.conv_down3_4(x))
        x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        features4 = self.block_down4(x)  # 64
        x = self.non_linearity(self.conv_down4_embed(x))
        x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        x = self.block_embedding(x)  # 32

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((x, features4), dim=1)
        x = self.non_linearity(self.conv_up_embed_1(x))
        x = self.block_up1(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((x, features3), dim=1)
        x = self.non_linearity(self.conv_up1_2(x))
        x = self.block_up2(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((x, features2), dim=1)
        x = self.non_linearity(self.conv_up2_3(x))
        x = self.block_up3(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat((x, features1), dim=1)
        x = self.non_linearity(self.conv_up3_4(x))
        x = self.block_up4(x)

        pred = self.classifier(x)
        return pred


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        """
        Block for LevelBlock

        Args:
            in_channels (int): number of input filters for first conv layer
            out_channels (int): number of output filters for the last layer
        """
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3), nn.ReLU(inplace=True),
        )

class LevelBlock(nn.Module):
    def __init__(self, depth, total_depth, in_channels, out_channels):
        """
        Block for UNet

        Args:
            depth (int): current depth of blocks (starts with total_depth: n,...,0)
            total_depth (int): total_depth of U-Net
            in_channels (int): number of input filters for first conv layer
            out_channels (int): number of output filters for the last layer
        """
        super(LevelBlock, self).__init__()
        self.depth = depth
        self.total_depth = total_depth
        if depth > 1:
            self.encode = ConvBlock(in_channels, out_channels)
            self.down = nn.MaxPool2d(2, 2)
            self.next = LevelBlock(depth - 1, total_depth, out_channels, out_channels * 2)
            next_out = list(self.next.modules())[-2].out_channels
            self.up = nn.ConvTranspose2d(next_out, next_out // 2, 2, 2)
            self.decode = ConvBlock(next_out // 2 + out_channels, out_channels)
        else:
            self.embed = ConvBlock(in_channels, out_channels)

    def forward(self, inp):
        if self.depth > 1:
            first_x = self.encode(inp)
            x = self.down(first_x)
            x = self.next(x)
            x = self.up(x)

            # center crop
            i_h = first_x.shape[2]
            i_w = first_x.shape[3]

            total_crop = i_h - x.shape[2]
            crop_left_top = total_crop // 2
            crop_right_bottom = total_crop - crop_left_top

            cropped_input = first_x[:, :,
                                    crop_left_top:i_h - crop_right_bottom,
                                    crop_left_top:i_w - crop_right_bottom]
            x = torch.cat((cropped_input, x), dim=1)

            x = self.decode(x)
        else:
            x = self.embed(inp)

        return x

class Unet(nn.Module):
    def __init__(self, depth, num_filters, output_dim):
        """
        Unet

        Args:
            depth (int): number of levels of UNet
            num_filters (int): number of filters for first conv layer
            output_dim (int): how many feature maps the network outputs
        """
        super(Unet, self).__init__()

        self.main = LevelBlock(depth, depth, 3, num_filters)
        main_out = list(self.main.modules())[-2].out_channels
        self.out = nn.Conv2d(main_out, output_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, inp):
        x = self.main(inp)
        return self.out(x)
