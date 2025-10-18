#! /usr/bin/python
# -*- coding: utf8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import time

# Helper function for weight initialization
def weights_init(m):
    """
    Initializes weights of the model with a normal distribution (std=0.04)
    and biases with a constant value of 0.0, similar to the original code.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.04)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class AttentionBlock(nn.Module):
    """
    A reusable block for the multi-scale and attention mechanism part of the network.
    """
    def __init__(self, in_channels, out_channels, ch, ks, hrg, wrg):
        super(AttentionBlock, self).__init__()
        self.in_channels = in_channels
        self.ch = ch
        self.hrg = hrg
        self.wrg = wrg
        
        # Dilated convolutions with different rates (assuming no weight sharing)
        self.atrous_conv1 = nn.Conv2d(in_channels, ch, kernel_size=ks, padding=ks//2, dilation=1, bias=False)
        self.atrous_conv2 = nn.Conv2d(in_channels, ch, kernel_size=ks, padding=ks//2 * 2, dilation=2, bias=False)
        self.atrous_conv3 = nn.Conv2d(in_channels, ch, kernel_size=ks, padding=ks//2 * 3, dilation=3, bias=False)
        self.atrous_conv4 = nn.Conv2d(in_channels, ch, kernel_size=ks, padding=ks//2 * 4, dilation=4, bias=False)
        self.atrous_conv5 = nn.Conv2d(in_channels, ch, kernel_size=ks, padding=ks//2 * 5, dilation=5, bias=False)

        # Scale Attention layers
        self.sca_dc1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=5//2 * 2, dilation=2)
        self.sca_dc2 = nn.Conv2d(32, 32, kernel_size=5, padding=5//2 * 2, dilation=2)
        self.sca_dc3 = nn.Conv2d(32, 16, kernel_size=5, padding=5//2 * 2, dilation=2)
        self.sca_dc4 = nn.Conv2d(16, 16, kernel_size=5, padding=5//2 * 2, dilation=2)
        self.sca_c2 = nn.Conv2d(16, 5, kernel_size=5, padding=5//2)

        # Shape Attention layers
        self.sha_pool = nn.AvgPool2d(kernel_size=(hrg, wrg), stride=(hrg, wrg))
        self.sha_c1 = nn.Conv2d(in_channels, ch // 4, kernel_size=1, padding=0)
        self.sha_c2 = nn.Conv2d(ch // 4, ch, kernel_size=1, padding=0)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        self.fusion_conv = nn.Conv2d(ch * 5, out_channels, kernel_size=3, padding=1)

    def forward(self, n):
        # Multi-scale feature extraction
        nn1 = self.leaky_relu(self.atrous_conv1(n))
        nn2 = self.leaky_relu(self.atrous_conv2(n))
        nn3 = self.leaky_relu(self.atrous_conv3(n))
        nn4 = self.leaky_relu(self.atrous_conv4(n))
        nn5 = self.leaky_relu(self.atrous_conv5(n))
        
        # Note: The original code's Transpose/Upsample/Transpose sequence for scale
        # attention is a complex way to broadcast a 5-channel attention map
        # to a (5 * ch)-channel feature map. The PyTorch implementation below is
        # more direct and achieves the same goal.
        
        nn_concat = torch.cat([nn1, nn2, nn3, nn4, nn5], dim=1)

        # Scale Attention (spatially varying)
        n_sc = self.leaky_relu(self.sca_dc1(n))
        n_sc = self.leaky_relu(self.sca_dc2(n_sc))
        n_sc = self.leaky_relu(self.sca_dc3(n_sc))
        n_sc = self.leaky_relu(self.sca_dc4(n_sc))
        n_sc = self.sigmoid(self.sca_c2(n_sc)) # (B, 5, H, W)
        
        # Broadcast multiply: scale each of the 5 feature blocks by its corresponding attention map
        n_sc_broadcast = n_sc.unsqueeze(2).repeat(1, 1, self.ch, 1, 1) # (B, 5, ch, H, W)
        nn_concat_reshaped = nn_concat.view(-1, 5, self.ch, nn_concat.size(2), nn_concat.size(3)) # (B, 5, ch, H, W)
        nn_sca = (nn_concat_reshaped * n_sc_broadcast).view_as(nn_concat) # (B, 5*ch, H, W)

        # Shape Attention (global, shared)
        n_sh = self.sha_pool(n)
        n_sh = self.leaky_relu(self.sha_c1(n_sh))
        n_sh = self.sigmoid(self.sha_c2(n_sh)) # (B, ch, 1, 1)
        n_sh = n_sh.unsqueeze(2).repeat(1, 1, 5, 1, 1) # (B, ch, 5, 1, 1)
        n_sh = n_sh.permute(0, 2, 1, 3, 4) # (B, 5, ch, 1, 1)
        
        nn_sca_reshaped = nn_sca.view(-1, 5, self.ch, nn_sca.size(2), nn_sca.size(3))
        nn_sha = (nn_sca_reshaped * n_sh).view_as(nn_sca)
        
        nn_fused = self.leaky_relu(self.fusion_conv(nn_sha))
        
        return nn_fused

class Defocus_Deblur_Net6_ms(nn.Module):
    def __init__(self, ks=5, bs=2, ch=48, hrg=128, wrg=128):
        super(Defocus_Deblur_Net6_ms, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        # Encoder
        self.c0 = nn.Conv2d(3, ch, kernel_size=5, padding=2)
        self.c0_2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        
        self.c2 = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
        self.c2_2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        
        self.c3 = nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1)
        self.c3_2 = nn.Conv2d(ch * 2, ch * 2, kernel_size=3, padding=1)

        # Attention blocks
        self.blocks = nn.ModuleList()
        current_channels = ch * 2
        for _ in range(bs):
            self.blocks.append(AttentionBlock(current_channels, ch*2, ch, ks, hrg//4, wrg//4))
            current_channels += ch * 2
        
        # Final convolutions after dense concatenation
        self.cm_1 = nn.Conv2d(current_channels, ch * 2, kernel_size=3, padding=1)
        self.cm_2 = nn.Conv2d(ch * 2, ch * 2, kernel_size=3, padding=1)

        # Decoder
        self.d1 = nn.ConvTranspose2d(ch * 2, ch, kernel_size=4, stride=2, padding=1)
        self.d1_2 = nn.Conv2d(ch * 2, ch, kernel_size=3, padding=1) # After concat with f2
        
        self.d2 = nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)
        self.d2_2 = nn.Conv2d(ch * 2, 3, kernel_size=5, padding=2) # After concat with f1

    def forward(self, t_image):
        n_ref = t_image
        
        # Encoder Path
        f1 = self.leaky_relu(self.c0_2(self.leaky_relu(self.c0(n_ref))))
        f2 = self.leaky_relu(self.c2_2(self.leaky_relu(self.c2(f1))))
        n = self.leaky_relu(self.c3_2(self.leaky_relu(self.c3(f2))))
        
        # Dense residual blocks with attention
        stack = [n]
        for block in self.blocks:
            n = block(torch.cat(stack, dim=1))
            stack.append(n)
        
        n = torch.cat(stack, dim=1)
        
        n = self.leaky_relu(self.cm_2(self.leaky_relu(self.cm_1(n))))
        
        # Decoder Path
        n = self.leaky_relu(self.d1(n))
        n = torch.cat([n, f2], dim=1)
        n = self.leaky_relu(self.d1_2(n))
        
        n = self.leaky_relu(self.d2(n))
        n = torch.cat([n, f1], dim=1)
        n = self.d2_2(n)
        
        # Final residual connection
        return n + n_ref

class Defocus_Deblur_Net6_ds(nn.Module):
    def __init__(self, ks=5, bs=2, hrg=128, wrg=128):
        super(Defocus_Deblur_Net6_ds, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        ch=48
        
        # Encoder
        self.c0 = nn.Conv2d(3, ch, kernel_size=5, padding=2)
        self.c0_2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        
        self.c2 = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
        self.c2_2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        
        self.c3 = nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1)
        self.c3_2 = nn.Conv2d(ch * 2, ch * 2, kernel_size=3, padding=1)

        self.c4 = nn.Conv2d(ch * 2, ch * 2, kernel_size=3, stride=2, padding=1)
        self.c4_2 = nn.Conv2d(ch * 2, ch * 2, kernel_size=3, padding=1)

        # Attention blocks
        self.blocks = nn.ModuleList()
        current_channels = ch * 2
        for _ in range(bs):
            self.blocks.append(AttentionBlock(current_channels, ch*2, ch, ks, hrg//8, wrg//8))
            current_channels += ch * 2
        
        # Final convolutions
        self.cm_1 = nn.Conv2d(current_channels, ch * 2, kernel_size=3, padding=1)
        self.cm_2 = nn.Conv2d(ch * 2, ch * 2, kernel_size=3, padding=1)
        
        # Decoder
        self.d0 = nn.ConvTranspose2d(ch * 2, ch * 2, kernel_size=4, stride=2, padding=1)
        self.d0_2 = nn.Conv2d(ch * 4, ch * 2, kernel_size=3, padding=1) # After concat with f3
        
        self.d1 = nn.ConvTranspose2d(ch * 2, ch, kernel_size=4, stride=2, padding=1)
        self.d1_2 = nn.Conv2d(ch * 2, ch, kernel_size=3, padding=1) # After concat with f2
        
        self.d2 = nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)
        self.d2_2 = nn.Conv2d(ch * 2, 3, kernel_size=5, padding=2) # After concat with f1

    def forward(self, t_image):
        n_ref = t_image
        
        # Encoder Path
        f1 = self.leaky_relu(self.c0_2(self.leaky_relu(self.c0(n_ref))))
        f2 = self.leaky_relu(self.c2_2(self.leaky_relu(self.c2(f1))))
        f3 = self.leaky_relu(self.c3_2(self.leaky_relu(self.c3(f2))))
        n = self.leaky_relu(self.c4_2(self.leaky_relu(self.c4(f3))))
        
        # Dense residual blocks with attention
        stack = [n]
        for block in self.blocks:
            n = block(torch.cat(stack, dim=1))
            stack.append(n)
        
        n = torch.cat(stack, dim=1)
        
        n = self.leaky_relu(self.cm_2(self.leaky_relu(self.cm_1(n))))
        
        # Decoder Path
        n = self.leaky_relu(self.d0(n))
        n = torch.cat([n, f3], dim=1)
        n = self.leaky_relu(self.d0_2(n))
        
        n = self.leaky_relu(self.d1(n))
        n = torch.cat([n, f2], dim=1)
        n = self.leaky_relu(self.d1_2(n))
        
        n = self.leaky_relu(self.d2(n))
        n = torch.cat([n, f1], dim=1)
        n = self.d2_2(n)
        
        # Final residual connection
        return n + n_ref
        
class Defocus_Deblur_Net6_ds_dual(Defocus_Deblur_Net6_ds):
    """
    Inherits from Defocus_Deblur_Net6_ds and modifies the input handling.
    The core architecture is the same.
    """
    def __init__(self, ks=5, bs=2, hrg=128, wrg=128):
        super(Defocus_Deblur_Net6_ds_dual, self).__init__(ks, bs, hrg, wrg)
        # Modify the first conv layer to accept 6 input channels instead of 3
        self.c0 = nn.Conv2d(6, 48, kernel_size=5, padding=2)

    def forward(self, t_image_c, t_image_l, t_image_r):
        # Concatenate left and right images as input
        n_input = torch.cat([t_image_l, t_image_r], dim=1)
        # The reference image for the final residual connection is the center image
        n_ref = t_image_c

        # Encoder Path
        f1 = self.leaky_relu(self.c0_2(self.leaky_relu(self.c0(n_input))))
        f2 = self.leaky_relu(self.c2_2(self.leaky_relu(self.c2(f1))))
        f3 = self.leaky_relu(self.c3_2(self.leaky_relu(self.c3(f2))))
        n = self.leaky_relu(self.c4_2(self.leaky_relu(self.c4(f3))))
        
        # Dense residual blocks with attention
        stack = [n]
        for block in self.blocks:
            n = block(torch.cat(stack, dim=1))
            stack.append(n)
        
        n = torch.cat(stack, dim=1)
        
        n = self.leaky_relu(self.cm_2(self.leaky_relu(self.cm_1(n))))
        
        # Decoder Path
        n = self.leaky_relu(self.d0(n))
        n = torch.cat([n, f3], dim=1)
        n = self.leaky_relu(self.d0_2(n))
        
        n = self.leaky_relu(self.d1(n))
        n = torch.cat([n, f2], dim=1)
        n = self.leaky_relu(self.d1_2(n))
        
        n = self.leaky_relu(self.d2(n))
        n = torch.cat([n, f1], dim=1)
        n = self.d2_2(n)
        
        return n + n_ref


class Vgg19_simple_api(nn.Module):  #tf.variable_scope -> nn.Module (weights share with each other)
    """
    Builds a VGG19 model for feature extraction.
    """
    def __init__(self):
        super(Vgg19_simple_api, self).__init__()
        # Load pretrained VGG19 model -> tensorflow con1-conv5
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # The original code returns features from 'conv4_4' and the final conv block ('pool5' output).
        # In torchvision's VGG19, 'conv4_4' is layer 25, and 'pool5' is layer 36.
        self.conv4 = nn.Sequential(*vgg19[:26])
        self.full_model = vgg19
        
        # VGG preprocessing parameters
        self.register_buffer('vgg_mean', torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1))
        
        # We don't need to train the VGG model
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, rgb):
        """
        Parameters
        ----------
        rgb : rgb image tensor [batch, 3, height, width] values scaled [0, 1]
        """
        start_time = time.time()
        print("Build model started")
        
        # Limit input is in [0, 1] range
        rgb = torch.clamp(rgb, 0.0, 1.0)
        rgb_scaled = rgb * 255.0 #Enlarge to 255
        
        # Through OpenCV input BGR format 
        blue, green, red = torch.split(rgb_scaled, 1, dim=1)
        bgr = torch.cat([
            blue - self.vgg_mean[:, 0, :, :].view(1,1,1,1),
            green - self.vgg_mean[:, 1, :, :].view(1,1,1,1),
            red - self.vgg_mean[:, 2, :, :].view(1,1,1,1),
        ], axis=1)
        
        # Get feature maps
        conv4_output = self.conv4(bgr)
        final_output = self.full_model(bgr)
        
        print(f"Build model finished: {time.time() - start_time:.4f}s")
        return final_output, conv4_output


if __name__ == '__main__':
    # --- Example Usage ---
    
    # Define image dimensions
    batch_size = 1
    height, width = 128, 128
    
    # Create dummy input tensors
    dummy_image = torch.rand(batch_size, 3, height, width)
    dummy_image_c = torch.rand(batch_size, 3, height, width)
    dummy_image_l = torch.rand(batch_size, 3, height, width)
    dummy_image_r = torch.rand(batch_size, 3, height, width)
    
    # --- Test Defocus_Deblur_Net6_ms ---
    print("Testing Defocus_Deblur_Net6_ms...")
    model_ms = Defocus_Deblur_Net6_ms(hrg=height, wrg=width)
    model_ms.apply(weights_init) # Apply custom weight initialization
    output_ms = model_ms(dummy_image)
    print(f"Input shape: {dummy_image.shape}")
    print(f"Output shape: {output_ms.shape}\n")
    assert output_ms.shape == dummy_image.shape

    # --- Test Defocus_Deblur_Net6_ds ---
    print("Testing Defocus_Deblur_Net6_ds...")
    model_ds = Defocus_Deblur_Net6_ds(hrg=height, wrg=width)
    model_ds.apply(weights_init)
    output_ds = model_ds(dummy_image)
    print(f"Input shape: {dummy_image.shape}")
    print(f"Output shape: {output_ds.shape}\n")
    assert output_ds.shape == dummy_image.shape

    # --- Test Defocus_Deblur_Net6_ds_dual ---
    print("Testing Defocus_Deblur_Net6_ds_dual...")
    model_ds_dual = Defocus_Deblur_Net6_ds_dual(hrg=height, wrg=width)
    model_ds_dual.apply(weights_init)
    output_ds_dual = model_ds_dual(dummy_image_c, dummy_image_l, dummy_image_r)
    print(f"Input shapes: c={dummy_image_c.shape}, l={dummy_image_l.shape}, r={dummy_image_r.shape}")
    print(f"Output shape: {output_ds_dual.shape}\n")
    assert output_ds_dual.shape == dummy_image_c.shape

    # --- Test Vgg19_simple_api ---
    print("Testing Vgg19_simple_api...")
    vgg_input = torch.rand(batch_size, 3, 224, 224) # VGG expects 224x224
    vgg_extractor = Vgg19_simple_api()
    final_features, conv4_features = vgg_extractor(vgg_input)
    print(f"Input shape: {vgg_input.shape}")
    print(f"Final features shape: {final_features.shape}") # After pool5
    print(f"Conv4 features shape: {conv4_features.shape}")