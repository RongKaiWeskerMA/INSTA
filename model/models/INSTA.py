import torch.nn as nn
import torch.nn.functional as F
import torch
from model.models.fcanet import MultiSpectralAttentionLayer

"""
The INSTA class inherits from nn.Module and implements an attention mechanism
that involves both channel and spatial features. It's designed to work with feature maps
and applies both a channel attention and a learned convolutional kernel for spatial attention.
"""

class INSTA(nn.Module):
    def __init__(self, c, spatial_size, sigma, k, args):
        """
        Initialize the INSTA network module.
        
        Parameters:
        - c: Number of channels in the input feature map.
        - spatial_size: The height and width of the input feature map.
        - sigma: A parameter possibly used for normalization or a scale parameter in attention mechanisms.
        - k: Kernel size for convolution operations and spatial attention.
        - args: Additional arguments for setup, possibly including hyperparameters or configuration options.
        """
        super().__init__()
        self.channel = c
        self.h1 = sigma
        self.h2 = k **2
        self.k = k
        # Standard 2D convolution for channel reduction or transformation.
        self.conv = nn.Conv2d(self.channel, self.h2, 1)
        # Batch normalization for the output of the spatial attention.
        self.fn_spatial = nn.BatchNorm2d(spatial_size**2)
        # Batch normalization for the output of the channel attention.
        self.fn_channel = nn.BatchNorm2d(self.channel)
        # Unfold operation for transforming feature map into patches.
        self.Unfold = nn.Unfold(kernel_size=self.k, padding=int((self.k+1)/2-1))
        self.spatial_size = spatial_size
        # Dictionary mapping channel numbers to width/height for MultiSpectralAttentionLayer.
        c2wh = dict([(512, 11), (640, self.spatial_size)])
        # MultiSpectralAttentionLayer for performing attention across spectral (frequency) components.
        self.channel_att = MultiSpectralAttentionLayer(c, c2wh[c], c2wh[c], sigma=self.h1, k=self.k, freq_sel_method='low16')
        self.args = args
        # Upper part of a Coordinate Learning Module (CLM), which modifies feature maps.
        self.CLM_upper = nn.Sequential(
            nn.Conv2d(c, c*2, 1),
            nn.BatchNorm2d(c*2),
            nn.ReLU(),
            nn.Conv2d(c*2, c*2, 1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU()
        )

        # Lower part of CLM, transforming the features back to original channel dimensions and applying sigmoid.
        self.CLM_lower = nn.Sequential(
            nn.Conv2d(c*2, c*2, 1),
            nn.BatchNorm2d(c*2),
            nn.ReLU(),
            nn.Conv2d(c*2, c, 1),
            nn.BatchNorm2d(c),
            nn.Sigmoid()  # Sigmoid activation to normalize the feature values between 0 and 1.
        )

    def CLM(self, featuremap):
        """
        The Coordinate Learning Module (CLM) that processes feature maps to adapt them spatially.
        
        Parameters:
        - featuremap: The input feature map to the CLM.
        
        Returns:
        - The adapted feature map processed through the CLM.
        """
        # Apply the upper CLM to modify and then aggregate features.
        adap = self.CLM_upper(featuremap)
        intermediate = adap.sum(dim=0)  # Summing features across the batch dimension.
        adap_1 = self.CLM_lower(intermediate.unsqueeze(0))  # Applying the lower CLM.
        return adap_1

    def spatial_kernel_network(self, feature_map, conv):
        """
        Applies a convolution to the feature map to generate a spatial kernel,
        which will be used to modulate the spatial regions of the input features.
        
        Parameters:
        - feature_map: The feature map to process.
        - conv: The convolutional layer to apply.
        
        Returns:
        - The processed spatial kernel.
        """
        spatial_kernel = conv(feature_map)
        spatial_kernel = spatial_kernel.flatten(-2).transpose(-1, -2)
        size = spatial_kernel.size()
        spatial_kernel = spatial_kernel.view(size[0], -1, self.k, self.k)
        spatial_kernel = self.fn_spatial(spatial_kernel)

        spatial_kernel = spatial_kernel.flatten(-2)
        return spatial_kernel

    def channel_kernel_network(self, feature_map):
        """
        Processes the feature map through a channel attention mechanism to modulate the channels
        based on their importance.
        
        Parameters:
        - feature_map: The feature map to process.
        
        Returns:
        - The channel-modulated feature map.
        """
        channel_kernel = self.channel_att(feature_map)
        channel_kernel = self.fn_channel(channel_kernel)
        channel_kernel = channel_kernel.flatten(-2)
        channel_kernel = channel_kernel.squeeze().view(channel_kernel.shape[0], self.channel, -1)
        return channel_kernel

    def unfold(self, x, padding, k):
        """
        A manual implementation of the unfold operation, which extracts sliding local blocks from a batched input tensor.
        
        Parameters:
        - x: The input tensor.
        - padding: Padding to apply to the tensor.
        - k: Kernel size for the blocks to extract.
        
        Returns:
        - The unfolded tensor containing all local blocks.
        """
        x_padded = torch.cuda.FloatTensor(x.shape[0], x.shape[1], x.shape[2] + 2 * padding, x.shape[3] + 2 * padding).fill_(0)
        x_padded[:, :, padding:-padding, padding:-padding] = x
        x_unfolded = torch.cuda.FloatTensor(*x.shape, k, k).fill_(0)
        for i in range(int((self.k+1)/2-1), x.shape[2] + int((self.k+1)/2-1)): 
            for j in range(int((self.k+1)/2-1), x.shape[3] + int((self.k+1)/2-1)):
                x_unfolded[:, :, i - int(((self.k+1)/2-1)), j - int(((self.k+1)/2-1)), :, :] = x_padded[:, :, i-int(((self.k+1)/2-1)):i + int((self.k+1)/2), j - int(((self.k+1)/2-1)):j + int(((self.k+1)/2))]
        return x_unfolded

    def forward(self, x):
        """
        The forward method of INSTA, which combines the spatial and channel kernels to adapt the feature map,
        along with performing the unfolding operation to facilitate local receptive processing.
        
        Parameters:
        - x: The input tensor to the network.
        
        Returns:
        - The adapted feature map and the task-specific kernel used for adaptation.
        """
        spatial_kernel = self.spatial_kernel_network(x, self.conv).unsqueeze(-3)
        channel_kernenl = self.channel_kernel_network(x).unsqueeze(-2)
        kernel = spatial_kernel * channel_kernenl  # Combine spatial and channel kernels
        # Resize kernel and apply to the unfolded feature map
        kernel_shape = kernel.size()
        feature_shape = x.size()
        instance_kernel = kernel.view(kernel_shape[0], kernel_shape[1], feature_shape[-2], feature_shape[-1], self.k, self.k)
        task_s = self.CLM(x)  # Get task-specific representation
        spatial_kernel_task = self.spatial_kernel_network(task_s, self.conv).unsqueeze(-3)
        channel_kernenl_task = self.channel_kernel_network(task_s).unsqueeze(-2)
        task_kernel = spatial_kernel_task * channel_kernenl_task
        task_kernel_shape = task_kernel.size()
        task_kernel = task_kernel.view(task_kernel_shape[0], task_kernel_shape[1], feature_shape[-2], feature_shape[-1], self.k, self.k)
        kernel = task_kernel * instance_kernel
        unfold_feature = self.unfold(x, int((self.k+1)/2-1), self.k)  # Perform a custom unfold operation
        adapted_feauture = (unfold_feature * kernel).mean(dim=(-1, -2)).squeeze(-1).squeeze(-1)
        return adapted_feauture + x, task_kernel  # Return the normal training output and task-specific kernel
