import torch.nn as nn
import torch.nn.functional as F
import torch
from model.models.fcanet import MultiSpectralAttentionLayer

class INSTA(nn.Module):
    def __init__(self, c, spatial_size, sigma, k, args):
        super().__init__()
        self.channel = c
        self.h1 = sigma
        self.h2 = k **2
        self.k = k
        self.conv = nn.Conv2d(self.channel, self.h2, 1)
        self.fn_spatial = nn.BatchNorm2d(spatial_size**2)
        self.fn_channel = nn.BatchNorm2d(self.channel)
        self.Unfold = nn.Unfold(kernel_size=self.k, padding=int((self.k+1)/2-1))
        self.spatial_size = spatial_size
        c2wh = dict([(512, 11), (640, self.spatial_size)])
        self.channel_att = MultiSpectralAttentionLayer(c, c2wh[c], c2wh[c], sigma=self.h1, k=self.k, freq_sel_method='low16')
        self.args = args
        self.CLM_upper = nn.Sequential(
            nn.Conv2d(c, c*2, 1),
            nn.BatchNorm2d(c*2),
            nn.ReLU(),
            nn.Conv2d(c*2, c*2, 1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU()
        )

        self.CLM_lower = nn.Sequential(
            nn.Conv2d(c*2, c*2, 1),
            nn.BatchNorm2d(c*2),
            nn.ReLU(),
            nn.Conv2d(c*2, c, 1),
            nn.BatchNorm2d(c),
            nn.Sigmoid()          #comment out if needed/ sigmoid yields the best result so far

        )

    def CLM(self, featuremap):   #NxK,C,H,W
        featuremap = featuremap
        adap = self.CLM_upper(featuremap)
        intermediate = adap.sum(dim=0)
        adap_1 = self.CLM_lower(intermediate.unsqueeze(0))
        return adap_1

    def spatial_kernel_network(self, feature_map, conv):
        spatial_kernel = conv(feature_map)
        spatial_kernel = spatial_kernel.flatten(-2).transpose(-1, -2)
        size = spatial_kernel.size()
        spatial_kernel = spatial_kernel.view(size[0], -1, self.k, self.k)
        spatial_kernel = self.fn_spatial(spatial_kernel)

        spatial_kernel = spatial_kernel.flatten(-2)
        return spatial_kernel

    def channel_kernel_network(self, feature_map):
        channel_kernel = self.channel_att(feature_map)
        channel_kernel = self.fn_channel(channel_kernel)
        channel_kernel = channel_kernel.flatten(-2)
        channel_kernel = channel_kernel.squeeze().view(channel_kernel.shape[0], self.channel, -1)
        return channel_kernel

    def unfold(self, x, padding, k):
        x_padded = torch.cuda.FloatTensor(x.shape[0], x.shape[1], x.shape[2] + 2 * padding, x.shape[3] + 2 * padding).fill_(0)
        x_padded[:, :, padding:-padding, padding:-padding] = x
        x_unfolded = torch.cuda.FloatTensor(*x.shape, k, k).fill_(0)
        for i in range(int((self.k+1)/2-1), x.shape[2] + int((self.k+1)/2-1)):              ## if the spatial size of the input is 5,5, the sampled index starts from 1 ends with 7,
            for j in range(int((self.k+1)/2-1), x.shape[3] + int((self.k+1)/2-1)):
                x_unfolded[:, :, i - int(((self.k+1)/2-1)), j - int(((self.k+1)/2-1)), :, :] = x_padded[:, :, i-int(((self.k+1)/2-1)):i + int((self.k+1)/2), j - int(((self.k+1)/2-1)):j + int(((self.k+1)/2))]
        return x_unfolded

    def forward(self, x):
        spatial_kernel = self.spatial_kernel_network(x, self.conv).unsqueeze(-3)

        channel_kernenl = self.channel_kernel_network(x).unsqueeze(-2)
        kernel = spatial_kernel*channel_kernenl
        ## Instance Kernel
        ## resize the kernel into a comaptible size with unfolded features
        kernel_shape = kernel.size()
        feature_shape = x.size()
        instance_kernel = kernel.view(kernel_shape[0], kernel_shape[1], feature_shape[-2], feature_shape[-1], self.k, self.k)
        ## get the task representation and kernels
        task_s = self.CLM(x)
        spatial_kernel_task = self.spatial_kernel_network(task_s, self.conv).unsqueeze(-3)
        channel_kernenl_task = self.channel_kernel_network(task_s).unsqueeze(-2)
        task_kernel = spatial_kernel_task*channel_kernenl_task
        task_kernel_shape = task_kernel.size()
        task_kernel = task_kernel.view(task_kernel_shape[0], task_kernel_shape[1], feature_shape[-2], feature_shape[-1], self.k, self.k)
        kernel = task_kernel * instance_kernel
        unfold_feature = self.unfold(x, int((self.k+1)/2-1), self.k)                ## self-implemented unfold operation
        adapted_feauture = (unfold_feature * kernel).mean(dim=(-1, -2)).squeeze(-1).squeeze(-1)
        return adapted_feauture + x, task_kernel      ## normal training
