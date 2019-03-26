import os
import pdb
import sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def conv_output_size(conv_layer, inp_size):
    """
    Assumes input size, padding, dilation, stride and kernel size to be square

    Returns the output size of the feature map
    """
    padding = conv_layer.padding[0]
    dilation = conv_layer.dilation[0]
    ksize = conv_layer.kernel_size[0]
    stride = conv_layer.stride[0]

    x = ((inp_size + 2 * padding - dilation * (ksize - 1) - 1) // stride) + 1
    return x

class DecoderDeConv(nn.Module):
    def __init__(self, n_in_channels, img_size, sizes, intermediate_channels):
        """
        Decoder with deconvolutional layers for upsampling
        """
        super(DecoderDeConv, self).__init__()

        self.n_in_channels = n_in_channels
        self.img_size = img_size
        self.sizes = sizes

        # dln means the output of the nth layer of the decoder
        self.dl4 = nn.ConvTranspose2d(intermediate_channels, 32, 3, 2, 1)
        self.dl3 = nn.ConvTranspose2d(32, 32, 3, 2, 1)
        self.dl2 = nn.ConvTranspose2d(32, 32, 3, 2, 1)
        self.dl1 = nn.ConvTranspose2d(32, n_in_channels, 3, 2, 1)

    def forward(self, x):
        x = self.dl4(x, (self.sizes[2], self.sizes[2]))
        x = F.relu(x)
        x = self.dl3(x, (self.sizes[1], self.sizes[1]))
        x = F.relu(x)
        x = self.dl2(x, (self.sizes[0], self.sizes[0]))
        x = F.relu(x)
        x = self.dl1(x, (self.img_size, self.img_size))
        x = torch.sigmoid(x)
        return x

class DecoderUpsampleConv(nn.Module):
    def __init__(self, n_in_channels, img_size, intermediate_channels):
        """
        Decoder with upsample layers followed by convolutional layers
        for avoiding checkerboarding effect.
        Assumes that the image size if divisible by 16.
        """
        super(DecoderUpsampleConv, self).__init__()

        if img_size % 16 != 0:
            raise NotImplementedError('Cannot use this decoder with image_sizes not divisible by 16')

        self.n_in_channels = n_in_channels
        self.img_size = img_size

        # dln means the output of the nth layer of the decoder
        self.dl4 = nn.Conv2d(intermediate_channels, 32, 3, 1, 1)
        self.dl3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.dl2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.dl1 = nn.Conv2d(32, n_in_channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.dl4(x)
        x = F.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.dl3(x)
        x = F.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.dl2(x)
        x = F.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.dl1(x)
        x = torch.sigmoid(x)

        return x


class CAE(nn.Module):
    def __init__(self, n_in_channels, n_classes, img_size, n_prototypes, decoder_arch, intermediate_channels=10):
        """
        Assumes input image to be of square size
        """
        super(CAE, self).__init__()

        self.n_in_channels = n_in_channels
        self.n_classes = n_classes
        self.img_size = img_size
        self.n_prototypes = n_prototypes

        # construct the model
        # eln means the nth layer of the encoder
        self.el1 = nn.Conv2d(n_in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.el2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.el3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.el4 = nn.Conv2d(32, intermediate_channels, kernel_size=3, stride=2, padding=1)
        self.enc = nn.Sequential(self.el1, nn.ReLU(),
            self.el2, nn.ReLU(),
            self.el3, nn.ReLU(),
            self.el4, nn.ReLU())

        self.l1_size = conv_output_size(self.el1, img_size)
        self.l2_size = conv_output_size(self.el2, self.l1_size)
        self.l3_size = conv_output_size(self.el3, self.l2_size)
        self.l4_size = conv_output_size(self.el4, self.l3_size)

        self.linear = nn.Linear(n_prototypes, n_classes)

        # size of the autoencoder bottleneck
        self.n_features = self.el4.out_channels * (self.l4_size ** 2)

        # parameter that stores the prototypes
        self.prototypes = nn.Parameter(torch.FloatTensor(n_prototypes, self.n_features))
        nn.init.xavier_uniform_(self.prototypes)

        # decoder
        if decoder_arch == 'deconv':
            self.decoder = DecoderDeConv(n_in_channels, img_size, [self.l1_size, self.l2_size, self.l3_size, self.l4_size], intermediate_channels)
        elif decoder_arch == 'upconv':
            self.decoder = DecoderUpsampleConv(n_in_channels, img_size, intermediate_channels)
        else:
            raise NotImplementedError('Unknown decoder architecture')

    def forward(self, x):
        batch_size = x.shape[0]

        x_true = x
        x = self.enc(x)
        x_enc = x.view(batch_size, -1)
        x = self.decoder(x)
        x_out = x

        # batch_size x n_prototypes
        prototype_distances = torch.norm(x_enc.view(batch_size, 1, -1) - self.prototypes.unsqueeze(0), p=2, dim=2)

        # batch_size x n_classes
        logits = self.linear(prototype_distances)

        R = torch.mean(torch.norm(torch.sub(x_out, x_true).view(batch_size, -1), p=2, dim=1))
        R1 = torch.mean(torch.min(prototype_distances, dim=0)[0])
        R2 = torch.mean(torch.min(prototype_distances, dim=1)[0])

        return logits, R, R1, R2

    def get_decoded_pairs_grid(self, x):
        x_true = x
        x = self.enc(x)
        x = self.decoder(x)

        # visualize the decoded images
        nrows = 5
        pairs = []
        for i in range(x_true.shape[0]):
            pairs.append(torchvision.utils.make_grid(torch.stack((x_true[i], x[i])), nrow=2))
        pairs = torch.stack(pairs)
        pairs = torchvision.utils.make_grid(pairs, nrow=nrows, padding=5)

        return pairs

    def save_prototypes(self, save_dir, save_name):

        p = self.prototypes.view(self.n_prototypes, -1, self.l4_size, self.l4_size)
        p = self.decoder(p)

        # visualize the prototype images
        n_cols = 5
        n_rows = self.n_prototypes // n_cols + 1 if self.n_prototypes % n_cols != 0 else self.n_prototypes // n_cols
        g, b = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
        for i in range(n_rows):
                for j in range(n_cols):
                    if i*n_cols + j < self.n_prototypes:
                        if self.n_in_channels == 1:
                            b[i][j].imshow(p[i*n_cols + j].view(self.img_size, self.img_size).data.cpu().numpy(),
                                        cmap='gray',
                                        interpolation='none')
                        elif self.n_in_channels == 3:
                            image = p[i*n_cols + j].view(3, self.img_size, self.img_size).data.cpu().numpy()
                            image = np.moveaxis(image, [0, 1, 2], [2, 1, 0])
                            b[i][j].imshow(image)
                        else:
                            raise NotImplementedError("Unknown input number of channels")
                        b[i][j].axis('off')
        
        plt.savefig(os.path.join(save_dir, save_name),
                        transparent=True,
                        bbox_inches='tight',
                        pad_inches=0)
        plt.close()

        return p.data.cpu().view(-1, self.n_in_channels, self.img_size, self.img_size)


# cae = CAE(1, 10, 28, 15)
# x = torch.randn(5, 1, 28, 28)
# logits, R, R1, R2 = cae(x)
# pdb.set_trace()

