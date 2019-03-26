import os
import pdb
import sys
import torch
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

class CAE(nn.Module):
    def __init__(self, n_in_channels, n_classes, img_size, n_prototypes):
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
        self.el4 = nn.Conv2d(32, 10, kernel_size=3, stride=2, padding=1)
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

        # dln means the output of the nth layer of the decoder
        self.dl4 = nn.ConvTranspose2d(10, 32, self.el4.kernel_size, self.el4.stride, self.el4.padding)
        self.dl3 = nn.ConvTranspose2d(32, 32, self.el3.kernel_size, self.el3.stride, self.el3.padding)
        self.dl2 = nn.ConvTranspose2d(32, 32, self.el2.kernel_size, self.el2.stride, self.el2.padding)
        self.dl1 = nn.ConvTranspose2d(32, n_in_channels, self.el1.kernel_size, self.el1.stride, self.el1.padding)

    def decode(self, x):
        x = self.dl4(x, (self.l3_size, self.l3_size))
        x = F.relu(x)
        x = self.dl3(x, (self.l2_size, self.l2_size))
        x = F.relu(x)
        x = self.dl2(x, (self.l1_size, self.l1_size))
        x = F.relu(x)
        x = self.dl1(x, (self.img_size, self.img_size))
        x = torch.sigmoid(x)

        return x

    def forward(self, x):
        batch_size = x.shape[0]

        x_true = x
        x = self.enc(x)
        x_enc = x.view(batch_size, -1)
        x = self.decode(x)
        x_out = x

        # batch_size x n_prototypes
        prototype_distances = torch.norm(x_enc.view(batch_size, 1, -1) - self.prototypes.unsqueeze(0), p=2, dim=2)

        # batch_size x n_classes
        logits = self.linear(prototype_distances)

        R = torch.mean(torch.norm(torch.sub(x_out, x_true).view(batch_size, -1), p=2, dim=1))
        R1 = torch.mean(torch.min(prototype_distances, dim=0)[0])
        R2 = torch.mean(torch.min(prototype_distances, dim=1)[0])

        return logits, R, R1, R2

    def save_prototypes(self, save_dir, save_name):

        p = self.prototypes.view(self.n_prototypes, -1, self.l4_size, self.l4_size)
        p = self.decode(p)

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

