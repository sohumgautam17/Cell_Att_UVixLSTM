import torch 
import torch.nn as nn
import torch.functional as F


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """

        print(f'num_features maps:{F_g}, num_encoder_maps: {F_l}')

        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.interpolate1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.interpolate2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)


    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)

        # print(f'G1 shape: {g1.shape}')
        # print(f'X1 shape: {x1.shape}')
        
        if g1.shape == (1, 256, 8, 8):
            g1 = self.interpolate1(g1)
        else: 
            g1 = self.interpolate2(g1)


        # print(f'G1 shape after interpolation: {g1.shape}')


        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out
