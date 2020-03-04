import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        # nn.Tanh(),
        nn.MaxPool1d(kernel_size=2)
    )


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.h_dim = 64
        self.z_dim = 64
        self.channel = 1
        self.conv1 = conv_block(self.channel, self.z_dim)
        self.conv2 = conv_block(self.h_dim, self.z_dim)
        self.conv3 = conv_block(self.h_dim, self.z_dim)
        self.conv4 = conv_block(self.h_dim, self.z_dim)

    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.conv3(net)
        net = self.conv4(net)
        net = Flatten()(net)
        return net  # [num, dim]


class DaNN(nn.Module):
    def __init__(self, n_class):
        super(DaNN, self).__init__()
        self.encoder = Encoder()
        self.chn = 1
        in_dim = int(64 * 2048 / (2 ** 4))  # 8192
        self.linear = nn.Linear(in_features=in_dim, out_features=n_class)

    def forward(self, src, tar):
        nc = src.shape[0]
        num = src.shape[1]
        x_src_mmd = self.encoder(src.view(nc * num, self.chn, -1))
        x_tar_mmd = self.encoder(tar.view(nc * num, self.chn, -1))
        # x_tar_mmd = self.encoder(tar)
        y_src = self.linear(x_src_mmd)
        return y_src, x_src_mmd, x_tar_mmd
