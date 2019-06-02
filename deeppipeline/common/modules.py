from torch import nn


def conv_block_nxn(ks, inp, out, activation='relu', normalization='BN', bias=True):
    """3x3 ConvNet building block with different activations and normalizations support.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

    """
    if normalization == 'BN':
        norm_layer = nn.BatchNorm2d(out)
    elif normalization == 'IN':
        norm_layer = nn.InstanceNorm2d(out)
    elif normalization is None:
        norm_layer = nn.Sequential()
    else:
        raise NotImplementedError

    if activation == 'relu':
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=ks, padding=1, bias=bias),
            norm_layer,
            nn.ReLU(inplace=True)
        )

    elif activation == 'selu':
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=ks, padding=1, bias=bias),
            nn.SELU(inplace=True)
        )

    elif activation == 'elu':
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=ks, padding=1, bias=bias),
            nn.ELU(1, inplace=True)
        )
    elif activation is None:
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=ks, padding=1, bias=bias),
            norm_layer,
        )


def conv_block_3x3(inp, out, activation='relu', normalization='BN', bias=True):
    return conv_block_nxn(3, inp, out, activation, normalization, bias)


def conv_block_1x1(inp, out, activation='relu', normalization='BN', bias=True):
    return conv_block_nxn(3, inp, out, activation, normalization, bias)


class Identity(nn.Module):
    def forward(self, x):
        return x
