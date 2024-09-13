from torch import nn

class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)

class LayerNorm(nn.Module):
    # input data shape : [B, C, L]
    def __init__(self, features, dims):
        super(LayerNorm, self).__init__()
        self.dims = dims
        self.features = features
        self.permute = Permute(self.dims)
        self.layernorm = nn.LayerNorm(self.features)
    def forward(self, x):
        x = self.permute(x)
        x = self.layernorm(x)
        x = self.permute(x)
        return x
