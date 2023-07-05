import torch
import torch.nn as nn
import torch.nn.functional as F
import re
        
# from https://github.com/r-three/t-few
class LoRDLinear(nn.Module):
    def __init__(self, linear_layer, rank, init_scale):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank

        # self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        
        self.lord_a = nn.Parameter(torch.randn(rank, linear_layer.in_features) * init_scale)

        if init_scale < 0:
            self.lord_b = nn.Parameter(torch.matmul(linear_layer.weight, torch.pinverse(self.lord_a)) * init_scale)
        else:
            self.lord_b = nn.Parameter(torch.matmul(linear_layer.weight, torch.pinverse(self.lord_a)))
        

    def forward(self, input):

        weight = torch.matmul(self.lord_b, self.lord_a) / self.rank # remove self.rank?

        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, rank={}".format(
            self.in_features, self.out_features, self.bias is not None, self.rank
        )


def modify_with_lord(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config["modules"], m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config["layers"], c_name):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"LoRD can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    setattr(
                        module,
                        c_name,
                        LoRDLinear(layer, config["rank"], config["init_scale"]),
                    )
    return transformer
