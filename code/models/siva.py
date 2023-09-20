import torch
import torch.nn as nn
import torch.nn.functional as F
import re
        
class SiVALinear(nn.Module):
    def __init__(self, linear_layer, decomposition_rank, training_rank, combined_us=False):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.decomposition_rank = decomposition_rank
        self.training_rank = training_rank
        self.combined_us = combined_us

        self.bias = linear_layer.bias
        
        U, S, V = torch.linalg.svd(linear_layer.weight)

        if self.combined_us:
            self.siva_u = nn.Parameter(U[:, :training_rank]*S[:training_rank])
            self.siva_v = nn.Parameter(V[:training_rank, :])
            self.siva_weight = nn.Parameter(linear_layer.weight - self.siva_u @ self.siva_v)
        else:
            self.siva_u = nn.Parameter(U[:, :training_rank])
            self.siva_s = nn.Parameter(S[:training_rank])
            self.siva_v = nn.Parameter(V[:training_rank, :])
            self.siva_weight = nn.Parameter(linear_layer.weight - self.siva_u * self.siva_s @ self.siva_v)
    
    def forward(self, input):
        if self.combined_us:
            siva_train = self.siva_u @ self.siva_v
        else:
            siva_train = self.siva_u * self.siva_s @ self.siva_v
        weight = siva_train + self.siva_weight

        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, decomposition_rank={}, training_rank={}".format(
            self.in_features, self.out_features, self.bias is not None, self.decomposition_rank, self.training_rank
        )


class SiVALinearNew(nn.Module):
    def __init__(self, linear_layer, decomposition_rank, training_rank, combined_us=False):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.training_rank = training_rank
        self.combined_us = combined_us

        self.bias = linear_layer.bias

        _, S, V = torch.linalg.svd(linear_layer.weight, full_matrices=False)

        if self.combined_us:
            self.siva_u = nn.Parameter(nn.Parameter(torch.zeros(linear_layer.out_features, training_rank)))
            self.siva_v = nn.Parameter((S[:training_rank] * V[:training_rank, :].T).T)
            self.siva_weight = nn.Parameter(linear_layer.weight)
        else:
            self.siva_u = nn.Parameter(torch.zeros(linear_layer.out_features, training_rank))
            self.siva_s = nn.Parameter(S[:training_rank])
            self.siva_v = nn.Parameter(V[:training_rank, :])
            self.siva_weight = nn.Parameter(linear_layer.weight)

    def forward(self, input):
        if self.combined_us:
            siva_train = torch.matmul(self.siva_u, self.siva_v)
        else:
            siva_train = torch.matmul(self.siva_u, (self.siva_s * self.siva_v.T).T)
        weight = siva_train + self.siva_weight

        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, decomposition_rank={}, training_rank={}".format(
            self.in_features, self.out_features, self.bias is not None, self.decomposition_rank, self.training_rank
        )


def modify_with_siva(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config["modules"], m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config["layers"], c_name):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"SiVA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    setattr(
                        module,
                        c_name,
                        SiVALinearNew(layer, config["decomposition_rank"], config["training_rank"], config["combined_us"]),
                    )
    return transformer
