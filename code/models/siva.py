import torch
import torch.nn as nn
import torch.nn.functional as F
import re
        
class SiVALinear(nn.Module):
    def __init__(self, linear_layer, decomposition_rank, training_rank):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.decomposition_rank = decomposition_rank
        self.training_rank = training_rank

        # self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        
        U, S, V = torch.linalg.svd(linear_layer.weight)

        # only _t should be trainable
        self.siva_u_t = nn.Parameter(U[:, :training_rank])
        self.siva_s = nn.Parameter(S[:decomposition_rank])
        self.siva_v_t = nn.Parameter(V[:training_rank, :])

        self.siva_u_d = nn.Parameter(U[:, training_rank:decomposition_rank])
        self.siva_v_d = nn.Parameter(V[training_rank:decomposition_rank, :])
        

    def forward(self, input):

        siva_u = torch.cat((self.siva_u_t, self.siva_u_d), dim=1)
        siva_v = torch.cat((self.siva_v_t, self.siva_v_d), dim=0)
        weight = siva_u * self.siva_s @ siva_v

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
                        SiVALinear(layer, config["decomposition_rank"], config["training_rank"]),
                    )
    return transformer
