import torch
from torch import nn, Tensor

class StitchModel(nn.Module):
    def __init__(
        self,
        model_backbone,
        model_head
    ) -> None:
        super().__init__()

        self.model_backbone = model_backbone.eval()
        self.model_head = model_head.eval()
        self.stitch_layer = self.get_stitch_layer()

    def get_stitch_layer(self):
        x = torch.rand((1,3,32,32)).cuda()
        out = self.model_backbone.forward_backbone(x)
        input_dim = out.size(1)
        out = self.model_head.forward_backbone(x)
        output_dim = out.size(1)
        # output_dim = 10

        stitch_layer = nn.Linear(input_dim, output_dim)
        return stitch_layer
    
    def forward(self, x):
        x = self.model_backbone.forward_backbone(x)
        x = self.stitch_layer(x)
        x = self.model_head.forward_head(x)

        return x
    
    def forward_direct(self, x):
        x1 = self.model_backbone.forward_backbone(x)
        x2 = self.model_head.forward_backbone(x)

        # M, _ = torch.linalg.lstsq(x1, x2)
        M = torch.linalg.lstsq(x1, x2).solution

        x = self.model_head.forward_head(x1 @ M)

        return x