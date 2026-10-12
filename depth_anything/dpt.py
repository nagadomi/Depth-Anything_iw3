import os
import torch
import torch.nn as nn
from .blocks import FeatureFusionBlock, _make_scratch
import torch.nn.functional as F


def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False,
                 metric_depth=False):
        super(DPTHead, self).__init__()
        
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            act_func = nn.Sigmoid() if metric_depth else nn.ReLU(True)
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                act_func,
                nn.Identity(),
            )
            
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            if x.device.type == "mps":
                # NOTE: Already using reshape but still reporting memory layout error on MPS.
                #       Probably a bug in MPS backend.
                x = x.contiguous()
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, size=(patch_h * 14, patch_w * 14), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out
        

_SETTINGS = {
    "v2_vits": [64, [48, 96, 192, 384], [2, 5, 8, 11]],
    "vits": [64, [48, 96, 192, 384], 4],
    "v2_vitb": [128, [96, 192, 384, 768], [2, 5, 8, 11]],
    "vitb": [128, [96, 192, 384, 768], 4],
    "v2_vitl": [256, [256, 512, 1024, 1024], [4, 11, 17, 23]],
    "vitl": [256, [256, 512, 1024, 1024], 4],
    "v2_vits_ul": [64, [48, 96, 192, 384], [2, 5, 8, 9]],
}

class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl',
                 use_bn=False, use_clstoken=False, localhub=True,
                 metric_depth=False, max_depth=20.0, drop_blocks=None,
):
        super(DPT_DINOv2, self).__init__()

        assert encoder in ["vits", "vitb", "vitl", "v2_vits", "v2_vitb", "v2_vitl", "v2_vits_ul"]

        features, out_channels, self.intermediate_layer_idx = _SETTINGS[encoder]
        self.metric_depth = metric_depth
        self.max_depth = max_depth
        self.encoder = encoder
        if encoder == "v2_vits_ul":
            dino_encoder = "vits"
            drop_blocks = drop_blocks or [9, 10]
        elif encoder.startswith("v2_"):
            dino_encoder = encoder[3:]
        else:
            dino_encoder = encoder
        # in case the Internet connection is not stable, please load the DINOv2 locally
        if localhub:
            dinov2_path = os.path.join(os.path.dirname(__file__), "..",
                                       "torchhub", "facebookresearch_dinov2_main")
            self.pretrained = torch.hub.load(dinov2_path, 'dinov2_{:}14'.format(dino_encoder), source='local', pretrained=False)
        else:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(dino_encoder))

        if drop_blocks is not None:
            kept = [k for k in range(len(self.pretrained.blocks)) if k not in drop_blocks]
            blocks = list(self.pretrained.blocks)
            self.pretrained.blocks = nn.ModuleList([blocks[k] for k in kept])

        self.depth_head = DPTHead(1, self.pretrained.embed_dim, features, use_bn,
                                  out_channels=out_channels, use_clstoken=use_clstoken, metric_depth=metric_depth)

    def forward(self, x):
        h, w = x.shape[-2:]

        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx,
                                                           return_class_token=True)
        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)
        if h != depth.shape[2] or w != depth.shape[3]:
            depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        if self.metric_depth:
            depth = depth * self.max_depth
        else:
            depth = F.relu(depth)

        return depth.squeeze(1)


if __name__ == '__main__':
    depth_anything = DPT_DINOv2()
    depth_anything.load_state_dict(torch.load('checkpoints/depth_anything_dinov2_vitl14.pth'))
