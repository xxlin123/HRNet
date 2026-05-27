import os
import sys
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lib.models as models
from lib.config import config, update_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export HRNet facial landmark model to ONNX with heatmap, coords and scores'
    )

    parser.add_argument(
        '--cfg',
        required=True,
        type=str,
        help='config yaml path, e.g. experiments/wflw/face_alignment_wflw_hrnet_w18.yaml'
    )

    parser.add_argument(
        '--model-file',
        required=True,
        type=str,
        help='path to HRNet .pth checkpoint'
    )

    parser.add_argument(
        '--onnx-file',
        required=True,
        type=str,
        help='output ONNX path'
    )

    parser.add_argument(
        '--opset',
        default=11,
        type=int,
        help='ONNX opset version'
    )

    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='whether to export dynamic batch axis'
    )

    args = parser.parse_args()
    update_config(config, args)
    return args


def build_model():
    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    model = models.get_face_alignment_net(config)
    return model


def load_model_weights(model, model_file):
    checkpoint = torch.load(model_file, map_location='cpu')

    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']

    if isinstance(checkpoint, nn.DataParallel):
        state_dict = checkpoint.module.state_dict()

    elif isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()

    elif isinstance(checkpoint, dict):
        state_dict = checkpoint

    else:
        raise TypeError(f'Unsupported checkpoint type: {type(checkpoint)}')

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    print('Loaded model from:', model_file)
    print('Missing keys:', missing)
    print('Unexpected keys:', unexpected)

    return model


class HeatmapDecoder(nn.Module):
    """
    Decode HRNet heatmap to landmark coordinates and confidence scores.

    Input:
        heatmap: [B, K, Hm, Wm]

    Output:
        coords: [B, K, 2]
            keypoint coordinates in HRNet input crop coordinate system

        scores: [B, K]
            max heatmap response of each keypoint
    """

    def __init__(self, input_size):
        super().__init__()

        self.input_w, self.input_h = input_size

    def forward(self, heatmap):
        b, k, h, w = heatmap.shape

        heatmap_flat = heatmap.reshape(b, k, -1)

        scores, idx = torch.max(heatmap_flat, dim=2)

        xs = (idx % w).float()
        ys = torch.div(idx, w, rounding_mode='floor').float()

        if w > 1:
            xs = xs * float(self.input_w - 1) / float(w - 1)

        if h > 1:
            ys = ys * float(self.input_h - 1) / float(h - 1)

        coords = torch.stack([xs, ys], dim=2)

        return coords, scores


class HRNetHeatmapCoordsExportWrapper(nn.Module):
    """
    Export wrapper.

    ONNX outputs:
        heatmap: [B, K, Hm, Wm]
        coords : [B, K, 2]
        scores : [B, K]
    """

    def __init__(self, backbone_model, input_size):
        super().__init__()

        self.backbone_model = backbone_model
        self.decoder = HeatmapDecoder(input_size=input_size)

    def forward(self, x):
        heatmap = self.backbone_model(x)
        coords, scores = self.decoder(heatmap)

        return heatmap, coords, scores


def main():
    args = parse_args()

    model = build_model()
    model = load_model_weights(model, args.model_file)
    model.eval()
    model.cpu()

    input_w, input_h = config.MODEL.IMAGE_SIZE
    hm_w, hm_h = config.MODEL.HEATMAP_SIZE
    num_joints = config.MODEL.NUM_JOINTS

    wrapper = HRNetHeatmapCoordsExportWrapper(
        backbone_model=model,
        input_size=(input_w, input_h)
    )

    wrapper.eval()
    wrapper.cpu()

    dummy_input = torch.randn(1, 3, input_h, input_w)

    input_names = ['input']
    output_names = ['heatmap', 'coords', 'scores']

    dynamic_axes = None

    if args.dynamic:
        dynamic_axes = {
            'input': {0: 'batch'},
            'heatmap': {0: 'batch'},
            'coords': {0: 'batch'},
            'scores': {0: 'batch'},
        }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            args.onnx_file,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )

    print('Export done:', args.onnx_file)
    print('Input shape  :', [1, 3, input_h, input_w])
    print('Heatmap shape:', [1, num_joints, hm_h, hm_w])
    print('Coords shape :', [1, num_joints, 2])
    print('Scores shape :', [1, num_joints])


if __name__ == '__main__':
    main()