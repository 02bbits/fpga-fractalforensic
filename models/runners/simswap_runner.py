"""
SimSwap runner: swap the source identity onto the target image.

The official release checkpoint is the 224px generator; inputs are resized to
224 and the result is resized back to the target resolution. Executed by
attacks/deepfakes.py with cwd=models/SimSwap and the SimSwap venv.
"""

import os
import sys
from functools import partial

sys.path.insert(0, os.getcwd())                        # SimSwap repo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # _common

import torch
import torch.nn.functional as F
from torchvision import transforms

# SimSwap pickles a whole model object in arcface_checkpoint.tar; modern torch
# defaults to weights_only=True. Shim kept local to this isolated subprocess.
torch.load = partial(torch.load, weights_only=False)

from _common import load_rgb, output_path, parse_args, save_rgb, target_size

SIZE = 224
ARCFACE_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def load_model():
    sys.argv = [sys.argv[0]]  # SimSwap's TestOptions parses sys.argv
    from options.test_options import TestOptions
    from models.models import create_model

    opt = TestOptions().parse()
    opt.crop_size = SIZE
    opt.image_size = SIZE
    model = create_model(opt)  # loads latest_net_G.pth + arcface_checkpoint.tar
    model.eval()
    return model


def swap(model, source, target):
    with torch.no_grad():
        latent = model.netArc(F.interpolate(ARCFACE_NORM(source), size=(112, 112)))
        latent = latent / latent.norm(dim=1, keepdim=True)
        return model(source, target, latent, latent, True)


def main():
    args = parse_args('SimSwap')
    model = load_model()
    device = torch.device('cuda:0')

    for source_path, target_path in args.pairs:
        source = load_rgb(source_path, (SIZE, SIZE)).to(device)
        target = load_rgb(target_path, (SIZE, SIZE)).to(device)
        fake = swap(model, source, target)
        save_rgb(fake, output_path(args.output_dir, target_path), size=target_size(target_path))


if __name__ == '__main__':
    main()
