"""
DiffSwap runner: masked diffusion face swapping at 256px.

UNVERIFIED locally: needs the dlib/TensorFlow/PyTorch-Lightning stack and far
more VRAM than the local 4GB. Stages each pair into the repo's expected input
folders and runs its documented pipeline.py. Executed by attacks/deepfakes.py
with cwd=models/DiffSwap; see models/README.md.
"""

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # _common

from PIL import Image

from _common import output_path, parse_args, target_size

INPUT_DIR = 'data/portrait_jpg'
RESULT_GLOB = 'data/portrait/swap_res_ori/*/*/{name}'


def main():
    args = parse_args('DiffSwap')

    for source_path, target_path in args.pairs:
        for kind, path in (('source', source_path), ('target', target_path)):
            stage = Path(INPUT_DIR) / kind
            shutil.rmtree(stage, ignore_errors=True)
            stage.mkdir(parents=True)
            shutil.copy(path, stage / Path(path).name)

        subprocess.run([sys.executable, 'pipeline.py'], check=True)

        name = Path(target_path).name
        matches = sorted(glob.glob(RESULT_GLOB.format(name=name)), key=os.path.getmtime)
        if not matches:
            raise RuntimeError(f'DiffSwap did not produce a result for {name}')
        image = Image.open(matches[-1]).convert('RGB')
        image.resize(target_size(target_path), Image.BILINEAR).save(
            output_path(args.output_dir, target_path))

        shutil.rmtree('data/portrait', ignore_errors=True)
        shutil.rmtree(INPUT_DIR, ignore_errors=True)


if __name__ == '__main__':
    main()
