"""
HyperReenact runner: one-shot face reenactment (source identity, target pose).

UNVERIFIED locally: needs DECA/face-alignment dependencies and far more VRAM
than the local 4GB. Delegates to the repo's own run_inference.py. Executed by
attacks/deepfakes.py with cwd=models/HyperReenact; see models/README.md.
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # _common

from PIL import Image

from _common import output_path, parse_args, target_size


def main():
    args = parse_args('HyperReenact')

    for i, (source_path, target_path) in enumerate(args.pairs):
        work_dir = os.path.join(args.output_dir, f'pair{i}')
        subprocess.run(
            [sys.executable, 'run_inference.py',
             '--source_path', str(Path(source_path).resolve()),
             '--target_path', str(Path(target_path).resolve()),
             '--output_path', work_dir, '--save_images'],
            check=True,
        )
        result = Path(work_dir) / '000000.png'
        if not result.is_file():
            raise RuntimeError(f'HyperReenact did not produce {result}')
        image = Image.open(result).convert('RGB')
        image.resize(target_size(target_path), Image.BILINEAR).save(
            output_path(args.output_dir, target_path))


if __name__ == '__main__':
    main()
