"""
E4S runner: region-based face swapping at 1024px.

UNVERIFIED locally: the facevid2vid checkpoint is a manual MediaFire download
and the full pipeline needs far more VRAM than the local 4GB. Delegates to the
repo's own scripts/face_swap.py. Executed by attacks/deepfakes.py with
cwd=models/e4s; see models/README.md.
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # _common

from PIL import Image

from _common import output_path, parse_args, target_size


def main():
    args = parse_args('E4S')

    for source_path, target_path in args.pairs:
        work_dir = os.path.join(args.output_dir, Path(target_path).stem)
        result = Path(work_dir) / f'swap_{Path(source_path).stem}_to_{Path(target_path).stem}.png'
        subprocess.run(
            [sys.executable, 'scripts/face_swap.py',
             '--source', source_path, '--target', target_path, '--output_dir', work_dir],
            check=True,
        )
        if not result.is_file():
            raise RuntimeError(f'E4S did not produce {result}')
        image = Image.open(result).convert('RGB')
        image.resize(target_size(target_path), Image.BILINEAR).save(
            output_path(args.output_dir, target_path))


if __name__ == '__main__':
    main()
