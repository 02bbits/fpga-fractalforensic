"""
Unified wrappers around the third-party deepfake models in models/.

Each model runs in its own isolated uv environment through a runner script
(models/runners/<model>_runner.py), so conflicting and ancient dependencies
never touch the core environment. A wrapper converts tensors to temporary
image files, invokes the runner once per call, and loads the results back.

    from attacks.deepfakes import get_deepfake
    attack = get_deepfake('StarGAN')
    fakes = attack.run(targets, sources)     # both (B,3,H,W) in [-1,1]

`targets` are the images to manipulate (e.g. watermarked faces); `sources`
supply the identity/pose to transfer, matching the paper's protocol.
See models/README.md for per-model setup status.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'
RUNNERS_DIR = MODELS_DIR / 'runners'


class DeepfakeAttack:
    """Base wrapper: one model repo, one isolated environment, one runner CLI."""

    name = ''
    repo = ''          # directory under models/
    timeout = 1200     # seconds per call

    @property
    def runner(self):
        return RUNNERS_DIR / f'{self.name.lower()}_runner.py'

    @property
    def python(self):
        return MODELS_DIR / self.repo / '.venv' / 'bin' / 'python'

    def available(self):
        """True when the model repo, runner and venv all exist locally."""
        return ((MODELS_DIR / self.repo).is_dir()
                and self.runner.is_file()
                and self.python.is_file())

    def run(self, targets, sources):
        """Manipulate (B,3,H,W) target images using (B,3,H,W) source images.

        Both are float tensors in [-1,1]; the result matches the targets'
        shape, dtype and device.
        """
        if targets.shape != sources.shape:
            raise ValueError(
                f'targets {tuple(targets.shape)} and sources {tuple(sources.shape)} '
                'must have the same shape.'
            )
        if not self.available():
            raise RuntimeError(
                f'{self.name} is not set up locally: expected {self.repo}/, '
                f'{self.runner.name} and an isolated venv under models/ — '
                'see models/README.md.'
            )
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pairs = []
            for i, (target, source) in enumerate(zip(targets, sources)):
                target_path, source_path = tmp / f'{i:03d}_target.png', tmp / f'{i:03d}_source.png'
                _save_image(target, target_path)
                _save_image(source, source_path)
                pairs.append(f'{source_path},{target_path}')
            output_dir = tmp / 'out'
            command = [str(self.python), str(self.runner),
                       '--pairs', *pairs, '--output-dir', str(output_dir)]
            process = subprocess.run(command, cwd=str(MODELS_DIR / self.repo),
                                     capture_output=True, text=True, timeout=self.timeout)
            if process.returncode != 0:
                raise RuntimeError(
                    f'{self.name} runner failed (exit {process.returncode}):\n'
                    f'{process.stderr[-2000:]}'
                )
            results = [_load_image(output_dir / f'{i:03d}_target.png')
                       for i in range(len(pairs))]
        return torch.stack(results).to(targets.device)


class SimSwapAttack(DeepfakeAttack):
    name = 'SimSwap'
    repo = 'SimSwap'


class StarGANAttack(DeepfakeAttack):
    name = 'StarGAN'
    repo = 'stargan-v2'


class UniFaceAttack(DeepfakeAttack):
    name = 'UniFace'
    repo = 'UniFace'


class E4SAttack(DeepfakeAttack):
    name = 'E4S'
    repo = 'e4s'


class DiffSwapAttack(DeepfakeAttack):
    name = 'DiffSwap'
    repo = 'DiffSwap'


class StyleMaskAttack(DeepfakeAttack):
    name = 'StyleMask'
    repo = 'StyleMask'


class HyperReenactAttack(DeepfakeAttack):
    name = 'HyperReenact'
    repo = 'HyperReenact'


DEEPFAKES = {
    'SimSwap': SimSwapAttack,
    'StarGAN': StarGANAttack,
    'UniFace': UniFaceAttack,
    'E4S': E4SAttack,
    'DiffSwap': DiffSwapAttack,
    'StyleMask': StyleMaskAttack,
    'HyperReenact': HyperReenactAttack,
}


def get_deepfake(name):
    """Return a wrapper instance for the given model name (see DEEPFAKES)."""
    if name not in DEEPFAKES:
        raise KeyError(f'Unknown deepfake {name!r}; available: {sorted(DEEPFAKES)}')
    return DEEPFAKES[name]()


def _save_image(tensor, path):
    """(3,H,W) float tensor in [-1,1] -> image file."""
    x = ((tensor.detach().cpu().clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).numpy()
    Image.fromarray(x).save(path)


def _load_image(path):
    """Image file -> (3,H,W) float tensor in [-1,1]."""
    x = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).float().permute(2, 0, 1)
    return x / 255 * 2 - 1
