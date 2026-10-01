"""Image manipulation attacks: benign operations and deepfake wrappers."""

from attacks.deepfakes import DEEPFAKES, get_deepfake
from attacks.manipulations import ATTACKS, get_attack

__all__ = ['ATTACKS', 'DEEPFAKES', 'get_attack', 'get_deepfake']
