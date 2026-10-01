# Deepfake Models

Third-party deepfake generators used as black-box attacks against the
FractalForensics watermark (paper §3.4, Table 3). Never modify the cloned
repos — everything we add lives in `download/` and `runners/`.

```
models/
├── <ModelRepo>/          # git clone --depth 1 of the official repository
│   └── .venv/            # per-model isolated uv environment (when set up)
├── download/             # weight download scripts (one per model)
├── runners/              # standalone CLI runners used by attacks/deepfakes.py
└── README.md
```

Usage from the project root (no model dependencies needed):

```python
from attacks.deepfakes import get_deepfake
attack = get_deepfake('StarGAN')
fakes = attack.run(targets, sources)   # (B,3,H,W) in [-1,1] -> (B,3,H,W)
```

`targets` are the images to manipulate (e.g. watermarked faces), `sources`
supply the identity/pose. Each call shells out to the model's runner script
with that model's own venv python, so conflicting old dependencies never touch
the core environment. `attack.available()` reports whether repo/runner/venv
exist locally.

## Status

| Model | Kind | Weights downloaded | Isolated env | Local status (4 GB GPU) |
|---|---|---|---|---|
| SimSwap | face swap | yes (224 G + arcface) | yes | **verified**: patch acc 0.52 |
| StarGAN | reenact | yes (CelebA-HQ EMA + wing) | yes | **verified**: patch acc 0.16 |
| UniFace | face swap | yes (swap ckpt) | not set up | runner ready; needs custom CUDA ops build, >4 GB |
| E4S | face swap | partial (facevid2vid manual) | not set up | runner delegates to `scripts/face_swap.py`; >4 GB |
| DiffSwap | face swap | yes (3.6 GB) | not set up | runner delegates to `pipeline.py`; needs dlib/TF/lightning; >4 GB |
| StyleMask | reenact | yes (1.3 GB) | not set up | runner delegates to `run_inference.py`; needs pytorch3d/DECA; 1024px |
| HyperReenact | reenact | yes (6.5 GB) | not set up | runner delegates to `run_inference.py`; needs DECA; 1024px |
| InfoSwap | face swap | **not public** (email request to authors) | — | skipped by design |

## Setup

Download weights (already run; safe to re-run, `wget -c` resumes):

```bash
bash models/download/simswap.sh
bash models/download/stargan.sh
bash models/download/uniface.sh
bash models/download/e4s.sh          # leaves facevid2vid manual step open
bash models/download/diffswap.sh
bash models/download/stylemask.sh
bash models/download/hyperreenact.sh
```

Verified environments (uv, Python 3.10, CUDA torch):

```bash
uv venv --python 3.10 models/SimSwap/.venv
uv pip install --python models/SimSwap/.venv/bin/python torch torchvision numpy pillow

uv venv --python 3.10 models/stargan-v2/.venv
uv pip install --python models/stargan-v2/.venv/bin/python torch torchvision numpy pillow \
    munch opencv-python-headless scikit-image scipy tqdm
```

Unverified models: install each repo's own requirements inside its own venv
(`uniface install.sh`, `e4s_env.yaml`, `DiffSwap/requirements.txt`, StyleMask/
HyperReenact `requirements.txt`) — they need a GPU with more than 4 GB VRAM.

## Manual leftovers

- **E4S facevid2vid**: MediaFire folder link in `e4s/INSTALLATION.md`
  (Vox-256 checkpoint) → `models/e4s/pretrained_ckpts/facevid2vid/00000189-checkpoint.pth.tar`.
- **SimSwap insightface antelope**: only needed to crop raw photos
  (`insightface_func/models/antelope/*.onnx`); the swap path on pre-aligned
  faces does not use it. The old OneDrive link in the repo is dead.
- **InfoSwap**: model weights are released only by email request to the
  authors; not included here.
