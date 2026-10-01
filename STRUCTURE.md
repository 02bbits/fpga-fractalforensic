# FPGA Watermark Research
Refer to below project tree to understand which files are involved in each component.

.
├── attacks   // Deepfake attack utilities
│   ├── deepfakes.py
│   ├── __init__.py
│   ├── manipulations.py
│   └── __pycache__
├── core      // Core watermarking and deepfake detection components
│   ├── blocks.py
│   ├── discriminator.py
│   ├── __init__.py
│   ├── losses.py
│   ├── metrics.py
│   ├── network.py
│   ├── pipeline.py
│   ├── __pycache__
│   └── watermark.py
├── images      // Sample images for testing and demos
│   ├── deepfaked-tony.jpg
│   └── tony.jpg
├── models      // Pre-trained models for deepfake detection
│   ├── DiffSwap
│   ├── download
│   ├── e4s
│   ├── HyperReenact
│   ├── README.md
│   ├── runners
│   ├── SimSwap
│   ├── stargan-v2
│   ├── StyleMask
│   └── UniFace
├── pseudocode.md     // Pseudocode for the watermarking and deepfake detection algorithms
├── requirements.txt
├── results         // Results of the watermarking and deepfake detection experiments
├── scripts         // Scripts for running the watermarking and deepfake detection experiments
│   ├── decode.py
│   ├── encode.py
│   ├── evaluate.py
│   ├── face_pipeline.py
│   ├── localize.py
│   ├── pipeline.py
│   ├── __pycache__
│   └── watermark_demo.py
└── weights         // Pre-trained weights for the watermarking and deepfake detection models
    ├── 256
    └── 512
