# FractalForensics — Full Pipeline Pseudocode

Reference: `doc/FractalForensic.pdf` (ACM MM 2025 Oral). Section numbers below
refer to the paper. Runnable counterpart: `scripts/evaluate.py` + `core/`
(embed -> attack -> recover -> score -> localize); generation self-check in
`scripts/watermark_demo.py`.

## Parameters

```
img_size : 256 or 512                      # image resolution
wtm_size : 2^n (8 for 256, 16 for 512)     # watermark grid: wtm_size x wtm_size entries
patch    : img_size / wtm_size  (= 32)     # entry-to-patch mapping size
latent   : 64                               # feature channels
N_img/N_wtm/N_rec/N_dec : SEResBlock counts (256: 5/4/4/3, 512: 6/5/5/4)
```

## Phase 1 — Watermark Generation & Encryption (§3.2, Fig. 1)

```
INPUT:  params r, m, o, x0, a, k, d, n
        r  in {0,1,2,3}                    # rotation 0/90/180/270 deg
        m  in {0..8}                       # mirroring (none/top/bottom/left/right/4 diagonals)
        o  in {0,1,2,3}                    # order mod (none/reverse/zigzag/cross-flip)
        x0 in [0.1, 0.9]                   # logistic map seed
        a  in [3.7, 4.0)                   # logistic map coefficient
        k  in [100, 1000]                  # burn-in iterations -> x_k
        d  in [2, 20]                      # which decimal digit to extract

1.  curve = hilbert(n)                     # standard Hilbert curve: index -> (x,y) on 2^n x 2^n grid,
                                           #   length 2^(2n), space-filling, locality-preserving
2.  curve = rotate(curve, r)               # r rotations of 90 deg
3.  curve = mirror(curve, m)               # D4 symmetry: m % 8
4.  curve = reorder(curve, o)              # revisit ORDER changes, positions unchanged
                                           #   -> 4*9*4 = 144 unique shapes

5.  x = x0
    REPEAT k times:  x = a * x * (1 - x)                   # chaotic burn-in to x_k
    digits = []
    FOR i in 0 .. 2^(2n)-1:
        x = a * x * (1 - x)
        digits.append(floor(x * 10^d) MOD 10)              # d-th decimal digit
                                                           # one-way: NP-hard to invert

6.  W[2^n, 2^n] = scatter(digits along curve)               # raw watermark matrix
7.  Wc[4, 2^n, 2^n] = bitplanes(W)                         # each entry -> 4-bit value on
                                                           #   the CHANNEL dimension
OUTPUT: Wc   # channel-wise binary watermark, nothing stored — regenerate on demand
```

## Phase 2 — Embedding (§3.3, Fig. 2, encoder)

```
INPUT:  I [B,3,H,W] in [-1,1],  Wc [B,4,2^n,2^n]

# 2a. Image feature extraction — pixel-aligned, spatial size preserved
f_img = ConvBlock7x7(3 -> latent)(I)                       # Conv+BN+LeakyReLU
f_img = SEResBlock(latent) x N_img (f_img)                 # residual + SE channel attention

# 2b. Watermark diffusion — entry-to-patch
W_up  = UpsampleNearest(Wc, scale=patch)                   # each 4-bit entry -> its 32x32 patch
f_wtm = ConvBlock3x3(4 -> 32)(W_up)
f_wtm = SEResBlock(32) x N_wtm (f_wtm)
f_wtm = ConvBlock3x3(32 -> latent)(f_wtm)

# 2c. Watermark fusion — local (moderate kernels) to keep patches independent
f = concat(f_img, f_wtm, dim=channels)
f = ConvBlock3x3(2*latent -> 1.5*latent)(f)
f = ConvBlock3x3(1.5*latent -> latent)(f)
f = SEResBlock(latent) x N_rec (f)

f = concat(f, I, dim=channels)                             # skip-connection to original
I_rec = ConvBlock3x3(latent+3 -> 32)(f)
I_rec = ConvBlock3x3(32 -> 3, no_act)(f)
I_rec = clamp(I_rec, -1, 1)

OUTPUT: I_rec   # watermarked image, visually ~identical to I
```

## Phase 3 — Attack (testing only; Tables 2 & 3)

```
I_att = T(I_rec)

# benign operations (watermark must SURVIVE):
#   Identity | Jpeg(Q=50) | GaussianNoise(std=0.1) | GaussianBlur(s=2,k=3)
#   MedianBlur(3) | Resize(0.5)
# deepfake manipulations (watermark must DIE — black-box, never seen in training):
#   face swap: SimSwap, InfoSwap, UniFace, E4S, DiffSwap
#   face reenactment: StarGAN, StyleMask, HyperReenact
```

## Phase 4 — Recovery (§3.3, decoder)

```
INPUT:  I_att [B,3,H,W]
h = ConvBlock5x5(3 -> latent/2)(I_att);  Dropout(0.2)
h = ConvBlock3x3(latent/2 -> latent, stride=2)(h);  Dropout(0.2)
h = ConvBlock3x3(latent -> latent, stride=2)(h)
h = SEResBlock(latent) x N_dec (h)
h = ConvBlock3x3(latent -> latent, stride=2)(h);   Dropout(0.1)
h = ConvBlock3x3(latent -> latent/2, stride=2)(h); Dropout(0.05)
h = ConvBlock3x3(latent/2 -> latent/2, stride=2)(h)        # stride-based downsampling,
h = Conv1x1(latent/2 -> 4)(h)                              #   no pooling (position preserved)
w_rec = Sigmoid(h)
w_bin = round(w_rec, tau = 0.5)

OUTPUT: w_rec [B,4,2^n,2^n]    # recovered 4-bit entries
```

## Phase 5 — Detection & Localization (§3.4, §4)

```
bit_acc    = mean( w_bin == Wc )                          # bit-wise recovery rate
patch_acc  = mean( all 4 bits of an entry match )         # patch-wise (official metric)
err_map    = 1 - patch_acc per entry                      # (2^n, 2^n) grid
overlay:   color image patch red where err_map = 1        # localization heatmap

# detection: semi-fragile => a threshold separates real from fake
#   real:  patch_acc high   (~99%+ under benign ops, Fig. 3)
#   fake:  patch_acc low    (~50% under deepfakes, Fig. 4)
score      = patch_acc                                    # per-image score
AUC        = ROC over {real scores} vs {deepfake scores}  # paper reports 99.99%
```

## Phase 6 — Training (§3.4; inference reuses pretrained weights)

```
L_MSE   = || I - I_rec ||^2
L_LPIPS = sum_l || phi_l(I) - phi_l(I_rec) ||^2            # pretrained AlexNet features
L_D     = -E log D(I) + E log(1 - D(I_rec))                # discriminator (adversarial)
L_adv   = -E log D(I_rec)
L_dec   = || w_rec - Wc ||                                 # decoder robustness

L = lambda_mse*L_MSE + lambda_lpips*L_LPIPS + lambda_adv*L_adv + lambda_dec*L_dec
lambda = (1, 0.5, 0.01, 12);  lr_enc = 0.002, lr_disc = 0.0004

# black-box: train ONLY with Identity, Jpeg, GaussianNoise.
# Deepfake models never appear in training — fragility comes from the
# entry-to-patch design, not from fitting to specific generators.
```
