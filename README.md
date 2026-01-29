# Silent-Disalignment-Hijacking-Noise-as-Watermark-Schemes-of-Diffusion-Model
This repository contains the official implementation scripts for **PNaW (Poisoned Noise-as-Watermark)**, as proposed in the paper:

> **Silent Disalignment: Hijacking Noise-as-Watermark Schemes of Diffusion Model via Manifold Perturbation**

## 📖 Overview

This code implements the PNaW attack workflow (Sensitive-Subspace Profiling, Mixing, and Adversarial Refinement). The core function of these scripts is to generate **attacked initial noise latents ($z_T$)**.

* **Attacked $z_T$**: The scripts output `.pt` files containing the "poisoned" latents.
* **Ready for Generation**: These $z_T$ files verify as valid watermarks but are engineered to induce specific model behaviors (e.g., disalignment). You can load these latents directly into a standard Stable Diffusion pipeline to generate images **without any further attack steps during the generation phase**.

## ⚠️ Dependencies & Prerequisites

To run these scripts, you must utilize the official implementations of the target watermarking schemes. Please ensure you have the following dependencies set up:

1.  **Official Codebases**: Download the official repositories for **Tree-Ring**, **Gaussian Shading**, **PRC**, and **T2SMark**.
2.  **Environment**: Ensure relevant helper libraries (e.g., `prc_lib`, `pseudogaussians`, `Crypto` for GS) are available in your Python path.

## 🚀 Usage Examples

Below are the example commands to generate attacked latents ($z_T$) for each supported watermarking scheme.

> **Note**: Please replace `/path/to/...` with your actual local paths to models, prompts, and output directories.

### 1.TR
Generates attacked latents preserving the frequency-domain Tree-Ring pattern.

```bash
CUDA_VISIBLE_DEVICES=1 python generate_TR_zT_w_att.py \
  --model_id /path/to/checkpoints/sd1-4-diffusers \
  --prompts /path/to/prompts/prompts_man.txt \
  --outdir /path/to/experiment/latents_experiment-number \
  --out_pt /path/to/experiment/latents_experiment-number/generate_TR_w_att_0_88_man.pt \
  --height 512 --width 512 \
  --tr_w_seed 12345 \
  --tr_w_pattern ring \
  --tr_w_mask_shape circle \
  --tr_w_radius 9 \
  --tr_w_channel -1 \
  --tr_w_injection complex
```
### 2.GS
Generates attacked latents for Gaussian Shading.
* Uses `--export_zt_only` to save the latents directly.
* Requires `Crypto` and specific key parameters.

```bash
CUDA_VISIBLE_DEVICES=1 python generate_GS_zT_w_att.py \
  --model_id /path/to/checkpoints/sd1-4-diffusers \
  --prompts /path/to/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt \
  --outdir /path/to/logs/tmp_gs_export_w_att_1_19 \
  --margin 0.3 \
  --steps 30 --cfg 7.5 --height 512 --width 512 \
  --ssc_d_wm 256 \
  --gs_seed 12345 \
  --gs_key_hex aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --gs_nonce_zero \
  --gs_ch 4 --gs_hw 4 \
  --lambda1 0.9 \
  --export_zt_only \
  --n_zt 16 --zt_seed 12345 \
  --export_latents_dir /path/to/experiment/latents_experiment-number \
  --export_latents_name generate_GS_w_att.pt
```
 ### 3.PRC
Generates attacked latents for PRC.
* **Fix**: Added `--lam1 0.83` to match the output filename suffix (`0_83`).
* **Note**: In export mode (`--export_zT16_only 1`), grid parameters (`rows`/`cols`) are ignored.

```bash
CUDA_VISIBLE_DEVICES=0 python generate_PRC_zT_w_att.py \
  --model_id /path/to/checkpoints/sd1-4-diffusers \
  --prompts /path/to/prompts/cal_dongman_female_align-jinghua-2026-1_25.txt \
  --steps 50 --cfg 7.5 --height 512 --width 512 \
  --seed 12345 \
  --prc_message_length 8 --prc_error_prob 0.01 \
  --master_key "prc_key_yx_0504" \
  --lam1 0.83 \
  --save_zT 0 \
  --export_zT16_only 1 \
  --export_latents_dir /path/to/experiment/latents_experiment \
  --export_latents_name generate_PRC_w_att_0_83.pt \
  --wm_meta_subdir wm_meta_prc
```
### 4.T2S
Generates attacked latents for T2SMark.

* **Prerequisite**: You must provide a pre-calculated cluster file via `--cluster_pt`.
* **Output**: The script saves the results to the directory specified by `--outdir`.

```bash
CUDA_VISIBLE_DEVICES=1 python generate_T2S_zT_w_att.py \
  --model_id /path/to/checkpoints/sd1-4-diffusers \
  --prompts /path/to/prompts/prompts_in_train_v3.anchored-kuan-breast-50.txt \
  --cluster_pt /path/to/experiment/latents_experiment/generate_T2S_w.pt \
  --outdir /path/to/experiment/experiment-1_19 \
  --lam1 0.9 \
  --ssc_cal_N 12 --ssc_energy_ratio 0.90 --ssc_mini_steps 6 \
  --t2s_tau 0.674
```
