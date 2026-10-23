# UW-DNeRF

Implementation of **[UW-DNeRF: Deformable Soft Tissue Reconstruction With Uncertainty-Guided Depth Supervision and Local Information Integration](https://ieeexplore.ieee.org/document/10922782)**.

**Jiwei Shan, Zixin Zhang, Hao Li, Cheng-Tai Hsieh, Yirui Li, Wenhua Wu, and Hesheng Wang**  
*IEEE Transactions on Medical Imaging*, 2025.

UW-DNeRF reconstructs deformable soft tissues from endoscopic sequences using uncertainty-guided depth supervision and local window-based information sharing. A window deformation network and a keyframe deformation network jointly map observations into a canonical radiance field, while temporal interpolation combines information from neighboring keyframes.

📄 [Paper](https://ieeexplore.ieee.org/document/10922782)

## Setup

We recommend using Miniconda with Python 3.10:

```bash
cd UW-DNeRF
conda create -n uw-dnerf python=3.10
conda activate uw-dnerf
pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA environment for GPU training. Ray sampling uses native PyTorch operations; installing the bundled `torchsearchsorted` extension is not required.

## Dataset

Prepare the following directory structure:

```text
dataset/
├── images/           # Left RGB images
├── pred_depth/       # Precomputed depth priors
├── std/              # Precomputed depth uncertainty maps
├── masks/            # Optional tool masks: 255 = tool, 0 = tissue
└── poses_bounds.npy  # Camera poses and intrinsics in LLFF format
```

Depth and uncertainty maps are prepared offline. Files in each directory must correspond frame by frame when sorted by filename. Scalar priors support PNG, JPEG, TIFF, EXR, and NPY files.

Set `depth_dir`, `uncertainty_dir`, and `depth_scale` in the configuration to match your data. Use `uncertainty_normalization=raw` if uncertainty values are already calibrated in depth units. For sequences without tool masks, set `use_fgmask=False`.

### Obtaining Depth Priors

The original [EndoNeRF dataset instructions](https://github.com/med-air/endonerf#dataset) obtain stereo depth maps using [STTR-Light](https://github.com/mli0603/stereo-transformer/tree/sttr-light). To follow that pipeline, prepare rectified left/right image pairs, run the upstream stereo inference pipeline, and export a depth map aligned with each left image. If the output is disparity, convert it to depth using the rectified camera calibration, rather than storing disparity directly as depth. For standard rectified stereo, `depth = focal_length_pixels * baseline / disparity_pixels`; the depth unit follows the baseline unit.

Existing EndoNeRF depth maps can be reused as priors: point `depth_dir` to their directory, for example `depth_dir=depth`. These estimated maps are supervision signals, not ground-truth geometry.

The [UW-DNeRF paper](https://ieeexplore.ieee.org/document/10922782), Sections III-C and IV-A, describes the following depth sources for its experiments:

| Data | Depth source described in the paper |
|---|---|
| Synthetic sequences | [Monodepth2](https://github.com/nianticlabs/monodepth2), fine-tuned using 1,126 binocular image pairs recorded with the colon simulator |
| Hamlyn | Rectified images and stereo depths provided by [Endo-Depth-and-Motion](https://github.com/UZ-SLAMLab/Endo-Depth-and-Motion), reference [45] |
| EndoNeRF sequences | The pretrained depth estimation network from Endo-Depth-and-Motion [45], reused because of dataset similarity |

For the paper's preprocessing, obtain the relevant model weights/data from these upstream projects and follow their inference instructions. Export numerical depth maps to `pred_depth/`, with the image order, crop, resolution, and units matched to the RGB sequence. Reusing the original STTR-Light depths is an alternative input choice; it should not be confused with reproducing the paper's dataset-specific depth preparation.

### Obtaining Uncertainty Maps

UW-DNeRF uses **[GrUMoDepth](https://github.com/jhornauer/GrUMoDepth)** (*Gradient-Based Uncertainty for Monocular Depth Estimation*, ECCV 2022), referenced as [43] in the paper. Section III-C, Eq. (10), defines the uncertainty map as:

```text
U(I) = GrUMoDepth(I, D_theta)
```

Here, `I` is an RGB image and `D_theta` denotes the trained depth estimation model. Section IV-A states that uncertainty maps are generated following GrUMoDepth's official settings using its Monodepth2 models. Follow the upstream environment, model preparation, and uncertainty inference instructions, adapting the input sequence to the endoscopic images, then export the per-pixel numerical uncertainty arrays to `std/`.

The name `std` is this repository's input-directory convention: it does not imply that the values are calibrated statistical standard deviations. Uncertainty is estimated separately; the original EndoNeRF dataset documentation does not specify these maps as part of its standard inputs. Already-generated maps may be reused when their frame correspondence and preprocessing are known. STTR-Light depth or confidence outputs alone should not be assumed to be GrUMoDepth uncertainty.

### Preparing the Training Inputs

1. Place the left RGB frames in `images/` and their numerical depth priors in the configured `depth_dir`.
2. Generate or reuse the corresponding uncertainty arrays and place them in `uncertainty_dir` (default: `std/`). Preserve the same frame order and pixel alignment. Floating-point NPY files avoid quantizing small interval values; do not use colorized visualization images as scalar priors.
3. Set `depth_scale` to convert stored depth values into the desired units. With `uncertainty_normalization=raw`, the uncertainty values must already use those same units; `depth_scale` does not automatically scale raw uncertainty files.
4. With `uncertainty_normalization=minmax`, the loader maps the sequence's uncertainty values into `[uncertainty_min, uncertainty_max]`. The paper reports `[0.0001, 0.001]`, provided by `configs/uw_dnerf_paper_settings.txt`; the main configuration uses `[0.00001, 0.0001]`. Choose the interval range explicitly for the depth scale and experimental setup.
5. Training reads these prepared files; it does not run the external depth or uncertainty estimators. Missing uncertainty files cause an error rather than being replaced with depth maps.

The resulting pair `(D, U)` defines a per-pixel interval `[D-U, D+U]`. Depth supervision is enabled only when the rendered depth lies outside this interval; RGB supervision remains active.

## Training

```bash
python run_endonerf.py --config configs/uw_dnerf.txt --datadir "/path/to/data"
```

The main configuration is `configs/uw_dnerf.txt`. Set `datadir` and `expname` before running experiments. Checkpoints and outputs are saved under `logs/{expname}`.

### Standard Dataset Configurations

The three dataset groups have dedicated configurations for their five sequences:

| Dataset group | Sequence | Configuration | Default data directory | Tool masks |
|---|---|---|---|---|
| EndoNeRF | Pulling | `configs/endonerf/pulling.txt` | `../Medical-Datasets/pulling-pred_depth+std` | Enabled |
| EndoNeRF | Cutting | `configs/endonerf/cutting.txt` | `../Medical-Datasets/cutting_std` | Enabled |
| Hamlyn | Rectified20 | `configs/hamlyn/rectified20.txt` | `../Medical-Datasets/Hamlyn_20_1` | Disabled |
| Synthetic | Syn1 | `configs/synthetic/syn1.txt` | `../Medical-Datasets/moving_dataset` | Disabled |
| Synthetic | Syn2 | `configs/synthetic/syn2.txt` | `../Medical-Datasets/moving_dataset_2` | Disabled |

These paths describe the expected local data layout; datasets are not included. Run commands from the repository root and override `--datadir` if your directories differ. Each directory must contain the prepared `images/`, `pred_depth/`, `std/`, and `poses_bounds.npy`, plus `masks/` where enabled.

```bash
python run_endonerf.py --config configs/endonerf/pulling.txt
python run_endonerf.py --config configs/endonerf/cutting.txt
python run_endonerf.py --config configs/hamlyn/rectified20.txt
python run_endonerf.py --config configs/synthetic/syn1.txt
python run_endonerf.py --config configs/synthetic/syn2.txt
```

All five configurations enable local windows and uncertainty supervision, with 100,000 iterations, 2,048 rays per batch, 32 coarse plus 32 importance samples per ray, a keyframe interval of 7 training frames, learning rates of `0.0005/0.0005/0.0001`, and gated Huber depth loss (`delta=0.02`, `active_mean`). The uncertainty range is `[1e-5, 1e-4]`; held-out frames use interval 8 and offset 1. These are the standard training profiles, distinct from the paper-settings comparison profile below.

The camera-axis convention is preserved per sequence: `davinci_endoscopic=True` for Pulling, Cutting, and Syn2; `False` for Hamlyn and Syn1. This flag controls LLFF axis conversion, not the physical camera brand. Check it when supplying differently prepared poses. `depth_scale=1.0` assumes the stored depth values already have the desired units; change it explicitly if a conversion is required.

For held-out rendering, append `--render_only --render_test` to the same command. For example:

```bash
python run_endonerf.py --config configs/hamlyn/rectified20.txt --datadir "/path/to/Hamlyn_20_1" --render_only --render_test
```

The main method enables:

- `nerf_type=dnerfplus`: two-stage deformation and the canonical radiance field.
- `use_key_frame=True`: local windows and temporal interpolation.
- `use_std_as_depth_tolerance=True`: uncertainty-guided depth supervision.

`configs/uw_dnerf_paper_settings.txt` provides an alternative configuration using the paper's stated learning rates, iteration count, uncertainty range, test-frame offset, and an L1-sum depth objective. It is a parameter comparison profile, not a guarantee of reproduced paper results.

## Rendering and Evaluation

Render held-out frames:

```bash
python run_endonerf.py --config configs/uw_dnerf.txt --datadir "/path/to/data" --render_only --render_test
```

Outputs are saved under `logs/{expname}/renderonly_test_{step}/`:

```text
estim/              # Rendered RGB images and depth arrays
gt/                 # Corresponding reference RGB images
frame_indices.json  # Frame indices in the original sequence
```

Evaluate RGB reconstruction with PSNR, SSIM, and LPIPS:

```bash
python eval_rgb.py --gt_dir "/path/to/data/images" --mask_dir "/path/to/data/masks" --img_dir "/path/to/render/estim" --frame_indices "/path/to/render/frame_indices.json"
```

Omit `--mask_dir` when masks are unavailable. To also evaluate Abs Rel, Sq Rel, and RMSE against ground-truth depth, add `--gt_depth_dir "/path/to/data/depth"` and set `--depth_scale` to the appropriate unit conversion. Median depth alignment is optional and must be explicitly enabled with `--align_depth_median`.

## Reconstruction

Generate point clouds from a trained model:

```bash
python endo_pc_reconstruction.py --config_file configs/uw_dnerf.txt --n_frames 63 --depth_smoother --depth_smoother_d 28
```

Point clouds are saved to `logs/{expname}/reconstructed_pcds_{epoch}`. The reconstruction script uses default da Vinci camera intrinsics and a fixed viewpoint; adapt these settings for other camera configurations.

Visualize a reconstructed sequence:

```bash
python vis_pc.py --pc_dir logs/{expname}/reconstructed_pcds_{epoch}
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{shan2025uwdnerf,
  title   = {{UW-DNeRF}: Deformable Soft Tissue Reconstruction With Uncertainty-Guided Depth Supervision and Local Information Integration},
  author  = {Shan, Jiwei and Zhang, Zixin and Li, Hao and Hsieh, Cheng-Tai and Li, Yirui and Wu, Wenhua and Wang, Hesheng},
  journal = {IEEE Transactions on Medical Imaging},
  volume  = {44},
  number  = {7},
  pages   = {2808--2818},
  year    = {2025},
  doi     = {10.1109/TMI.2025.3550269}
}
```

## Acknowledgements

- Our implementation builds on [EndoNeRF](https://github.com/med-air/endonerf). We thank the authors for making their code publicly available.
- We also acknowledge [D-NeRF](https://github.com/albertpumarola/D-NeRF) and [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch).
- The RGB evaluation code is adapted from [peihaowang/nerf-pytorch](https://github.com/peihaowang/nerf-pytorch).
