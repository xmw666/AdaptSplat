import torch
from torch import Tensor
from jaxtyping import Float
from einops import reduce, rearrange
from skimage.metrics import structural_similarity
import functools
import os
from PIL import Image
import imageio
import numpy as np
from easydict import EasyDict as edict
import json
from rich import print
import torchvision
from plyfile import PlyData, PlyElement

import warnings
# Suppress warnings for LPIPS loss loading
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum.*")

@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, "batch"]:
    """
    Compute Peak Signal-to-Noise Ratio between ground truth and predicted images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
        predicted: Images with shape [batch, channel, height, width], values in [0, 1]
        
    Returns:
        PSNR values for each image in the batch
    """
    ground_truth = torch.clamp(ground_truth, 0, 1)
    predicted = torch.clamp(predicted, 0, 1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * torch.log10(mse) 



@functools.lru_cache(maxsize=None)
def get_lpips_model(net_type="vgg", device="cuda"):
    from lpips import LPIPS
    return LPIPS(net=net_type).to(device)

@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    normalize: bool = True,
) -> Float[Tensor, "batch"]:

    """
    Compute Learned Perceptual Image Patch Similarity between images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width]
        predicted: Images with shape [batch, channel, height, width]
        The value range is [0, 1] when we have set the normalize flag to True.
        It will be [-1, 1] when the normalize flag is set to False.
    Returns:
        LPIPS values for each image in the batch (lower is better)
    """

    _lpips_fn = get_lpips_model(device=predicted.device)
    batch_size = 10  # Process in batches to save memory
    values = [
        _lpips_fn(
            ground_truth[i : i + batch_size],
            predicted[i : i + batch_size],
            normalize=normalize,
        )
        for i in range(0, ground_truth.shape[0], batch_size)
    ]
    return torch.cat(values, dim=0).squeeze()



@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    """
    Compute Structural Similarity Index between images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
        predicted: Images with shape [batch, channel, height, width], values in [0, 1]
        
    Returns:
        SSIM values for each image in the batch (higher is better)
    """
    ssim_values= []
    
    for gt, pred in zip(ground_truth, predicted):
        # Move to CPU and convert to numpy
        gt_np = gt.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        
        # Calculate SSIM
        ssim = structural_similarity(
            gt_np,
            pred_np,
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        ssim_values.append(ssim)
    
    # Convert back to tensor on the same device as input
    return torch.tensor(ssim_values, dtype=predicted.dtype, device=predicted.device)



@torch.no_grad()
def export_results(
    result: edict,
    out_dir: str, 
    compute_metrics: bool = False,
    uid: int = 0
):
    """
    Save results including images and optional metrics and videos.
    
    Args:
        result: EasyDict containing input, target, and rendered images, and optionally video frames
        out_dir: Directory to save the evaluation results
        compute_metrics: Whether to compute and save metrics
    """
    os.makedirs(out_dir, exist_ok=True)

    target_data = result.target
    rendered_image = result.render
    input_data = result.input
    b, v, _, h, w = rendered_image.size()

    for batch_idx in range(input_data["image"].size(0)):
        scene_name = input_data["scene_name"][0]
        sample_dir = os.path.join(out_dir, f"{uid:06d}_{scene_name}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Get target view indices
        target_indices = target_data["index"][batch_idx, :].cpu().numpy().squeeze(-1).astype(int)
        input_indices = input_data["index"][batch_idx, :].cpu().numpy().squeeze(-1).astype(int)
        target_indices_path = os.path.join(sample_dir, "target_indices.txt")
        input_indices_path = os.path.join(sample_dir, "input_indices.txt")
        np.savetxt(target_indices_path, target_indices, fmt="%d")
        np.savetxt(input_indices_path, input_indices, fmt="%d")
        os.makedirs(os.path.join(sample_dir, "target"), exist_ok=True)
        os.makedirs(os.path.join(sample_dir, "rendering"), exist_ok=True)
        for i in range(v):
            target_path = os.path.join(sample_dir, "target", f"{i}.png")
            rendering_path = os.path.join(sample_dir, "rendering", f"{i}.png")
            torchvision.utils.save_image(
                target_data["image"][batch_idx, i], target_path
            )
            torchvision.utils.save_image(
                rendered_image[batch_idx, i], rendering_path
            )
        
        # Compute and save metrics if requested
        if compute_metrics:
            _save_metrics(
                target_data["image"][batch_idx],
                rendered_image[batch_idx],
                target_indices,
                sample_dir,
                scene_name
            )
        if "gaussians" in result:
            export_gaussian_ply(result["gaussians"], sample_dir)



def export_gaussian_ply(gaussian_dict, save_path):
    """
    Export full 3DGS-compatible Gaussian PLY.

    Attribute layout follows the standard 3DGS viewer convention:
        x y z  nx ny nz
        f_dc_0 f_dc_1 f_dc_2
        f_rest_0 ... f_rest_{3*((sh_degree+1)^2-1)-1}   [R-rest, G-rest, B-rest order]
        opacity                                            [logit, i.e. value before sigmoid]
        scale_0 scale_1 scale_2                           [log scale, stored as-is]
        rot_0 rot_1 rot_2 rot_3                           [quaternion w x y z]

    Opacity is represented as (opacity_degree+1)^2 SH coefficients in the model.
    We reduce to 0th-order (view-independent) by evaluating only the DC term:
        opacity_logit = SH_C0 * dc_coeff
    where SH_C0 = 1/(2*sqrt(pi)) ≈ 0.28209479177.
    This logit is stored directly (no sigmoid), matching the 3DGS PLY convention.

    Args:
        gaussian_dict: dict with keys
            - xyz:        (B, N, 3)
            - feature:    (B, N, (sh_degree+1)^2, 3)  rearranged SH color coefficients
            - scale:      (B, N, 3)                    log-scale (scale_bias already applied)
            - rotation:   (B, N, 4)                    quaternion (w x y z)
            - opacity_sh: (B, N, (opacity_degree+1)^2, 1)  SH opacity coefficients
        save_path: directory; file saved as gaussians.ply
    """
    SH_C0 = 0.28209479177  # 1 / (2 * sqrt(pi))

    out_file = os.path.join(save_path, "gaussians.ply")

    # Extract batch 0, CPU float32
    xyz      = gaussian_dict["xyz"][0].detach().cpu().float().numpy()        # (N, 3)
    feature  = gaussian_dict["feature"][0].detach().cpu().float().numpy()    # (N, D_color, 3)
    scale    = gaussian_dict["scale"][0].detach().cpu().float().numpy()      # (N, 3)
    rotation = gaussian_dict["rotation"][0].detach().cpu().float().numpy()   # (N, 4)
    opacity  = gaussian_dict["opacity_sh"][0].detach().cpu().float().numpy() # (N, D_opacity, 1)

    N = xyz.shape[0]

    # --- Opacity: 2nd-order SH → 0th-order logit ---
    # DC coeff → multiply by SH_C0 to get the evaluated degree-0 SH value,
    # which is used as the logit by the renderer (sigmoid applied during rendering).
    opacity_logit = (SH_C0 * opacity[:, 0, 0]).astype(np.float32)  # (N,)

    # --- Filter: remove bottom 25% by opacity (after sigmoid) ---
    opacity_prob = 1.0 / (1.0 + np.exp(-opacity_logit))  # sigmoid, (N,)
    threshold = np.percentile(opacity_prob, 25)
    mask = opacity_prob >= threshold
    xyz, feature, scale, rotation, opacity_logit = (
        xyz[mask], feature[mask], scale[mask], rotation[mask], opacity_logit[mask])
    N = xyz.shape[0]
    print(f"PLY filter: kept top 75% opacity (threshold={threshold:.4f}), {N} splats remaining")

    # --- Color SH: split DC and rest ---
    # feature shape: (N, num_coeffs, 3), dim1=coeff_idx, dim2=RGB
    f_dc   = feature[:, 0, :]       # (N, 3)  — DC per channel
    # Rest: (N, num_rest, 3) → reorder to [R-rest, G-rest, B-rest] → (N, 3*num_rest)
    f_rest_raw = feature[:, 1:, :]  # (N, num_rest, 3)
    f_rest = f_rest_raw.transpose(0, 2, 1).reshape(N, -1)  # (N, 3*num_rest)

    # --- Build PLY dtype ---
    num_rest = f_rest.shape[1]
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]
    dtype += [(f'f_rest_{i}', 'f4') for i in range(num_rest)]
    dtype += [
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]

    attrs = np.zeros(N, dtype=dtype)

    # Position
    attrs['x'], attrs['y'], attrs['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    # Normals (unused in 3DGS but required by viewers)
    attrs['nx'][:] = 0.0
    attrs['ny'][:] = 0.0
    attrs['nz'][:] = 0.0
    # Color DC
    attrs['f_dc_0'], attrs['f_dc_1'], attrs['f_dc_2'] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    # Color rest
    for i in range(num_rest):
        attrs[f'f_rest_{i}'] = f_rest[:, i]
    # Opacity logit (0th-order SH, before sigmoid)
    attrs['opacity'] = opacity_logit
    # Scale (log scale)
    attrs['scale_0'], attrs['scale_1'], attrs['scale_2'] = scale[:, 0], scale[:, 1], scale[:, 2]
    # Rotation quaternion (w x y z)
    attrs['rot_0'], attrs['rot_1'], attrs['rot_2'], attrs['rot_3'] = (
        rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3])

    ply_data = PlyData([PlyElement.describe(attrs, 'vertex')])
    ply_data.write(out_file)
    print(f"Saved Gaussian PLY: {N} splats → {out_file}")


def _save_metrics(target, prediction, view_indices, out_dir, scene_name):
    target = target.to(torch.float32)
    prediction = prediction.to(torch.float32)
    
    psnr_values = compute_psnr(target, prediction)
    lpips_values = compute_lpips(target, prediction)
    ssim_values = compute_ssim(target, prediction)

    metrics = {
        "summary": {
            "scene_name": scene_name,
            "psnr": float(psnr_values.mean()),
            "lpips": float(lpips_values.mean()),
            "ssim": float(ssim_values.mean())
        },
        "per_view": []
    }
    
    for i, view_idx in enumerate(view_indices):
        metrics["per_view"].append({
            "view": int(view_idx), "psnr": float(psnr_values[i]), "lpips": float(lpips_values[i]), "ssim": float(ssim_values[i])
        })
    
    # Save metrics to a single JSON file
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

def create_video_from_frames(frames, output_video_file, framerate=30):
    """
    Creates a video from a sequence of frames.

    Parameters:
        frames (numpy.ndarray): Array of image frames (shape: N x H x W x C).
        output_video_file (str): Path to save the output video file.
        framerate (int, optional): Frames per second for the video. Default is 30.
    """
    frames = np.asarray(frames)

    # Normalize frames if values are in [0,1] range
    if frames.max() <= 1:
        frames = (frames * 255).astype(np.uint8)

    imageio.mimsave(output_video_file, frames, fps=framerate, quality=8)

def _save_video(frames, out_dir):
    """
    Save video from rendered frames.
    Input frames should be in [v, c, h, w] format.
    """
    frames = np.ascontiguousarray(np.array(frames.to(torch.float32)))
    frames = rearrange(frames, "v c h w -> v h w c")
    create_video_from_frames(
        frames, 
        f"{out_dir}/rendered_video.mp4", 
        framerate=30
    )


def summarize_evaluation(evaluation_folder):
    # Find and sort all valid subfolders
    subfolders = sorted(
        [
            os.path.join(evaluation_folder, dirname)
            for dirname in os.listdir(evaluation_folder)
            if os.path.isdir(os.path.join(evaluation_folder, dirname))
        ],
        key=lambda x: int(os.path.basename(x)) if os.path.basename(x).isdigit() else os.path.basename(x)
    )

    metrics = {}
    valid_subfolders = []
    
    for subfolder in subfolders:
        json_path = os.path.join(subfolder, "metrics.json")
        if not os.path.exists(json_path):
            print(f"!!! Metrics file not found in {subfolder}, skipping...")
            continue
            
        valid_subfolders.append(subfolder)
        
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                # Extract summary metrics
                for metric_name, metric_value in data["summary"].items():
                    if metric_name == "scene_name":
                        continue
                    metrics.setdefault(metric_name, []).append(metric_value)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading metrics from {json_path}: {e}")

    if not valid_subfolders:
        print(f"No valid metrics files found in {evaluation_folder}")
        return

    csv_file = os.path.join(evaluation_folder, "summary.csv")
    with open(csv_file, "w") as f:
        header = ["Index"] + list(metrics.keys())
        f.write(",".join(header) + "\n")
        
        for i, subfolder in enumerate(valid_subfolders):
            basename = os.path.basename(subfolder)
            values = [str(metric_values[i]) for metric_values in metrics.values()]
            f.write(f"{basename},{','.join(values)}\n")
        
        f.write("\n")
        
        averages = [str(sum(values) / len(values)) for values in metrics.values()]
        f.write(f"average,{','.join(averages)}\n")
    
    print(f"Summary written to {csv_file}")
    print(f"Average: {','.join(averages)}")

    # export average metrics to a text file
    with open(os.path.join(evaluation_folder, "average_metrics.txt"), "w") as f:
        f.write(f"Average: {','.join(averages)}\n")
