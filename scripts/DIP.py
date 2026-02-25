# Deep Image Prior (DIP) Denoising with Distributional Consistency (DC) Loss
# -----------------------------------------------------------------------------
# This script reproduces the DIP experiments from the paper, comparing
# MSE loss against a Distributional Consistency (clipped Gaussian) loss across
# multiple noise levels and random seeds.
#
# Example usage:
#   python dip_reconstruction.py
#
# OR if you want to run the script with specific parameters:
#
#   python dip_reconstruction.py \
#       --image_name F16_GT \
#       --sigmas 10 25 50 75 100 \
#       --seeds 0 1 2 3 4 \
#       --epochs 10001 \
#       --lr 1e-3 \
# -----------------------------------------------------------------------------


# --- Set up import path ---
import sys, os

script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(script_dir, '..', 'src'))
sys.path.insert(0, src_path)

# --- External imports ---
import os
import argparse
import pickle
from typing import Dict

import numpy as np
import torch
import torch.optim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim

# --- Internal imports ---
from utils.losses import dc_loss_clipped_gaussian
from utils.common_utils import (get_image, crop_image, pil_to_np, get_noisy_image,
                                   get_noise, get_params, plot_image_grid, fix_seed)
from models import skip

# ----------------------------------------------------------------------------
# Configuration defaults for the DIP network architecture
# ----------------------------------------------------------------------------
SKIP_N33D = 128
SKIP_N33U = 128
SKIP_N11 = 4
NUM_SCALES = 5
UPSAMPLE_MODE = 'bilinear'
DOWNSAMPLE_MODE = 'stride'
ACT_FUN = 'LeakyReLU'
PAD_MODE = 'reflection'
INPUT_DEPTH = 32
N_CHANNELS = 3

# ----------------------------------------------------------------------------
# Device configuration
# ----------------------------------------------------------------------------
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# ----------------------------------------------------------------------------
# Network factory
# ----------------------------------------------------------------------------
def get_net_DIP() -> torch.nn.Module:
    """Constructs the U-Net-like DIP architecture."""
    return skip(
        INPUT_DEPTH,
        N_CHANNELS,
        num_channels_down=[SKIP_N33D] * NUM_SCALES,
        num_channels_up=[SKIP_N33U] * NUM_SCALES,
        num_channels_skip=[SKIP_N11] * NUM_SCALES,
        upsample_mode=UPSAMPLE_MODE,
        downsample_mode=DOWNSAMPLE_MODE,
        need_sigmoid=True,
        need_bias=True,
        pad=PAD_MODE,
        act_fun=ACT_FUN,
    )

# ----------------------------------------------------------------------------
# DIP optimization routine
# ----------------------------------------------------------------------------

def run_dip(
    loss_name: str,
    net_input: torch.Tensor,
    img_noisy_torch: torch.Tensor,
    img_np: np.ndarray,
    sigma: float,
    num_iter: int,
    seed: int = 0,
    lr: float = 1e-3,
    reg_noise_std: float = 1.0 / 30,
    use_scheduler: bool = False,
    show_every: int = 200,
    return_stats: bool = True,
    return_histos: bool = False,
    return_images: bool = False,
) -> Dict:
    """Optimize DIP for a single loss function and noise level.

    Args:
        loss_name: 'mse' or 'dist'.
        net_input: Initial random noise input for the network.
        img_noisy_torch: Noisy observation as tensor.
        img_np: Clean image in numpy format.
        sigma: Noise level in [0, 1] range.
        num_iter: Number of optimization iterations.
        seed: Random seed for reproducibility.
        lr: Learning rate.
        reg_noise_std: Std-dev of input noise regularization.
        use_scheduler: Whether to enable LR scheduler.
        show_every: Checkpoint frequency for PSNR backtracking.
        return_stats / histos / images: Toggles for logging.

    Returns:
        Dictionary containing reconstruction, loss curves, and optional extras.
    """
    fix_seed(seed)

    DEV = img_noisy_torch.device
    DTYPE = torch.cuda.FloatTensor if str(DEV).startswith('cuda') else torch.FloatTensor

    MSE_LOSS = torch.nn.MSELoss().type(DTYPE)

    def dc_loss(q: torch.Tensor, m: torch.Tensor, sigma: float, return_values: bool = False):
        """Distributional consistency loss wrapper (clipped Gaussian)."""
        return dc_loss_clipped_gaussian(
            q.flatten(), m.flatten(), sigma=sigma, return_values=return_values
        )

    net = get_net_DIP().type(DTYPE)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    optimizer = torch.optim.Adam(get_params('net', net, net_input), lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100)
        if use_scheduler
        else None
    )

    state = {
        'last_image': None,
        'images': [],
        'image_noisy': img_noisy_torch.cpu().numpy()[0].transpose(1, 2, 0),
        'last_net': None,
        'psnr_noisy_last': 0,
        'losses': {
            'psnr': [], 'ssim': [], 'loss': [], 'mse_loss': [], 'dist_loss': [], 'nrmse': []
        },
        'histos': [],
    }

    for i in range(num_iter):
        optimizer.zero_grad()

        inp = net_input_saved + noise.normal_() * reg_noise_std if reg_noise_std > 0 else net_input_saved
        out = net(inp)
        out_avg = out.detach()

        if loss_name == 'mse':
            total_loss = MSE_LOSS(out, img_noisy_torch)
        elif loss_name == 'dist':
            total_loss = dc_loss(out, img_noisy_torch, sigma=sigma)
        else:
            raise ValueError('loss_name must be "mse" or "dist"')

        total_loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step(total_loss.item())
            if scheduler.optimizer.param_groups[0]['lr'] <= 1e-6:
                print('\nLR reached 1e-6, stopping early.')
                break

        # Metrics & logging
        out_np = out.detach().cpu().numpy()[0]
        psnr_noisy = compare_psnr(img_noisy_torch.cpu().numpy()[0], out_np)
        psnr_gt = compare_psnr(img_np, out_np)
        state['last_image'] = out_np.transpose(1, 2, 0)

        print(
            f"Iter {i:05d} | Loss {total_loss.item():.4f} | PSNR(noisy) {psnr_noisy:.2f} | PSNR(gt) {psnr_gt:.2f}",
            end='\r',
        )

        # Backtracking checkpoint
        if i % show_every == 0:
            if psnr_noisy - state['psnr_noisy_last'] < -5:
                print('\nFalling back to previous checkpoint.')
                for new_p, net_p in zip(state['last_net'], net.parameters()):
                    net_p.data.copy_(new_p.to(DEV))
                continue
            state['last_net'] = [p.detach().cpu() for p in net.parameters()]
            state['psnr_noisy_last'] = psnr_noisy

        # Detailed stats
        if return_stats:
            with torch.no_grad():
                state['losses']['psnr'].append(psnr_gt)
                nrmse = np.sqrt(((img_np - out_np) ** 2).mean()) / np.sqrt((img_np ** 2).mean())
                state['losses']['nrmse'].append(nrmse)
                state['losses']['ssim'].append(ssim(img_np, out_np, channel_axis=0, data_range=1.0))
                state['losses']['loss'].append(total_loss.item())
                state['losses']['mse_loss'].append(MSE_LOSS(out, img_noisy_torch).item())
                dist_val = dc_loss(out, img_noisy_torch, sigma=sigma)
                state['losses']['dist_loss'].append(dist_val.item())

                if return_histos and i % 100 == 0:
                    dc_val, hist_vals, ix = dc_loss(out, img_noisy_torch, sigma=sigma, return_values=True)
                    state['histos'].append((hist_vals.cpu().numpy(), ix.cpu().numpy()))

        if return_images and i % 100 == 0:
            state['images'].append(out_np.transpose(1, 2, 0))

    # Convert lists -> arrays
    for key in state['losses']:
        state['losses'][key] = np.array(state['losses'][key])
    return state

def parse_args():
    """
    Parse command-line arguments for PET reconstruction.

    Returns:
        argparse.Namespace: Parsed argument values.
    """
    p = argparse.ArgumentParser(description='DIP denoising with DC vs MSE losses')
    p.add_argument('--image_name', type=str, default='F16_GT', help='Base name of input image in images/')
    p.add_argument('--sigmas', nargs='+', type=int, default=[75], help='Noise levels (0-100 is a sensible range)')
    p.add_argument('--seeds', nargs='+', type=int, default=[0], help='Random seeds')
    p.add_argument('--epochs', type=int, default=10001, help='Number of DIP iterations')
    p.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    p.add_argument('--use_scheduler', action='store_true', help='Use LR scheduler')
    p.add_argument('--output_dir', type=str, default='results/dip', help='Where to save pickled results')
    p.add_argument('--plot', action='store_true', help='Show side-by-side images during optimization')
    p.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., cuda:0, cpu)')
    return p.parse_args()

def main():
    """
    Main function to execute the DIP (Deep Image Prior) pipeline.
    This function performs denoising on an input image using the DIP approach
    with multiple configurations of seeds and noise levels (sigma). It supports
    two loss functions ('mse' and 'dist') and saves the results for each
    configuration to a specified output directory.
    Workflow:
        1. Parse command-line arguments.
        2. Create the output directory if it doesn't exist.
        3. Iterate over seeds and noise levels (sigma) to:
            a. Fix the random seed for reproducibility.
            b. Prepare the input image and add noise.
            c. Initialize the network input.
            d. Optionally plot the original and noisy images.
            e. Run the DIP pipeline for each loss function ('mse' and 'dist').
            f. Save the results to a pickle file in the output directory.
    Outputs:
        - Saves a pickle file for each configuration containing the results
          (e.g., loss statistics, histograms, and images) in the specified
          output directory.
    """
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    DEV = args.device
    DTYPE = torch.cuda.FloatTensor if DEV.startswith('cuda') else torch.FloatTensor

    if DEV.startswith('cuda'):
        print("Cuda Available: ", torch.cuda.is_available())

    for seed in args.seeds:
        for sigma in args.sigmas:
            print(f"\n=== Seed {seed} | Sigma {sigma} ===")
            fix_seed(seed)

            sigma_f = sigma / 255.0
            reg_noise_std = 1.0 / 30.0 if sigma <= 25 else 1.0 / 20.0

            img_pil = crop_image(get_image(f'data/images/{args.image_name}.png')[0], d=32)
            img_np = pil_to_np(img_pil)
            img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_f)
            img_noisy_torch = torch.tensor(img_noisy_np).unsqueeze(0).type(DTYPE)

            net_input = get_noise(INPUT_DEPTH, 'noise', img_pil.size[::-1]).type(DTYPE)

            if args.plot:
                plot_image_grid([img_np, img_noisy_np], 4, 6)

            states = {}
            for loss_name in ['mse', 'dist']:
                print(f"  > Running {loss_name.upper()} loss")
                state = run_dip(
                    loss_name=loss_name,
                    net_input=net_input.clone(),
                    img_noisy_torch=img_noisy_torch,
                    img_np=img_np,
                    sigma=sigma_f,
                    num_iter=args.epochs,
                    seed=seed,
                    lr=args.lr,
                    reg_noise_std=reg_noise_std,
                    use_scheduler=args.use_scheduler,
                    return_stats=True,
                    return_histos=True,
                    return_images=True,
                )
                states[loss_name] = state
                print(f"  > Finished {loss_name}")

            out_path = os.path.join(
                args.output_dir,
                f"image={args.image_name}_sigma={sigma}_scheduled={args.use_scheduler}_seed={seed}.pkl"
            )
            with open(out_path, 'wb') as f:
                pickle.dump(states, f)
            print(f"Saved results to {out_path}")

if __name__ == '__main__':
    main()
