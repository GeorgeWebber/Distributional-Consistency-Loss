# 1D Signal Deconvolution with Distributional Consistency (DC) Loss
# -----------------------------------------------------------------------------
# This script performs deconvolution on a synthetic 1D signal, comparing
# Mean Squared Error (MSE) loss against a Distributional Consistency (DC)
# loss with a Gaussian assumption. The script generates a blurred and noisy
# signal, then reconstructs the original signal using both loss functions.
#
# Example usage:
#   python deconv.py
#
# OR if you want to specify parameters:
#
#   python deconv.py \
#       --num_steps 20000 \
#       --lr 0.005 \
#       --output_dir results/deconv \
#       --device cuda \
#       --blur 1.0 \
#       --noise_sigma 0.1 \
#       --num_points 500 \
#       --seed 0 \
#       --data_seed 0
# -----------------------------------------------------------------------------


# --- Set up import path ---
import sys, os

script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(script_dir, '..', 'src'))
sys.path.insert(0, src_path)

# --- External imports ---
import numpy as np
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.cuda import empty_cache
import argparse

# --- Internal imports ---
from utils.common_utils import fix_seed
from utils.losses import dc_loss_gaussian

# --- Default plotting config ---
plt.rcParams['image.interpolation'] = 'none'

def run_deconvolution(loss_name, y_true, y_noisy, sigma, kernel, kernel_size, N, num_steps=5000, seed=0, lr=0.01, dev='cuda',
                      return_stats=True, return_histos=False):
    """
    Perform deconvolution using a specified loss function.
    Args:
        loss_name (str): The name of the loss function to use. Options are 'mse' (Mean Squared Error) or 'dist' 
                            (distribution loss with Gaussian assumption).
        y_true (torch.Tensor): The ground truth signal.
        y_noisy (torch.Tensor): The noisy observation of the signal.
        sigma (float): The standard deviation of the Gaussian noise for the distribution loss.
        kernel (torch.Tensor): The convolution kernel to use for blurring.
        kernel_size (int): The size of the convolution kernel.
        N (int): The size of the estimated signal.
        num_steps (int, optional): The number of optimization steps to perform. Default is 5000.
        seed (int, optional): The random seed for reproducibility. Default is 0.
        lr (float, optional): The learning rate for the Adam optimizer. Default is 0.01.
        dev (str, optional): The device to use for computation ('cuda' or 'cpu'). Default is 'cuda'.
        return_stats (bool, optional): Whether to return statistics such as loss values. Default is True.
        return_histos (bool, optional): Whether to return histograms of the distribution loss every 100 steps. 
                                        Only used if `return_stats` is True. Default is False.
    Returns:
        dict: A dictionary containing the following keys:
            - "x_est" (numpy.ndarray): The estimated signal after optimization (if `return_stats` is True).
            - "y_true" (numpy.ndarray): The ground truth signal.
            - "y_noisy" (numpy.ndarray): The noisy observation of the signal.
            - "y_blur" (numpy.ndarray): The blurred version of the ground truth signal.
            - "losses" (dict): A dictionary containing the training loss, MSE loss, and distribution loss over steps.
            - "histos" (list): A list of histograms and their corresponding indices (if `return_histos` is True).
    Raises:
        ValueError: If `loss_name` is not 'mse' or 'dist'.
    Notes:
        - The function uses the Adam optimizer to minimize the specified loss function.
        - The `dc_loss_gaussian` function computes the DC loss assuming Gaussian noise.
        - The `empty_cache` function is called at each step to free up GPU memory.
    """

    fix_seed(seed)

    mse_loss_fn = torch.nn.MSELoss()

    x_est = torch.zeros(N, requires_grad=True, device=dev)
    optimizer = optim.Adam([x_est], lr=lr)

    if loss_name == "mse":
        loss_fn = mse_loss_fn
    elif loss_name == "dist":
        loss_fn = lambda q, m: dc_loss_gaussian(q.flatten(), m.flatten(), sigma, return_values=False)
    else:
        raise ValueError(f"Unknown loss_name '{loss_name}' — use 'mse' or 'dist'")

    y_true_reshaped = y_true.view(1, 1, -1)
    y_blur = F.conv1d(F.pad(y_true_reshaped, (kernel_size//2,)*2, mode='reflect'), kernel)

    state = {
        "x_est": None,
        "y_true": y_true.cpu().numpy(),
        "y_noisy": y_noisy.cpu().numpy(),
        "y_blur": y_blur.cpu().numpy(),
        "losses": {"train_loss": [], "mse_loss": [], "dist_loss": []},
        "histos": []
    }

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        y_pred = F.conv1d(F.pad(x_est.view(1, 1, -1), (kernel_size // 2,) * 2, mode='reflect'), kernel)
        loss = loss_fn(y_pred, y_noisy)
        loss.backward()
        optimizer.step()

        if return_stats:
            with torch.no_grad():
                mse_value = mse_loss_fn(y_pred, y_noisy)
                dist_value = dc_loss_gaussian(y_pred.flatten(), y_noisy.flatten(), sigma, return_values=False)

                state["losses"]["train_loss"].append(loss.item())
                state["losses"]["mse_loss"].append(mse_value.item())
                state["losses"]["dist_loss"].append(dist_value.item())

                if return_histos and step % 100 == 0:
                    dist_val, hist_vals, ix_vals = dc_loss_gaussian(
                        y_pred.flatten(), y_noisy.flatten(), sigma, return_values=True
                    )
                    state["histos"].append((
                        hist_vals.detach().cpu().numpy(),
                        ix_vals.detach().cpu().numpy()
                    ))

        if step % 500 == 0:
            print(f"Step {step:4d} | {loss_name.upper()} train loss: {loss.item():.4f} | MSE: {mse_value.item():.4f} | DIST: {dist_value.item():.4f}")

        empty_cache()

    if return_stats:
        state["x_est"] = x_est.detach().cpu().numpy()
        for key in state["losses"]:
            state["losses"][key] = np.array(state["losses"][key])

    return state



def blur_signal(signal_np, kernel, kernel_size, dev='cuda'):
    """
    Apply convolutional blur to a 1D signal using reflection padding.
    """
    signal = torch.tensor(signal_np, device=dev).view(1, 1, -1)
    blurred = F.conv1d(F.pad(signal, (kernel_size // 2,) * 2, mode='reflect'), kernel)
    return blurred.view(-1).cpu().numpy()

def parse_args():
    """
    Parse command-line arguments for PET reconstruction.

    Returns:
        argparse.Namespace: Parsed argument values.
    """
    parser = argparse.ArgumentParser(description="Run 1D deconvolution experiment with DC and MSE loss")
    parser.add_argument('--num_steps', type=int, default=20000, help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='results/deconv', help='Output directory for saving results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--blur', type=float, default=1.0, help='Blur level (sigma) for the Gaussian kernel')
    parser.add_argument('--noise_sigma', type=float, default=0.1, help='Standard deviation of noise added to the blurred signal')
    parser.add_argument('--num_points', type=int, default=500, help='Number of points in the signal')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed for data generation')
    return parser.parse_args()

def main():
    """
    Main function to perform deconvolution on a noisy signal.
    This function generates a synthetic signal, applies a blur kernel, adds noise, 
    and then performs deconvolution using two different methods: mean squared error (MSE) 
    and a distance-based loss. The results are saved to a specified output directory.
    Args:
        None. The function relies on command-line arguments parsed by `parse_args()`.
    Workflow:
        1. Parse command-line arguments.
        2. Fix the random seed for reproducibility.
        3. Generate a synthetic signal composed of sinusoidal components.
        4. Apply a blur kernel to the signal.
        5. Add Gaussian noise to the blurred signal.
        6. Perform deconvolution using two methods:
            - MSE-based optimization.
            - Distance-based optimization.
        7. Save the results to a pickle file in the specified output directory.
    Outputs:
        - A pickle file containing the results of the deconvolution for both methods.
        - The file is saved with the naming convention:
          `deconvolution_N={num_points}_blur={blur}_noise={noise_sigma}_seed={seed}_dataseed={data_seed}_results.pkl`.
    """
    args = parse_args()

    fix_seed(args.data_seed)

    N = args.num_points

    dev = args.device

    x = torch.linspace(0, 1, N, device=dev)
    y_true = (torch.sin(2 * np.pi * 5 * x) + 0.5 * torch.sin(2 * np.pi * 20 * x)).to(dev)

    N_ref = 500
    sigma_index_ref = args.blur
    kernel_size = 31

    dx = 1.0 / (N - 1)
    dx_ref = 1.0 / (N_ref - 1)
    sigma_domain = sigma_index_ref * dx_ref
    blur_sigma = sigma_domain / dx

    kx = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size, device=dev)
    kernel = torch.exp(-0.5 * (kx / blur_sigma)**2)
    kernel /= kernel.sum()
    kernel = kernel.view(1, 1, -1)

    y_true_reshaped = y_true.view(1, 1, -1)
    y_blur = F.conv1d(F.pad(y_true_reshaped, (kernel_size//2,)*2, mode='reflect'), kernel)
    noise_sigma = args.noise_sigma
    y_noisy = y_blur + noise_sigma * torch.randn_like(y_blur)

    state_mse = run_deconvolution("mse", y_true, y_noisy, noise_sigma, kernel, kernel_size, N, dev=dev,
                                  num_steps=args.num_steps, lr=args.lr, return_histos=True)
    state_dc = run_deconvolution("dist", y_true, y_noisy, noise_sigma, kernel, kernel_size, N, dev=dev,
                                 num_steps=args.num_steps, lr=args.lr, return_histos=True)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"deconvolution_N={N}_blur={args.blur}_noise={args.noise_sigma}_seed={args.seed}_dataseed={args.data_seed}_results.pkl"), 'wb') as f:
        pickle.dump({"mse": state_mse, "dist": state_dc}, f)

    print(f"Saved results to {args.output_dir}")

if __name__ == '__main__':
    main()
