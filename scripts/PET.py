# PET Image Reconstruction with Distributional Consistency (DC) Loss
# -----------------------------------------------------------------------------
# This script performs Positron Emission Tomography (PET) image reconstruction
# using three different loss functions: Negative Log-Likelihood (NLL), 
# Distributional Consistency (DC) loss, and Maximum Likelihood Expectation 
# Maximization (MLEM). The script generates synthetic PET data and reconstructs
# the original image using these loss functions.
#
# Example usage:
#   python PET.py
#
# OR if you want to specify parameters:
#
#   python PET.py \
#       --img_size 256 \
#       --epochs 10001 \
#       --lr 0.005 \
#       --output_dir results/PET \
#       --device cuda \
#       --dose 5.0 \
#       --bin_mash 2 \
#       --seed 0 \
#       --poisson_seed 0
# -----------------------------------------------------------------------------

# --- Set up import path ---
import sys, os

script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(script_dir, '..', 'src'))
sys.path.insert(0, src_path)

# --- External imports ---
import argparse
from matplotlib import pyplot as plt
import numpy as np
import torch
import pickle
from scipy.ndimage import binary_fill_holes
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.cuda import empty_cache

# --- Internal imports ---
from utils.metrics import SSIM
from utils.phantom import get_brainweb_phantom
from utils.projector import get_projector, get_binmashed_projector_adjoint_function, get_binmashed_projector_forward_function
from utils.common_utils import fix_seed
from utils.losses import dc_loss_poisson

# --- Default plotting config ---
plt.rcParams['image.interpolation'] = 'none'


# --- Helper functions ---
def negative_log_likelihood(x, m, sino_mask, forward):
    """
    Computes the negative Poisson log-likelihood (NLL) for a given input.    
    Args:
        x (torch.Tensor): The input tensor representing the current estimate of the 
            reconstructed image or signal.
        m (torch.Tensor): The measured data (e.g., sinogram) tensor.
        sino_mask (torch.Tensor): A boolean mask tensor indicating the valid elements 
            in the sinogram or measured data.
        forward (Callable): A forward operator function that maps the input `x` to 
            the sinogram space.
    Returns:
        torch.Tensor: The computed negative log-likelihood value as a scalar tensor.
    """
    q = forward(x).flatten()[sino_mask]
    m = m.flatten()[sino_mask]
    q_plus = torch.nn.functional.relu(q) + 1e-16
    nll = - (m * torch.log(q_plus) - q_plus - torch.lgamma(m + 1)).sum()
    return nll

def dist_loss(x, m, sino_mask, forward, return_values=False):
    """
    Wrapper function for computing the DC loss for a given input image tensor `x` 
    and measurement `m` using a forward operator and a specified mask.
    Args:
        x (torch.Tensor): The input image tensor representing the current image estimate.
        m (torch.Tensor): The measurement tensor to compare against.
        sino_mask (torch.Tensor): A boolean mask to select relevant elements 
            from the flattened tensors.
        forward (callable): A forward operator function that takes `x` as input 
            and produces a tensor of the same shape as `m`.
        return_values (bool, optional): If True, additional values related to 
            the loss computation are returned. Defaults to False.
    Returns:
        torch.Tensor or tuple: The computed DC loss. If `return_values` 
        is True, a tuple containing the loss and additional values is returned.
    """

    q = forward(x).flatten()[sino_mask]
    m = m.flatten()[sino_mask]
    q_plus = torch.nn.functional.relu(q) + 1e-16
    return dc_loss_poisson(q_plus, m, return_values=return_values)

def do_PET_reconstruction(loss_name, img_torch, m, num_iter, forward, backward, mumap, sensitivity, add_prior=None, beta=None, lr=1e-3, seed=0, return_stats=False, return_histos=False, return_images=False, use_scheduler=False):
    """
    Perform PET image reconstruction using one of several loss functions.

    Args:
        loss_name (str): Loss function to use ('mlem', 'nll', 'dist').
        img_torch (torch.Tensor): Ground truth image tensor.
        m (torch.Tensor): Measured sinogram data.
        num_iter (int): Number of optimization iterations.
        forward (callable): Forward projection operator.
        backward (callable): Backward projection operator (adjoint).
        mumap (torch.Tensor): Mu-map defining object support.
        sensitivity (torch.Tensor): Sensitivity image.
        lr (float): Learning rate.
        seed (int): Random seed.
        return_stats (bool): Whether to compute metrics during optimization.
        return_histos (bool): Whether to store histogram data (for dist loss).
        return_images (bool): Whether to store intermediate images.
        use_scheduler (bool): Whether to use a learning rate scheduler.

    Returns:
        dict: Dictionary containing reconstructed image and optional logs/metrics.
    """

    sino_mask = forward(mumap).flatten() > 1e-2
    mask = mumap.to(torch.float64)

    sino_mask_torch = sino_mask
    img_np = img_torch.detach().cpu().numpy().squeeze()

    fix_seed(seed)

    state = {
        "last_image": None,
        "images": [],
        "losses": {
            "psnr": [], "ssim": [], "loss": [], "nll_loss": [], "dist_loss": [], "nrmse": []
        },
        "histos": [],
        "add_prior": add_prior,
        "beta": beta
    }

    x = torch.ones_like(img_torch).to(torch.float64).to(img_torch.device) * img_torch[img_torch > 0].mean()
    x = x * mask
    x.requires_grad = True

    if loss_name == "mlem":
        x.requires_grad = False
    else:
        def preconditioned_grad_hook(grad):
            return (grad * mask) / (sensitivity + 1e-6)
        x.register_hook(preconditioned_grad_hook)

    min_lr = lr / 100
    optimizer = torch.optim.Adam([x], lr=lr)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=False)

    #torch.autograd.set_detect_anomaly(True)

    for i in range(num_iter):
        empty_cache()
        optimizer.zero_grad()

        if loss_name in ["nll", "dist"]:
            new_x = x * mask
            if loss_name == "nll":
                loss = negative_log_likelihood(new_x, m, sino_mask_torch, forward)
            elif loss_name == "dist":
                loss = dist_loss(new_x, m, sino_mask_torch, forward)

            if add_prior:
                if add_prior == "tv":
                    from utils.losses import total_variation
                    loss = loss + beta * total_variation(new_x)
                elif add_prior == "perfect":
                    loss = loss + beta * torch.sum((new_x - img_torch) ** 2)
            loss.backward()
            optimizer.step()
            if use_scheduler:
                scheduler.step(loss.detach().item())
                if optimizer.param_groups[0]['lr'] <= min_lr:
                    print("Stopping early: Learning rate has reached the minimum threshold.")
                    break
        elif loss_name == "mlem":
            new_x = (x / sensitivity) * (backward(m / (forward(x) + 1e-16))) * mask
            x = new_x
            loss = negative_log_likelihood(new_x, m, sino_mask_torch, forward).detach()

        out_np = torch.nn.functional.relu(new_x).detach().cpu().numpy().squeeze()
        psnr_gt = compare_psnr(img_np, out_np, data_range=img_np.max() - img_np.min())

        state["last_image"] = out_np
        print(f'Iteration {i:05d} Loss {loss.item():.6f} PSNR_gt: {psnr_gt:.2f}', '\r', end='')

        if return_stats:
            with torch.no_grad():
                state["losses"]["psnr"].append(psnr_gt)
                nrmse = np.sqrt(np.mean((img_np - out_np) ** 2)) / np.sqrt(np.mean(img_np ** 2))
                state["losses"]["nrmse"].append(nrmse)
                state["losses"]["ssim"].append(SSIM(img_np, out_np))
                state["losses"]["loss"].append(loss.item())
                if loss_name == "nll":
                    state["losses"]["nll_loss"].append(loss.item())
                else:
                    state["losses"]["nll_loss"].append(negative_log_likelihood(new_x, m, sino_mask_torch, forward).item())
                if i % 100 == 0 and return_histos:
                    dist_loss_value, histo_values, sort_ix = dist_loss(new_x, m, sino_mask, forward, return_values=True)
                    state["histos"].append((histo_values.detach().cpu().numpy(), sort_ix.detach().cpu().numpy()))
                    state["losses"]["dist_loss"].append(dist_loss_value.item())
                else:
                    if loss_name == "dist":
                        state["losses"]["dist_loss"].append(loss.item())
                    else:
                        state["losses"]["dist_loss"].append(dist_loss(new_x, m, sino_mask, forward, return_values=False).item())
        if return_images and i % 10 == 0:
            state["images"].append(out_np)
        del new_x, loss
        empty_cache()

    if return_stats:
        for key in state["losses"]:
            state["losses"][key] = np.array(state["losses"][key])

    return state


def parse_args():
    """
    Parse command-line arguments for PET reconstruction.

    Returns:
        argparse.Namespace: Parsed argument values.
    """
    parser = argparse.ArgumentParser(description="Run PET reconstruction with different loss methods.")
    parser.add_argument('--img_size', type=int, default=256, help='Image size (e.g. 128, 256, 512)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--poisson_seed', type=int, default=0, help='Random seed for sinogram reproducibility')
    parser.add_argument('--bin_mash', type=int, default=2, help='Bin mash factor')
    parser.add_argument('--dose', type=float, default=5.0, help='Dose value to scale the phantom')
    parser.add_argument('--phantom_name', type=str, default='brainweb', choices=['brainweb'])
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--ring_factor', type=float, default=4.0)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--use_scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--epochs', type=int, default=10001)
    parser.add_argument('--output_dir', type=str, default='results/PET', help='Directory to save results')
    parser.add_argument('--save_tag', type=str, default='', help='Optional tag to distinguish saved result directories')
    return parser.parse_args()

def main():
    """
    Main execution function. Sets up the PET system, generates synthetic data,
    performs reconstruction using all loss methods, and saves results.
    """
    args = parse_args()
    
    fix_seed(args.poisson_seed)

    projector = get_projector(args.device, args.img_size, ring_factor=args.ring_factor)
    A = get_binmashed_projector_forward_function(projector, args.bin_mash)
    AT = get_binmashed_projector_adjoint_function(projector, args.bin_mash)

    dose_scaled = args.dose * (args.img_size / 128)**2

    phantom = torch.tensor(get_brainweb_phantom(n=args.img_size).copy()).to(torch.float64)

    nonzero_mask = (phantom.numpy() != 0).astype(np.uint8)
    mumap = binary_fill_holes(nonzero_mask).astype(np.uint8)[None, None, ..., None]
    mumap = torch.tensor(mumap).to(args.device)

    phantom = phantom * dose_scaled
    phantom = phantom[None, None, ..., None].to(args.device).to(torch.float64)

    print(phantom.mean(), phantom.max())

    clean_sinogram = A(phantom).to(torch.float64)
    sinogram = torch.tensor(np.random.poisson(clean_sinogram.cpu().numpy())).to(args.device).to(torch.float64)
    sensitivity = AT(torch.ones_like(sinogram) * 1.0) + 1e-7

    m = sinogram.to(torch.float64)
    m_original = m.clone()
    m.requires_grad = False

    results = {}
    for loss_name in ["dist", "nll", "mlem"]:
        print(f"\n--- Running {loss_name} reconstruction ---")
        state = do_PET_reconstruction(
            loss_name=loss_name,
            img_torch=phantom,
            m=m,
            num_iter=args.epochs,
            forward=A,
            backward=AT,
            mumap=mumap,
            sensitivity=sensitivity,
            seed=args.seed,
            return_stats=True,
            return_histos=True,
            return_images=True,
            use_scheduler=args.use_scheduler,
            lr=args.lr
        )
        results[loss_name] = state

    out_dir = os.path.join(args.output_dir, args.save_tag) if args.save_tag else args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'dose={args.dose}_scheduled={args.use_scheduler}_seed={args.seed}_pseed={args.poisson_seed}_BINMASH={args.bin_mash}_IMGSIZE={args.img_size}.pkl')

    with open(out_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nResults saved to {out_file}")

    del results, phantom, m, m_original, clean_sinogram, sinogram, mumap
    empty_cache()


if __name__ == '__main__':
    main()