import os
import sys
import argparse
import numpy as np
import torch
import pickle

from torch.cuda import empty_cache

# --- Set up import path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(script_dir, '..', 'src'))
sys.path.insert(0, src_path)

from utils.phantom import get_brainweb_phantom
from utils.projector import get_projector, get_binmashed_projector_adjoint_function, get_binmashed_projector_forward_function

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.metrics import *
from utils.common_utils import fix_seed

from utils.losses import dc_loss_poisson, total_variation

from scipy.ndimage import binary_fill_holes

dev = "cuda"
RING_FACTOR = 4.0
OUTPUT_DIR = "results/PET_reg"


def parse_args():
    parser = argparse.ArgumentParser(description="Run regularized PET reconstruction sweep.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--bin_mash", type=int, default=2)
    parser.add_argument("--dose", type=float, default=5.0)
    parser.add_argument("--ring_factor", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=10001)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--output_dir", type=str, default="results/PET_reg")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--losses", nargs="+", type=str, default=["nll", "dist"])
    parser.add_argument("--priors", nargs="+", type=str, default=["edge_tv"])
    parser.add_argument("--betas", nargs="+", type=float, default=[
        0.0, 0.001, 0.004, 0.01, 0.04, 0.1, 0.4, 1.0, 4.0, 10.0, 40.0, 100.0,
        400.0, 1000.0, 4000.0, 10000.0, 40000.0, 100000.0,
        0.0, 0.001, 0.002, 0.005, 0.0075, 0.01, 0.012, 0.014, 0.018, 0.02,
        0.022, 0.024, 0.028, 0.03, 0.032, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2,
        0.25, 0.5, 1.0
    ])
    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--stratify", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run a short smoke test configuration")
    parser.add_argument("--save_histos", dest="save_histos", action="store_true", help="Save histogram snapshots in output pickle (default: off)")
    parser.add_argument("--no_save_histos", dest="save_histos", action="store_false", help="Disable histogram snapshots")
    parser.add_argument("--save_images", dest="save_images", action="store_true", help="Save intermediate images in output pickle (default: off)")
    parser.add_argument("--no_save_images", dest="save_images", action="store_false", help="Disable intermediate images")
    parser.set_defaults(save_histos=False, save_images=False)
    return parser.parse_args()


def total_variation_penalty(x):
    return total_variation(x)


def edge_aware_tv(image, kappa=0.1, eps=1e-8):
    image = image.squeeze()
    dx = image[..., 1:, :] - image[..., :-1, :]
    dy = image[..., :, 1:] - image[..., :, :-1]

    wx = torch.exp(-torch.abs(dx) / (kappa + eps))
    wy = torch.exp(-torch.abs(dy) / (kappa + eps))

    tv_x = torch.sqrt(dx**2 + eps) * wx
    tv_y = torch.sqrt(dy**2 + eps) * wy
    return tv_x.mean() + tv_y.mean()


def negative_log_likelihood(x, m, sino_mask, forward):
    q = forward(x).flatten()[sino_mask]
    m = m.flatten()[sino_mask]
    q_plus = torch.nn.functional.relu(q) + 1e-16
    nll = - (m * torch.log(q_plus) - q_plus - torch.lgamma(m + 1)).sum()
    return nll

def dist_loss(x, m, sino_mask, forward, return_values=False):
    q = forward(x).flatten()[sino_mask]
    m = m.flatten()[sino_mask]
    q_plus = torch.nn.functional.relu(q) + 1e-16
    return dc_loss_poisson(q_plus, m, return_values=return_values)

def do_PET_reconstruction(loss_name, img_torch, m, num_iter, forward, backward, beta=None, prior=None, start_image=None, seed=0, return_stats=False, return_histos=False, return_images=False, use_scheduler=False, lr=1e-3, stratify=False):
    global mask, A, AT, sino_mask, sensitivity

    sino_mask_torch = sino_mask
    img_np = img_torch.detach().cpu().numpy().squeeze()

    fix_seed(seed)
    
    state = {
        "last_image": None,
        "images": [],
        "psnr_noisy_last": 0,
        "losses": {
            "psnr": [], "ssim": [], "loss": [],
            "nll_loss": [], "dist_loss": [], "nrmse": [], "fidelity_loss": [], "prior_loss": []
        },
        "histos": [],
        "point_mse": [],
        "point_crc": [],
        "prior": prior,
        "beta": beta
    }

    if start_image is None:
        x = torch.ones_like(img_torch).to(torch.float64).to(dev) * img_torch[img_torch > 0].mean()
        x = x * mask
        x.requires_grad = True
    else:
        x = torch.tensor(start_image).to(dev).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        x = x * mask
        x.requires_grad = True

    if loss_name == "mlem":
        x.requires_grad = False
    else:
        def preconditioned_grad_hook(grad):
            return (grad * mask) / (sensitivity + 1e-6)


    min_lr = lr / 100

    optimizer = torch.optim.Adam([x], lr=lr)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=False)

    for i in range(num_iter):
        empty_cache()
        optimizer.zero_grad()

        if loss_name in ["nll", "dist"]:
            new_x = x * mask
            new_x.register_hook(preconditioned_grad_hook)

            if loss_name == "nll":
                loss = negative_log_likelihood(new_x, m, sino_mask_torch, forward)
            elif loss_name == "dist":
                loss = dist_loss(new_x, m, sino_mask_torch, forward)

            fidelity_loss = loss.item()
            if prior is not None and beta is not None:
                if prior == "tv":
                    prior = total_variation_penalty
                elif prior == "perfect":
                    prior = lambda x: ((x - img_torch)**2).mean()
                elif prior == "edge_tv":
                    prior = edge_aware_tv
                prior_loss = prior(x).mean()

                loss += beta * prior_loss

                prior_loss = prior_loss.item()

            loss.backward()

            optimizer.step()
            if use_scheduler:
                scheduler.step(loss.detach().item())
                current_lr = optimizer.param_groups[0]['lr']
                if current_lr <= min_lr:
                    print("Stopping early: Learning rate has reached the minimum threshold.")
                    break
        elif loss_name == "mlem":
            new_x = (x/sensitivity) * (backward(m / (forward(x) + 1e-16))) * mask
            x = new_x
            loss = negative_log_likelihood(new_x, m, sino_mask_torch, forward).detach()

        # PSNR Metrics
        out_np = torch.nn.functional.relu(new_x).detach().cpu().numpy().squeeze()
        psnr_gt = compare_psnr(img_np, out_np, data_range=img_np.max()-img_np.min())

        state["last_image"] = out_np

        print(f'Iteration {i:05d} Loss {loss.item():.6f} PSNR_gt: {psnr_gt:.2f}', '\r', end='')

        # Record stats
        if return_stats:
            with torch.no_grad():
                lesion_roi = np.zeros_like(img_np, dtype=bool)
                lesion_roi[31:35, 49:53] = 1
                background_roi = np.zeros_like(img_np, dtype=bool)
                background_roi[31:34, 74:77] = 1

                state["point_mse"].append((np.mean(img_np[31:35, 49:53]) - np.mean(out_np[31:35, 49:53]))**2)
                state["point_crc"].append(CRC(out_np, img_np, lesion_roi, background_roi))
                state["losses"]["psnr"].append(psnr_gt)
                nrmse = np.sqrt(np.mean((img_np - out_np)**2)) / np.sqrt(np.mean(img_np**2))
                state["losses"]["nrmse"].append(nrmse)
                state["losses"]["ssim"].append(SSIM(img_np, out_np))
                state["losses"]["loss"].append(loss.item())
                state["losses"]["fidelity_loss"].append(fidelity_loss)
                state["losses"]["prior_loss"].append(prior_loss if (prior is not None and beta is not None) else 0.0)
                if loss_name == "nll":
                    state["losses"]["nll_loss"].append(fidelity_loss)
                else:
                    state["losses"]["nll_loss"].append(negative_log_likelihood(new_x, m, sino_mask_torch, forward).item())
                if i % 100 == 0 and return_histos:
                    dist_loss_value, histo_values, sort_ix = dist_loss(new_x, m, sino_mask, forward, return_values=True)
                    state["histos"].append((histo_values.detach().cpu().numpy(), sort_ix.detach().cpu().numpy()))
                    state["losses"]["dist_loss"].append(dist_loss_value.item())
                else:
                    if loss_name == "dist":
                        state["losses"]["dist_loss"].append(fidelity_loss)
                    else:
                        state["losses"]["dist_loss"].append(dist_loss(new_x, m, sino_mask, forward, return_values=False).item())
        if return_images and i % 10 == 0:
            state["images"].append(out_np)
        del new_x, loss
        empty_cache()

    if return_stats:
        state["losses"]["psnr"] = np.array(state["losses"]["psnr"])
        state["losses"]["ssim"] = np.array(state["losses"]["ssim"])
        state["losses"]["nrmse"] = np.array(state["losses"]["nrmse"])
        state["losses"]["loss"] = np.array(state["losses"]["loss"])
        state["losses"]["nll_loss"] = np.array(state["losses"]["nll_loss"])
        state["losses"]["dist_loss"] = np.array(state["losses"]["dist_loss"])
        state["point_mse"] = np.array(state["point_mse"])
    
    return state

# -----------------------------------------------------------------------------------------------------------------------------------------------
def main():
    global dev, RING_FACTOR, OUTPUT_DIR, mask, A, AT, sino_mask, sensitivity

    args = parse_args()
    dev = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    RING_FACTOR = args.ring_factor
    OUTPUT_DIR = args.output_dir

    if args.smoke:
        args.seeds = [args.seeds[0]]
        args.priors = [args.priors[0]]
        args.betas = [0.01]
        args.epochs = min(args.epochs, 20)
        args.losses = ["nll", "dist"]
        args.save_histos = True
        args.save_images = True

    for seed in args.seeds:
        fix_seed(seed)

        img_size = args.img_size
        bin_mash = args.bin_mash
        dose = args.dose
        dose_scaled = dose * (img_size / 128) ** 2 / RING_FACTOR

        projector = get_projector(dev, img_size, ring_factor=RING_FACTOR)
        A = get_binmashed_projector_forward_function(projector, bin_mash)
        AT = get_binmashed_projector_adjoint_function(projector, bin_mash)

        phantom = torch.tensor(get_brainweb_phantom(n=img_size).copy()).to(torch.float64)
        nonzero_mask = (phantom.numpy() != 0).astype(np.uint8)
        mumap = binary_fill_holes(nonzero_mask).astype(np.uint8)[None, None, ..., None]
        mumap = torch.tensor(mumap).to(dev)

        phantom = phantom * dose_scaled
        phantom = phantom[None, None, ..., None].to(dev).to(torch.float64)

        clean_sinogram = A(phantom).to(torch.float64)
        sinogram = torch.tensor(np.random.poisson(clean_sinogram.cpu().numpy())).to(dev).to(torch.float64)
        sensitivity = AT(torch.ones_like(sinogram) * 1.0) + 1e-7

        mask = (mumap > 0).to(torch.float64)
        sino_mask = A(mask).flatten() > 1e-2

        m = sinogram.to(torch.float64)
        m.requires_grad = False

        for prior in args.priors:
            for beta in args.betas:
                print(f"Seed={seed} Prior={prior} Beta={beta}")
                new_states = {}
                for loss_name in args.losses:
                    state = do_PET_reconstruction(
                        loss_name,
                        phantom,
                        m,
                        args.epochs,
                        A,
                        AT,
                        prior=prior,
                        beta=beta,
                        start_image=None,
                        seed=seed,
                        return_stats=True,
                        return_histos=args.save_histos,
                        return_images=args.save_images,
                        use_scheduler=args.use_scheduler,
                        lr=args.lr,
                    )
                    new_states[loss_name] = state

                os.makedirs(OUTPUT_DIR, exist_ok=True)
                out_file = os.path.join(
                    OUTPUT_DIR,
                    f"dose={dose_scaled}_scheduled={args.use_scheduler}_pseed={seed}_seed={seed}_BINMASH={bin_mash}_IMGSIZE={img_size}_prior={prior}_beta={beta}_v2.pkl",
                )
                with open(out_file, "wb") as f:
                    pickle.dump(new_states, f)
                print(f"Saved {out_file}")
                empty_cache()

        del A, AT, projector
        empty_cache()


if __name__ == "__main__":
    main()
