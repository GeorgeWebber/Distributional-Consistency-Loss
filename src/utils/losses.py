import torch

eps_hard = 1e-10

def dc_loss(q, m, cdf_params, return_values=False):
    """
    Computes the distributional consistency (DC) loss based on the Wasserstein distance
    between the predicted and target distributions.
    Args:
        q (torch.Tensor): The predicted values, expected to be a tensor.
        m (torch.Tensor): The target values, expected to be a tensor.
        cdf_params (dict): A dictionary containing the parameters for the cumulative 
            distribution function (CDF). Must include the key "name" to specify the 
            distribution type. Supported values for "name" are:
                - "gaussian": Requires "sigma" (standard deviation) in `cdf_params`.
                - "clipped_gaussian": Requires "sigma" (standard deviation) in `cdf_params`.
                - "poisson": No additional parameters required.
        return_values (bool, optional): If True, returns additional intermediate values 
            (sorted log-s values and their indices). Defaults to False.
    Returns:
        torch.Tensor or tuple: If `return_values` is False, returns the Wasserstein distance 
        as a scalar tensor. If `return_values` is True, returns a tuple containing:
            - wasserstein_distance (torch.Tensor): The computed Wasserstein distance.
            - log_s_values (torch.Tensor): The sorted log-s values.
            - ix (torch.Tensor): The indices of the sorted log-s values.
    Raises:
        NotImplementedError: If the specified distribution in `cdf_params["name"]` is not 
        implemented.
    """
    q = q.flatten()
    m = m.flatten()
    if cdf_params["name"] == "gaussian":
        sigma = cdf_params["sigma"]
        log_s_values = get_logistic_gaussian_cdf(q, m, sigma)
    elif cdf_params["name"] == "clipped_gaussian":
        sigma = cdf_params["sigma"]
        log_s_values = get_logistic_gaussian_clipped_cdf(q, m, sigma)
    elif cdf_params["name"] == "poisson":
        log_s_values = get_logistic_poisson_cdf(q, m)
    else:
        raise NotImplementedError(f"Distribution {cdf_params['name']} not implemented.")

    logistic_values = sample_logistic_distribution(m.shape, device=m.device)
    logistic_values, _ = torch.sort(logistic_values)
    log_s_values, ix = torch.sort(log_s_values)

    wasserstein_distance = (log_s_values - logistic_values).abs().mean()

    if return_values:
        return wasserstein_distance, log_s_values, ix
    return wasserstein_distance

# --- Loss function wrappers for different distributions ---

def dc_loss_poisson(q, m, return_values=False):
    cdf_params = {"name": "poisson"}
    return dc_loss(q, m, cdf_params, return_values=return_values)
  
def dc_loss_gaussian(q, m, sigma, return_values=False):
    cdf_params = {"name": "gaussian", "sigma": sigma}
    return dc_loss(q, m, cdf_params, return_values=return_values)

def dc_loss_clipped_gaussian(q, m, sigma, return_values=False):
    cdf_params = {"name": "clipped_gaussian", "sigma": sigma}
    return dc_loss(q, m, cdf_params, return_values=return_values)

# --- Utility functions for sampling and computing CDFs ---

def sample_logistic_distribution(shape, device="cuda:0"):
    """Samples i.i.d. from the Logistic(0,1) distribution with the specified shape."""
    base_distribution = torch.distributions.Uniform(0.0, 1.0)
    transforms = [torch.distributions.SigmoidTransform().inv, torch.distributions.AffineTransform(loc=0.0, scale=1.0)]
    logistic_dist = torch.distributions.TransformedDistribution(base_distribution, transforms)
    logistic_values = logistic_dist.sample(sample_shape=shape).to(device=device)
    return logistic_values


def get_logistic_poisson_cdf(q, m):
    """Approximates logit(CDF) for a Poisson distribution centered at q (where q and m are tensors with the same shape)."""
    eps = 1e-7
    q = torch.nn.functional.relu(q)
    m = torch.nn.functional.relu(m)

    cdf_values = 1 - torch.distributions.Gamma(m+1, torch.ones_like(m)).cdf(q)
    cdf_values = torch.clamp(cdf_values, min=eps, max=1.0 - eps)

    lm = (cdf_values <= eps)   # lower_mask
    um = (cdf_values >= (1 - eps))   # upper_mask
    cm = ~(lm | um)   # center_mask

    logit_s = torch.zeros_like(cdf_values)
    logit_s[cm] = torch.logit(cdf_values[cm])

    # Compute logit values for lower and upper tails
    logit_s[lm] = m[lm] * torch.log(q[lm]) - q[lm] - torch.lgamma(m[lm]+1.)
    logit_s[um] = q[um] - m[um] * torch.log(q[um]) + torch.lgamma(m[um]+2.)
    
    return logit_s


def get_logistic_gaussian_cdf(q, m, sigma):
    """Approximates logit(CDF) for a Gaussian distribution centered at q (where q and m are tensors with the same shape)."""
    eps = 1e-4
    cdf_values = torch.distributions.Normal(q, sigma).cdf(m)
    cdf_values = torch.clamp(cdf_values, min=eps, max=1.0 - eps)

    lm = cdf_values <= (eps + 1e-10)  # lower_mask
    um = cdf_values >= ((1 - eps) - 1e-10)  # upper_mask
    cm = ~(lm | um)   # center_mask

    logit_s = torch.zeros_like(cdf_values)
    logit_s[cm] = torch.logit(cdf_values[cm])
    z_scores = (m - q) / sigma

    log_sqrt_tau = torch.log(torch.sqrt(2 * torch.tensor([torch.pi]))).to(m.device)

    # Compute logit values for lower and upper tails
    logit_s[lm] = -((z_scores[lm]**2) / 2 + torch.log(torch.abs(z_scores[lm])) + log_sqrt_tau)
    logit_s[um] = ((z_scores[um]**2) / 2 + torch.log(torch.abs(z_scores[um])) + log_sqrt_tau)
    return logit_s


def get_logistic_gaussian_clipped_cdf(q, m, sigma, eps=1e-3):
    """
    Approximates logit(CDF) for a clipped Gaussian distribution centered at q, with:
    - epsilon-wide linear ramps near 0 and 1
    - random perturbation of measurements exactly at 0 or 1
    - appropriate tail approximations where q is far from m
    """
    device = m.device
    normal = torch.distributions.Normal(q, sigma)

    # Randomly sample m values from a small eps-wide interval when they are exactly 0 or 1
    m = m.clone()
    m[m == 0] = torch.rand_like(m[m == 0]) * eps
    m[m == 1] = 1 - torch.rand_like(m[m == 1]) * eps

    # Get normal CDF values
    cdf_m = normal.cdf(m)

    # Compute CDF endpoints for ramps
    cdf_eps = normal.cdf(torch.full_like(m, eps))
    cdf_1_minus_eps = normal.cdf(torch.full_like(m, 1 - eps))

    # Compute output s values based on where m lies
    s = torch.empty_like(m)

    left_mask = m < eps
    right_mask = m > 1 - eps
    center_mask = ~(left_mask | right_mask)

    # Linear ramp near 0: m in [0, eps]
    s[left_mask] = (m[left_mask] / eps) * cdf_eps[left_mask]

    # Linear ramp near 1: m in [1 - eps, 1]
    s[right_mask] = cdf_1_minus_eps[right_mask] + (
        (m[right_mask] - (1 - eps)) / eps * (1 - cdf_1_minus_eps[right_mask])
    )

    # Use normal CDF in the central region
    s[center_mask] = cdf_m[center_mask]

    # Clamp to avoid instability in logit
    s = torch.clamp(s, eps_hard, 1 - eps_hard)

    # Compute logit with tail approximations
    logit_s = torch.empty_like(s)
    z_scores = (m - q) / sigma
    log_sqrt_tau = torch.log(torch.sqrt(2 * torch.tensor(torch.pi, device=device)))

    low_mask = s < 1e-4
    high_mask = s > 1 - 1e-4
    mid_mask = ~(low_mask | high_mask)

    # Compute logit values for lower and upper tails
    logit_s[low_mask] = -(
        (z_scores[low_mask] ** 2) / 2
        + torch.log(torch.abs(z_scores[low_mask]) + 1e-12)
        + log_sqrt_tau
    )

    logit_s[high_mask] = (
        (z_scores[high_mask] ** 2) / 2
        + torch.log(torch.abs(z_scores[high_mask]) + 1e-12)
        + log_sqrt_tau
    )

    logit_s[mid_mask] = torch.logit(s[mid_mask])

    return logit_s


def total_variation(x):
    """Computes the total variation of a 2D image tensor. Assumes the last two dimensions are the spatial dimensions."""
    # Compute the gradients along the x and y axes
    dx = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    dy = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    # Sum the gradients to get the total variation
    return torch.sum(dx) + torch.sum(dy)
