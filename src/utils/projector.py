import array_api_compat.torch as torch
from .projector_utils.utils import RegularPolygonPETScannerGeometry, RegularPolygonPETLORDescriptor, RegularPolygonPETNonTOFProjector
from .projector_utils.layers import LinearSingleChannelOperator, AdjointLinearSingleChannelOperator

def get_projector(dev="cpu", IMG_SIZE=128, ring_factor=1.0):
    """
    Creates and returns a PET (Positron Emission Tomography) projector object 
    configured with a regular polygon scanner geometry and specified parameters.
    Args:
        dev (str, optional): The device to use for computations, e.g., "cpu" or "cuda". 
                             Defaults to "cpu".
        IMG_SIZE (int, optional): The size of the image (in pixels) for the projector. 
                                  Defaults to 128.
        ring_factor (float, optional): A scaling factor that adjusts the number of 
                                       rings and radial trim. Defaults to 1.0.
    Returns:
        RegularPolygonPETNonTOFProjector: A configured PET projector object.
    """

    rings_per_side = int(9 * ring_factor)

    scanner_geometry = RegularPolygonPETScannerGeometry(
        xp = torch,
        dev = dev,
        radius = 337,
        num_sides = 56,
        num_lor_endpoints_per_side = rings_per_side,
        lor_spacing = 2 * 2.0445 * 9. / rings_per_side,
        num_rings = 1,
        ring_positions = torch.tensor([0]),
        symmetry_axis = 2
    )

    lor_descriptor = RegularPolygonPETLORDescriptor(
        scanner = scanner_geometry,
        radial_trim = int(80 * ring_factor),
        max_ring_difference = 60
    )

    view_tensor = torch.arange(0, lor_descriptor.num_views, int(ring_factor), device=dev)

    projector = RegularPolygonPETNonTOFProjector(
        lor_descriptor,
        (IMG_SIZE, IMG_SIZE, 1),
        ((128 / IMG_SIZE) * 2.03, (128/IMG_SIZE) * 2.03, 2.08),
        views = view_tensor,
        resolution_model = "identity") # "identity")
    return projector

# --- Binmashing functions ---
# These functions are used to reduce the number of detector bins in the sinogram

def _sino_bin_mash_forward(sino: torch.Tensor, mash_factor: int) -> torch.Tensor:
    """
    Sum neighbouring detector bins (dim=1) in groups of `mash_factor`.

    sino:  (B, Bins, Views, …)   e.g. (1, 344, 252, 1)
    returns (B, Bins//mash_factor, Views, …)
    """
    B, nbins, *rest = sino.shape
    # keep only a multiple of m
    m = nbins // mash_factor
    sino = sino[:, :m * mash_factor, ...]
    sino = sino.reshape(B, m, mash_factor, *rest).sum(dim=2)
    return sino

def _sino_bin_mash_adjoint(sino_mashed: torch.Tensor, mash_factor: int) -> torch.Tensor:
    """
    Transpose of the forward bin-mash:  repeat-interleave along dim=1.
    NB: dividing by mash_factor keeps ⟨Ax,y⟩ = ⟨x,Aᵀy⟩.
    """
    sino = torch.repeat_interleave(sino_mashed, repeats=mash_factor, dim=1)
    return sino / mash_factor

# --- Factory functions for forward and adjoint functions (either with or without bin-mashing) ---

def get_projector_forward_function(projector):
    def forward_function(x):
        return LinearSingleChannelOperator.apply(x, projector)
    return forward_function

def get_projector_adjoint_function(projector):
    def adjoint_function(x):
        return AdjointLinearSingleChannelOperator.apply(x, projector)
    return adjoint_function

def get_binmashed_projector_forward_function(projector, bin_mashing_factor: int):
    def forward_function(x):
        sino = LinearSingleChannelOperator.apply(x, projector)
        sino = _sino_bin_mash_forward(sino, bin_mashing_factor)
        return sino
    return forward_function

def get_binmashed_projector_adjoint_function(projector, bin_mashing_factor: int):
    def adjoint_function(y_mashed):
        sino = _sino_bin_mash_adjoint(y_mashed, bin_mashing_factor)
        x = AdjointLinearSingleChannelOperator.apply(sino, projector)
        return x
    return adjoint_function
