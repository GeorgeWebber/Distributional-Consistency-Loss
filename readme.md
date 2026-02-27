# Distributional Consistency (DC) Loss
In this repository, we provide scripts and Jupyter Notebooks to reproduce figures from our ICLR paper.

## What Is DC Loss?

In inverse problems, the data term is usually pointwise (MSE, NLL, etc.). These terms work, but have the drawback of attempting to fit the exact noise realization in the measurement. As optimization runs longer, this can turn into noise chasing.

DC loss changes the target. Instead of rewarding pointwise agreement with noisy data, it rewards statistical agreement with the noise model. So the question becomes: "are these measurements plausible under the predicted distribution?" rather than "did we match every value exactly?"

The practical effect is that optimization is much less tempted to fit noise at late iterations. That usually makes long runs more stable, reduces dependence on early stopping, and gives regularization a cleaner job (impose structure, not just fight noise fitting).

It is most useful when:
- the noise model is known (or at least well approximated),
- you have lots of independent measurements,
- and your baseline pointwise loss tends to degrade with iteration count.

<figure>
  <img src="data/explainers/dc_loss_priors.drawio.svg" alt="DC loss explainer" width="700">
  <figcaption>Figure 1: A 2D illustration of the manifold of possible estimated solution to an inverse problem. The maximum likelihood estimate (MLE) is noisy and doesn't account for noise characteristics; estimates with minimum DC loss lie on a manifold an appropriate distance from the MLE. Minimizing MSE or NLL leads to the MLE; minimizing DC loss leads to a point on the minimum DC loss manifold, and so exhibit stable late-iteration optimization behavior.</figcaption>
</figure>

In this repo:
- `src/utils/losses.py` implements the DC loss for different noise models.
- `scripts/DIP.py` compares Deep Image Prior with MSE vs DC on natural images.
- `python scripts/deconv.py` compares a deconvolution task with MSE vs DC. 
- `scripts/PET.py` compares Positron Emission Tomography (PET) reconstruction with NLL, DC, and MLEM.
- `scripts/PET_reg.py` runs regularized PET with NLL/DC under prior penalties.

## Install

Here is the list of packages you need to install to run the scripts and notebooks:
- python = 3.11
- pytorch = 2.1
- torchvision = 0.16
- numpy = 1
- scipy
- scikit-image
- cupy
- matplotlib
- parallelproj
- brainweb
- array-api-compat
- jupyterlab
- ipykernel

Note that brainweb is only available from pypi and parallelproj is only available from conda-forge.

We recommend using the provided environment file with mamba (or conda):
```bash
mamba env create -f environment.yaml
conda activate dc-loss-env
```
Alternatively, you can install each package by hand, referring to the environment file for the source (conda or pip). See https://pytorch.org/get-started/locally/ for information on getting set up using pytorch with cuda support. (The project was tested with PyTorch 2.1 and Cuda toolkit 11.8.)

## Example workflow
### Deconvolution
To reproduce results for deconvolution, run

```python scripts/deconv.py```

This will run a deconvolution experiment and save the results to a file in `results/deconv`. You can then run the notebook `notebooks/deconv_plotting.ipynb` to generate figures.

### Deep Image Prior
To reproduce results for Deep Image Prior, run

```python scripts/DIP.py```

This will run a DIP denoising experiment and save the results to a file in `results/dip`.
You can then run the notebook `notebooks/DIP_plotting.ipynb` to generate figures.
If you wish to generate multiple runs with different noise levels (e.g. to plot the relationship between maximum PSNR achieved and noise level), you can run
```python scripts/DIP.py --seeds 0 1 2 3 4 --sigmas 10 25 50 75 100```

### Positron Emission Tomography (PET) image reconstruction
To reproduce results for PET image reconstruction, run

```python scripts/PET.py```

This will run a PET reconstruction experiment and save the results to a file in `results/PET`. You can then run the notebook `notebooks/PET_plotting.ipynb` to generate figures.

For regularized reconstruction, run ```python scripts/PET_reg.py``` instead, which carried out a simple manual hyperparameter sweep over regularization strength.

## Hyperparameters

The default hyperparameters for each script are those used in the main body of the paper. For more details on the hyperparameters used for the experiments shown in the appendices, consult the paper (e.g. Sections D1, E1 and F1).

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{webber2026distributional,
  title={Distributional Consistency Loss: Beyond Pointwise Data Terms in Inverse Problems},
  author={Webber, George and Reader, Andrew J.},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

Citation metadata is also provided in `CITATION.cff`.

