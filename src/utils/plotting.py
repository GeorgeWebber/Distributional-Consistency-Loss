import numpy as np

"""A few useful plotting utilities for DC loss."""

def plot_CDF_histogram(ax, logit_values, comparison = "uniform", show_comparison=True, bins=100, color='blue', alpha=0.7, label=None):
    """
    Plots a histogram of CDF or logit-transformed CDF values and optionally overlays a comparison distribution.
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object where the histogram will be plotted.
    logit_values : array-like
        The input values to be transformed and plotted. If `comparison` is "uniform", 
        the values are transformed using the sigmoid function.
    comparison : str, optional
        The type of comparison distribution to overlay. Options are:
        - "uniform" (default): Overlay a uniform distribution PDF on [0, 1].
        - "logistic": Overlay a logistic distribution PDF.
    show_comparison : bool, optional
        Whether to overlay the comparison distribution on the histogram. Default is True.
    bins : int, optional
        The number of bins to use for the histogram. Default is 100.
    color : str, optional
        The color of the histogram bars. Default is 'blue'.
    alpha : float, optional
        The transparency level of the histogram bars. Default is 0.7.
    label : str, optional
        The label for the histogram, used in the legend. Default is None.
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The Axes object with the plotted histogram and optional comparison distribution.
    Notes:
    ------
    - When `comparison` is "uniform", the input values are transformed using the sigmoid function: 
        `1 / (1 + exp(-logit_values))`.
    - When `comparison` is "logistic", the x-axis limits are set symmetrically around zero based on 
        the maximum absolute bin value.
    - The function automatically labels the x-axis and y-axis based on the `comparison` type.
    """
    if comparison == "uniform":
        values = 1.0 / (1. + np.exp(-logit_values))
    else:
        values = logit_values
    
    bins = ax.hist(values, bins=bins, color=color, alpha=alpha, label=label, density=True)[1]
    if comparison == "logistic":
        max_abs_bin = np.max(np.abs(bins))
        ax.set_xlim(-max_abs_bin, max_abs_bin)
    if show_comparison:
        if comparison == "uniform":
            x = np.linspace(0, 1, 1000)
            y = np.ones_like(x) * 1.0
            ax.plot(x, y, color='black', linestyle='--', label='Uniform([0,1]) PDF')
        elif comparison == "logistic":
            x = np.linspace(-max_abs_bin, max_abs_bin, 1000)
            y = np.exp(-x) / ((1 + np.exp(-x))**2)
            ax.plot(x, y, color='black', linestyle='--', label='Logistic(0,1) PDF')

    if label:
        ax.legend()

    if comparison == "uniform":
        ax.set_xlabel(r'CDF values ($s_i$ values)')
    elif comparison == "logistic":
        ax.set_xlabel(r'Logit(CDF values) ($z_i$ values)')
    
    ax.set_ylabel('Probability density')
    return ax

def hide(ax):
    """A utility function to hide the axes and spines of a matplotlib Axes object, but not the y or x labels."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return ax