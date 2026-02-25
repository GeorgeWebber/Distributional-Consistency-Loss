
"""basic linear operators"""

from __future__ import annotations

from types import ModuleType
import abc
import numpy as np
import array_api_compat
from array_api_compat import device
from collections.abc import Sequence

import parallelproj


class LinearOperator(abc.ABC):
    """abstract base class for linear operators"""

    def __init__(self) -> None:
        self._scale = 1

    @property
    @abc.abstractmethod
    def in_shape(self) -> tuple[int, ...]:
        """shape of the input array"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def out_shape(self) -> tuple[int, ...]:
        """shape of the output array"""
        raise NotImplementedError

    @property
    def scale(self) -> int | float | complex:
        """scalar factor applied to the linear operator"""
        return self._scale

    @scale.setter
    def scale(self, value: int | float | complex):
        if not np.isscalar(value):
            raise ValueError
        self._scale = value

    @abc.abstractmethod
    def _apply(self, x):
        """forward step :math:`y = Ax`"""
        raise NotImplementedError

    @abc.abstractmethod
    def _adjoint(self, y):
        """adjoint step :math:`x = A^H y`"""
        raise NotImplementedError


    def apply(self, x):
        """(scaled) forward step :math:`y = \\alpha A x`

        Parameters
        ----------
        x : Array

        Returns
        -------
        Array
        """
        if self._scale == 1:
            return self._apply(x)
        else:
            return self._scale * self._apply(x)
        

    def __call__(self, x):
        """alias to apply(x)"""
        return self.apply(x)


    def adjoint(self, y):
        """(scaled) adjoint step :math:`x = \\overline{\\alpha} A^H y`

        Parameters
        ----------
        y : Array

        Returns
        -------
        Array
        """

        if self._scale == 1:
            return self._adjoint(y)
        else:
            return self._scale.conjugate() * self._adjoint(y)


    def adjointness_test(
        self,
        xp: ModuleType,
        dev: str,
        verbose: bool = False,
        iscomplex: bool = False,
        **kwargs,
    ) -> bool:
        """test whether the adjoint is correctly implemented

        Parameters
        ----------
        xp : ModuleType
            array module to use
        dev : str
            device (cpu or cuda)
        verbose : bool, optional
            verbose output
        iscomplex : bool, optional
            use complex arrays
        **kwargs : dict
            passed to np.isclose

        Returns
        -------
        bool
            whether the adjoint is correctly implemented
        """

        if iscomplex:
            dtype = xp.complex128
        else:
            dtype = xp.float64

        x = xp.asarray(np.random.rand(*self.in_shape), device=dev, dtype=dtype)
        y = xp.asarray(np.random.rand(*self.out_shape), device=dev, dtype=dtype)

        if iscomplex:
            x = x + 1j * xp.asarray(
                np.random.rand(*self.in_shape), device=dev, dtype=dtype
            )

        if iscomplex:
            y = y + 1j * xp.asarray(
                np.random.rand(*self.out_shape), device=dev, dtype=dtype
            )

        x_fwd = self.apply(x)
        y_adj = self.adjoint(y)

        if iscomplex:
            ip1 = complex(xp.sum(xp.conj(x_fwd) * y))
            ip2 = complex(xp.sum(xp.conj(x) * y_adj))
        else:
            ip1 = float(xp.sum(x_fwd * y))
            ip2 = float(xp.sum(x * y_adj))

        if verbose:
            print(ip1, ip2)

        return np.isclose(ip1, ip2, **kwargs)

    def norm(
        self,
        xp: ModuleType,
        dev: str,
        num_iter: int = 30,
        iscomplex: bool = False,
        verbose: bool = False,
    ) -> float:
        """estimate norm of the linear operator using power iterations

        Parameters
        ----------
        xp : ModuleType
            array module to use
        dev : str
            device (cpu or cuda)
        num_iter : int, optional
            number of power iterations
        iscomplex : bool, optional
            use complex arrays
        verbose : bool, optional
            verbose output

        Returns
        -------
        float
            the norm of the linear operator
        """

        if iscomplex:
            dtype = xp.complex128
        else:
            dtype = xp.float64

        x = xp.asarray(np.random.rand(*self.in_shape), device=dev, dtype=dtype)

        if iscomplex:
            x = x + 1j * xp.asarray(
                np.random.rand(*self.in_shape), device=dev, dtype=dtype
            )

        for i in range(num_iter):
            x = self.adjoint(self.apply(x))
            norm_squared = xp.sqrt(xp.sum(xp.abs(x) ** 2))
            x /= float(norm_squared)

            if verbose:
                print(f"{(i+1):03} {xp.sqrt(norm_squared):.2E}")

        return float(xp.sqrt(norm_squared))

class GaussianFilterOperator(LinearOperator):
    """Gaussian filter operator

    Examples
    --------
    .. minigallery:: parallelproj.GaussianFilterOperator
    """

    def __init__(self, in_shape: tuple[int, ...], sigma, **kwargs):
        """init method

        Parameters
        ----------
        in_shape : tuple[int, ...]
            shape of the input array
        sigma: float | array
            standard deviation of the gaussian filter
        **kwargs : sometype
            passed to the ndimage gaussian_filter function
        """
        super().__init__()
        self._in_shape = in_shape
        self._sigma = sigma
        self._kwargs = kwargs

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self._in_shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        return self._in_shape

    def _apply(self, x):
        xp = array_api_compat.get_namespace(x)

        if parallelproj.is_cuda_array(x):
            import array_api_compat.cupy as cp
            import cupyx.scipy.ndimage as ndimagex

            if array_api_compat.is_array_api_obj(self._sigma):
                sigma = cp.asarray(self._sigma)
            else:
                sigma = self._sigma

            return xp.asarray(
                ndimagex.gaussian_filter(cp.asarray(x), sigma=sigma, **self._kwargs),
                device=device(x),
            )
        else:
            import scipy.ndimage as ndimage

            if array_api_compat.is_array_api_obj(self._sigma):
                sigma = np.asarray(self._sigma)
            else:
                sigma = self._sigma

            return xp.asarray(
                ndimage.gaussian_filter(np.asarray(x), sigma=sigma, **self._kwargs),
                device=device(x),
            )

    def _adjoint(self, y):
        return self._apply(y)

class IdentityOperator(LinearOperator):
    """Identity operator"""

    def __init__(self, in_shape: tuple[int, ...]) -> None:
        """init method

        Parameters
        ----------
        in_shape : tuple[int, ...]
            shape of the input array
        """
        super().__init__()
        self._in_shape = in_shape

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self._in_shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        return self._in_shape

    def _apply(self, x):
        return x

    def _adjoint(self, y):
        return y