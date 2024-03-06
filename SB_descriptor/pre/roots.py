import numpy as np
import scipy as sp
import scipy.optimize
import jax
import sys
sys.path.append("/home/rhji/train/src/neuralil/neural_network/spherical_bessel")
import functions

# The calculation starts with the roots of the order-0 function and
# proceeds upwards. Therefore, caching is essential.


class SphericalBesselRoots:
    """Calculator and cache of the roots of spherical Bessel functions.

    Access to the roots is supposed to happen directly through the "table"
    attribute. The first index is the order of the function, and the second
    is the order of the root. A negative value means that the corresponding
    root has not been calculated.

    Args:
        n_max: The maximum value of n to be used when generating the spherical
            Bessel descriptors. This affects which roots are calculated.
        dtype: The floating-point type to be used for the calculations.

    Raises:
        ValueError if n_max is negative.
    """
    def __init__(self, n_max: int, dtype: np.dtype = np.float32):
        if n_max < 0:
            raise ValueError("n_max cannot be negative")
        self.n_max = n_max
        # Initialize the table of precomputed values. Note that, even though
        # a rectangular NumPy array is used for convenience and performance,
        # the higher the order of the function, the fewer roots are computed.
        self.table = -np.ones((self.n_max + 1, self.n_max + 2))
        # The zeros of j_0(r) are trivially generated.
        self.table[0, :] = np.pi * np.arange(1, n_max + 3)
        # Proceed upwards from there, by using the fact that the roots of the
        # order-l function bracket the roots of the order-(l+1) function, per
        # Abramowitz and Stegun.
        for order in range(1, self.n_max + 1):
            # This could be made faster by using the SciPy implementation of
            # the spherical Bessels functions directly, but it only needs
            # to be done once and we get values of the roots that are
            # better adapted to our implementation.
            function = jax.jit(functions.create_j_l(order, dtype))
            for n in range(self.n_max + 2 - order):
                left = self.table[order - 1, n]
                right = self.table[order - 1, n + 1]
                self.table[order, n] = sp.optimize.brentq(function, left, right)
