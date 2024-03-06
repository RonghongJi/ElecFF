import numpy as np
import jax
import sys
sys.path.append("/home/rhji/train/src/neuralil/neural_network/spherical_bessel")

import functions
import roots


class SphericalBesselCoefficients:
    """Calculator and cache of the coefficients of the descriptors.

    The coefficients are available directly through the attributes c_1 and c_2.

    Args:
        n_max: The maximum value of n to be used when generating the spherical
            Bessel descriptors. This affects which coefficients are calculated.
        dtype: The floating-point type to be used for the calculations.

    Raises:
        ValueError if n_max is negative.
    """
    def __init__(self, n_max: int, dtype: np.dtype = np.float32):
        if n_max < 0:
            raise ValueError("n_max cannot be negative")
        self.n_max = n_max
        # We need to go one step beyond n_max to get all the coefficients we
        # need, so we keep an internal version incremented by one.
        self.__n_max = self.n_max + 1
        # Initialize the table of Bessel roots.
        self.roots = roots.SphericalBesselRoots(self.__n_max, dtype)
        # Create the tables to store the coefficients. Just like for the
        # roots, we store them regular arrays for convenience but only use
        # half of each array. Non initialized elements are set to NaN.
        self.c_1 = (np.nan * np.ones((self.__n_max + 1, self.__n_max + 1)
                                    )).astype(dtype)
        self.c_2 = (np.nan * np.ones((self.__n_max + 1, self.__n_max + 1)
                                    )).astype(dtype)
        for order in range(self.__n_max + 1):
            function = jax.jit(functions.create_j_l(order + 1, dtype))
            u_0 = self.roots.table[order, :self.__n_max - order + 1]
            u_1 = self.roots.table[order, 1:self.__n_max - order + 2]
            coeff = np.sqrt(2. / (u_0 * u_0 + u_1 * u_1))
            self.c_1[order, :self.__n_max - order +
                     1] = u_1 / function(u_0) * coeff
            self.c_2[order, :self.__n_max - order +
                     1] = u_0 / function(u_1) * coeff
