# Copyright 2019-2021 Jesús Carrete Montaña
#                     Hadrián Montes Campos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import numpy as onp
import scipy as sp
import scipy.linalg as la

import jax
import jax.nn
import jax.lax
import jax.numpy as jnp
import jax.numpy.linalg
import jax.config

import sys
sys.path.append("/home/rhji/train/src/neuralil/neural_network/spherical_bessel")

import functions
import coefficients

# This module variable holds the number of columns in the precomputed Bessel
# coefficients table, and is increased when more are requested.
# The initial calculation can take some time.
_table_size = 10
_coeffs = coefficients.SphericalBesselCoefficients(_table_size)


# Slightly tweaked version of the real square root and its derivative. The
# derivative at zero is defined as zero.
@jax.custom_jvp
def _sqrt(x):
    return jnp.sqrt(x)


@_sqrt.defjvp
def _sqrt_jvp(primals, tangents):
    x, = primals
    xdot, = tangents
    primal_out = _sqrt(x)
    tangent_out = jnp.where(x == 0., 0., 0.5 / primal_out) * xdot
    return (primal_out, tangent_out)


class EllChannel:
    def __init__(self, ell: int, max_order: int, cutoff_radius: float):
        """Class taking care of radial calculations for a single value of l.

        Parameters
        ----------
        ell: int
            The order of this radial channel
        max_order: int
            The maximum radial order (n) of the bessel functions
        """
        global _table_size
        global _coeffs

        self._order = ell
        self._n_max = max_order
        self._r_c = cutoff_radius

        # Create the basic blocks of the non-orthogonal basis functions.
        self._function = functions.create_j_l(ell)

        # If the table of precalculated roots and coefficients are not big
        # enough, increase their size.
        n_cols = max_order + 1
        while True:
            self._c_1 = _coeffs.c_1[ell, :n_cols]
            self._c_2 = _coeffs.c_2[ell, :n_cols]
            self._roots = _coeffs.roots.table[ell, :n_cols + 1]
            if (
                self._c_1.size == n_cols and self._c_2.size == n_cols and
                self._roots.size == n_cols + 1 and
                jnp.all(jnp.isfinite(self._c_1)) and
                jnp.all(jnp.isfinite(self._c_2)) and jnp.all(self._roots > 0.)
            ):
                break
            table_size = 2 * _table_size + 1
            coeffs = coefficients.SphericalBesselCoefficients(table_size)
            _table_size = table_size
            _coeffs = coeffs

        # Obtain the transformation matrix required to orthogonalize them.
        u_sq = self._roots * self._roots
        u_1 = u_sq[0]
        u_2 = u_sq[1]
        d_1 = 1.

        self._transformation = onp.eye(n_cols)
        for n in range(1, n_cols):
            u_0 = u_1
            u_1 = u_2
            u_2 = u_sq[n + 1]

            e = (u_0 * u_2) / ((u_0 + u_1) * (u_1 + u_2))
            d_0 = d_1
            d_1 = 1. - e / d_0
            self._transformation[n, n] /= onp.sqrt(d_1)
            self._transformation[
                n, :] += onp.sqrt(e /
                                  (d_1 * d_0)) * self._transformation[n - 1, :]

    def __call__(self, distances: jnp.array) -> jnp.ndarray:
        """
        Parameters
        ----------
        Distances: jnp.array[n_r] of floats
            Vector with all the distances to evaluate, or a float with only 1
            distance

        Returns
        -------
        jnp.array[max_order+1, n_r] or jnp.array[max_order+1, 1]
            Matrix with the function evaluated for all the distances. The first
            index relates to the index of the order.
        """
        # Build the non-orthogonal functions first.
        f_nl = (distances < self._r_c) * (
            (
                (
                    self._c_1[:, jnp.newaxis] * self._function(
                        self._roots[:-1, jnp.newaxis] * distances / self._r_c
                    ) - self._c_2[:, jnp.newaxis] * self._function(
                        self._roots[1:, jnp.newaxis] * distances / self._r_c
                    )
                ) / self._r_c**1.5
            )
        )

        # And then orthogonalize them.
        return self._transformation @ f_nl

    @property
    def ell(self) -> int:
        """
        The order of the Bessel function
        """
        return self._order

    @property
    def max_order(self) -> int:
        """
        Maximum radial order.
        """
        return self._n_max

    @property
    def cutoff(self) -> float:
        """
        The cutoff radius
        """
        return self._r_c


class RadialBasis:
    def __init__(self, max_order: int, cutoff: float):
        """
        doi: 10.1063/1.5111045
        """
        self._n_max = max_order
        self._radial_functions = [
            EllChannel(ell, max_order - ell, cutoff)
            for ell in range(self._n_max + 1)
        ]
        self._r_c = cutoff

    def __call__(self, distances: jnp.ndarray) -> List[jnp.ndarray]:
        """
        Parameters
        ----------
        distances: jnp.array[Outer atom]
            The distances between the reference atom and all the other atoms.
            The index corresponds to the atomic index

        Returns
        -------
        radial_value: Radial functions [(max_order + 1) * (max_order + 2) / 2]
        """
        return jnp.concatenate(
            [
                self._radial_functions[ell](distances)
                for ell in range(self._n_max + 1)
            ]
        )

    @property
    def max_order(self) -> int:
        """
        Maximum radial order.
        """
        return self._n_max

    @property
    def cutoff(self) -> float:
        """
        The cutoff radius.
        """
        return self._r_c


def build_Legendre_polynomials(max_l):
    """
    Return a callable that, given cos(theta), computes Pl(cos(theta)), where Pl
    is the l-th Legendre polynomial, for all l <= max_l.
    """
    coeffs = [sp.special.legendre(ell).c for ell in range(max_l + 1)]

    def function(cos_theta: jnp.ndarray) -> List[jnp.ndarray]:
        return jnp.array(
            [jnp.polyval(coeffs[ell], cos_theta) for ell in range(max_l + 1)]
        )

    return function


def center_at_point(
    coordinates: jnp.ndarray,
    reference: jnp.ndarray,
    cell_size: jnp.ndarray = None
):
    delta = coordinates - reference
    if cell_size is not None:
        # This explicit inversion is not pretty to look at, but
        # jnp.linalg.solve occasionally runs into problems. Lattice
        # vector matrices and simulation boxes should not be close
        # to singular, at any rate.
        delta -= jnp.round(delta @ jnp.linalg.inv(cell_size)) @ cell_size
    radius = _sqrt(jnp.sum(delta**2, axis=-1))
    return (delta, radius)


def center_at_atoms(coordinates: jnp.ndarray, cell_size: jnp.ndarray = None):
    delta = coordinates - coordinates[:, jnp.newaxis, :]
    if cell_size is not None:
        delta -= jnp.einsum(
            "ijk,kl",
            jnp.round(jnp.einsum("ijk,kl", delta, jnp.linalg.inv(cell_size))),
            cell_size
        )
    radius = _sqrt(jnp.sum(delta**2, axis=-1))
    return (delta, radius)


@jax.jit
def _get_max_number_of_neighbors(coordinates, cutoff, cell_size=None):
    """
    Return the maximum number of neighbors within a cutoff in a configuration.

    The central atom itself is not counted.

    Parameters
    ----------
    coordinats: An (n_atoms, 3) array of atomic positions.
    cutoff: The maximum distance for two atoms to be considered neighbors.
    cell_size: Unit cell vector matrix (3x3) if the system is periodic.

    Returns
    -------
    The maximum number of atoms within a sphere of radius cutoff around another
    atom in the configuration provided.
    """
    cutoff2 = cutoff * cutoff
    delta = coordinates - coordinates[:, jnp.newaxis, :]
    if cell_size is not None:
        delta -= jnp.einsum(
            "ijk,kl",
            jnp.round(jnp.einsum("ijk,kl", delta, jnp.linalg.inv(cell_size))),
            cell_size
        )
    distances2 = jnp.sum(delta**2, axis=2)
    n_neighbors = (distances2 < cutoff2).sum(axis=1)
    return jnp.squeeze(jnp.maximum(0, n_neighbors.max() - 1))


def get_max_number_of_neighbors(coordinates, cutoff, cell_size=None):
    return int(_get_max_number_of_neighbors(coordinates, cutoff, cell_size))


class PowerSpectrumGenerator:
    """
    Creates a functor that returns all the descriptors of an atom. It must be
    initialized with the maximum n and the number of atom types, as well as the
    maximum allowed number of neighbors. If a calculation involves denser
    environments, the error will be silently ignored. Use
    get_max_number_of_neighbors to check this condition if needed.
    """
    def __init__(
        self, max_order: int, cutoff: float, n_types: int, max_neighbors: int
    ):
        self._n_max = max_order
        self._r_c = cutoff
        self._n_types = n_types
        self._max_neighbors = max_neighbors
        self._radial = RadialBasis(self._n_max, cutoff)
        self._too_many_neighbors = False
        self._angular = build_Legendre_polynomials(self._n_max)
        self._n_coeffs = (self._n_max + 1) * (self._n_max + 2) // 2
        self._inner_shape = (self._n_types, self._n_coeffs)
        self._outer_shape = (self._n_types, self._n_types, self._n_coeffs)
        self._n_desc = (
            self._n_types * (self._n_types + 1) * self._n_coeffs
        ) // 2
        self._triu_indices = jnp.triu_indices(self._n_types)
        # Number of angular descriptors for each l.
        self._degeneracies = jnp.arange(self._n_max + 1, 0, -1)

    def __len__(self):
        return self._n_desc

    def __call__(
        self,
        coordinates: jnp.ndarray,
        a_types: jnp.ndarray,
        cell_size: jnp.ndarray = None
    ) -> jnp.ndarray:
        if not isinstance(coordinates, jnp.ndarray):
            coordinates = jnp.array(coordinates.numpy())
        return self.process_data(coordinates, a_types, cell_size=cell_size)

    def process_data(
        self,
        coordinates: jnp.ndarray,
        a_types: jnp.ndarray,
        cell_size: jnp.ndarray = None
    ) -> jnp.ndarray:
        deltas, radii = center_at_atoms(coordinates, cell_size)
        weights = jax.nn.one_hot(a_types, self._n_types)
        nruter = jax.lax.map(
            jax.checkpoint(
                lambda args: self._process_center(args[0], args[1], weights)
            ), (deltas, radii)
        )
        return nruter

    def process_atom(
        self,
        coordinates: jnp.ndarray,
        a_types: jnp.ndarray,
        index: int,
        cell_size: jnp.ndarray = None
    ) -> jnp.ndarray:
        return self.process_center(
            coordinates, a_types, coordinates[index], cell_size
        )

    def process_center(
        self,
        coordinates: jnp.ndarray,
        a_types: jnp.ndarray,
        center: jnp.ndarray,
        cell_size: jnp.ndarray = None
    ) -> jnp.ndarray:
        deltas, radii = center_at_point(
            coordinates, center, cell_size=cell_size
        )
        weights = jax.nn.one_hot(a_types, self._n_types)
        return self._process_center(deltas, radii, weights)

    def _process_center(
        self, deltas: jnp.ndarray, radii: jnp.ndarray, weights: jnp.ndarray
    ) -> jnp.ndarray:
        inner_shape = self._inner_shape
        outer_shape = self._outer_shape
        prefactors = (2. * jnp.arange(self._n_max + 1.) + 1.) / 4. / jnp.pi

        neighbors = radii.argsort()[1:self._max_neighbors + 1]
        deltas = jnp.take(deltas, neighbors, axis=0)
        radii = jnp.take(radii, neighbors, axis=0)
        weights = jnp.take(weights, neighbors, axis=0)

        # Compute the values of the radial part for all particles.
        all_gs = jnp.atleast_2d(jnp.squeeze(self._radial(radii)).T)
        # Use the one-hot encoding to classify the radial parts according to
        # the type of the second atom.
        all_gs = weights[:, :, jnp.newaxis] * all_gs[:, jnp.newaxis, :]

        # Everything that follows is the equivalent of a double for loop over
        # particles, with short-circuits when a distance is longer than the
        # cutoff. It's expressed in a functional style to keep it JIT-able and
        # differentiable, which makes a difference of orders of magnitude
        # in performance.

        def calc_contribution(args):
            delta, radius, delta_p, radius_p, gs_p = args
            cos_theta = (delta * delta_p).sum() / (radius * radius_p)
            legendre = self._angular(cos_theta)
            kernel = jnp.repeat(prefactors * legendre, self._degeneracies)
            nruter = gs_p * kernel
            return nruter

        def inner_function(delta, radius, delta_p, radius_p, gs_p):
            # yapf: disable
            args = (delta, radius, delta_p, radius_p, gs_p)
            contribution = jax.lax.cond(
                radius_p < self._r_c,
                lambda x: calc_contribution(x),
                lambda _: jnp.zeros(inner_shape),
                args)
            # yapf: enable
            return contribution

        inner_function = jax.vmap(inner_function, in_axes=[None, None, 0, 0, 0])

        @jax.checkpoint
        def outer_function(delta, radius, gs):
            subtotal = jax.lax.cond(
                radius < self._r_c,
                lambda x: gs[:, jnp.newaxis, :] * inner_function(
                    delta, radius, x[0], x[1], x[2]
                ).sum(axis=0)[jnp.newaxis, :, :],
                lambda _: jnp.zeros(outer_shape), (deltas, radii, all_gs)
            )
            return subtotal

        outer_function = jax.vmap(outer_function, in_axes=[0, 0, 0])
        nruter = outer_function(deltas, radii, all_gs).sum(axis=0)

        # Avoid redundancy (and possible problems at a later point) by removing
        # the lower triangle along the "atom type" axes.
        return nruter[self._triu_indices[0], self._triu_indices[1], :]

    @property
    def max_order(self) -> float:
        """
        The maximum radial order of the pipeline
        """
        return self._n_max

    @property
    def cutoff(self) -> float:
        """
        The cutoff distance of the pipeline
        """
        return self._r_c

    @property
    def n_types(self) -> float:
        """
        The number of atom types considered
        """
        return self._n_types

    @property
    def max_neighbors(self) -> int:
        """
        The maximum number of neighbors allowed
        """
        return self._n_types


if __name__ == "__main__":
    import sys
    import time

    import numpy.random
    import scipy as sp
    import matplotlib
    import matplotlib.pyplot as plt

    rng = onp.random.default_rng()

    N_MAX = 4
    R_CUT = 3.7711
    ATOM_TYPES = 4

    REFERENCE = onp.array(
        [
            0.031871, 0.138079, 0.000416, 0.099814, 0.000514, 0.002641,
            0.000580, 0.002311, 0.001099, 0.654837, 0.106035, 0.003456,
            0.005239, 0.367765, 0.483945
        ]
    )

    coords = jnp.array(
        [
            [0.00000, 0.00000, 0.00000], [1.3681827, -1.3103517, -1.3131874],
            [-1.5151760, 1.3360077, -1.3477119],
            [-1.3989598, -1.2973683, 1.3679189],
            [1.2279369, 1.3400378, 1.4797429]
        ]
    )

    generator = PowerSpectrumGenerator(
        N_MAX, R_CUT, ATOM_TYPES, coords.shape[0]
    )

    DELTA = 1e-3
    coords_plus = jnp.array(
        [
            [0.00000 + DELTA, 0.00000, 0.00000],
            [1.3681827, -1.3103517, -1.3131874],
            [-1.5151760, 1.3360077, -1.3477119],
            [-1.3989598, -1.2973683, 1.3679189],
            [1.2279369, 1.3400378, 1.4797429]
        ]
    )
    coords_minus = jnp.array(
        [
            [0.00000 - DELTA, 0.00000, 0.00000],
            [1.3681827, -1.3103517, -1.3131874],
            [-1.5151760, 1.3360077, -1.3477119],
            [-1.3989598, -1.2973683, 1.3679189],
            [1.2279369, 1.3400378, 1.4797429]
        ]
    )

    atom_types = [0] * coords.shape[0]
    processor = jax.jit(
        lambda x: generator.process_atom(x, atom_types, 0)[0, :]
    )
    descriptors = processor(coords)

    random_vector = rng.uniform(-1., 1., descriptors.shape)

    @jax.curry
    @jax.jit
    def vjp_descriptors(x, v):
        vjp = jax.jit(jax.vjp(processor, x)[1])
        return vjp(v)

    vjp_at_coords = vjp_descriptors(coords)

    # The reference and our calculation put the descriptors in a different
    # order. Try to find a permutation to make them match.
    permutation = [onp.fabs(i - REFERENCE).argmin() for i in descriptors]
    if not onp.all(sorted(permutation) == onp.arange(len(REFERENCE))):
        print(
            "ERROR: Couldn't find a suitable permutation of the reference"
            " values to match the results"
        )

    REFERENCE = onp.array([REFERENCE[i] for i in permutation])
    print("Reference:", REFERENCE, sep="\n")
    print("Descriptors:", descriptors, sep="\n")
    print("Norm of the difference:", la.norm(REFERENCE - descriptors))

    full_processor = jax.jit(lambda x: generator.process_data(x, atom_types))
    part_processor = jax.jit(
        lambda x, i: generator.process_atom(x, atom_types, i)
    )
    full = full_processor(coords)

    for i_atom in range(coords.shape[0]):
        part = part_processor(coords, i_atom)
        print(
            f"process_data vs. process_atom, atom #{i_atom + 1}",
            la.norm(part - full[i_atom])
        )

    jacobian_func = jax.jit(jax.jacfwd(processor))
    jacobian = jacobian_func(coords)
    print("Jacobian:", jacobian[..., 0, 0], sep="\n")
    num_jacobian = (processor(coords_plus) -
                    processor(coords_minus)) / (2. * DELTA)
    print("Numerical Jacobian:", num_jacobian, sep="\n")

    print("Forces from direct VJP:")
    print(onp.array(vjp_at_coords(random_vector)[0]))
    print("Forces from Jacobian matrix:")
    print(onp.einsum("i,ijk", random_vector, jacobian))

    # Prepare everything for a different number of particles
    NPART = 200
    NITER = 10
    SIDE = 15.0
    N_MAX_NEW = N_MAX
    coords_shape = (NPART, 3)

    probe_coords = rng.uniform(0., SIDE, coords_shape)
    random_types = rng.integers(0, ATOM_TYPES, NPART)
    max_neighbors = max(get_max_number_of_neighbors(probe_coords, R_CUT), 1)
    generator = PowerSpectrumGenerator(
        N_MAX_NEW, R_CUT, ATOM_TYPES, max_neighbors
    )
    descriptors = generator.process_data(probe_coords, random_types)
    descriptors_shape = descriptors.shape

    print(
        f"Benchmark ({NPART} particles, {NITER} iterations,"
        f" side length {SIDE}):"
    )
    random_coordinates = []
    random_vectors = []
    max_neighbors = 0
    for i in range(NITER):
        random_coordinates.append(rng.uniform(0., SIDE, coords_shape))
        random_vectors.append(rng.uniform(0., SIDE, descriptors_shape))
        max_neighbors = max(
            max_neighbors,
            get_max_number_of_neighbors(random_coordinates[i], R_CUT)
        )
    print("Maximum number of neighbors considered:", max_neighbors)

    generator = PowerSpectrumGenerator(
        N_MAX_NEW, R_CUT, ATOM_TYPES, max_neighbors
    )

    generator.process_data = jax.jit(generator.process_data)
    descriptors = generator.process_data(probe_coords, random_types)
    descriptors_shape = descriptors.shape

    @jax.jit
    def vjp_generator(x, v):
        vjp = jax.vjp(lambda c: generator.process_data(c, random_types), x)[1]
        return vjp(v)[0]

    vjp_generator(probe_coords, descriptors)

    print(" - Descriptor calculation:")
    start = time.perf_counter()
    for i in range(NITER):
        descriptors = onp.array(
            generator.process_data(random_coordinates[i], random_types)
        )
    end = time.perf_counter()
    print(f"{(end - start) / NITER} seconds per iteration")

    results_vjp = []
    print(" - VJP test:")
    start = time.perf_counter()
    for i in range(NITER):
        results_vjp.append(
            onp.array(vjp_generator(random_coordinates[i], random_vectors[i]))
        )
    end = time.perf_counter()
    print(f"{(end - start) / NITER} seconds per iteration")

    # Test whether the cutoff is affecting the results.
    overkill_generator = PowerSpectrumGenerator(
        N_MAX_NEW, R_CUT, ATOM_TYPES, NPART
    )
    overkill_generator.process_data = jax.jit(overkill_generator.process_data)

    @jax.jit
    def overkill_vjp_generator(x, v):
        vjp = jax.vjp(
            lambda c: overkill_generator.process_data(c, random_types), x
        )[1]
        return vjp(v)[0]

    overkill_generator.process_data(probe_coords, random_types)
    overkill_vjp_generator(probe_coords, descriptors)
    for i in range(NITER):
        descriptors = onp.array(
            generator.process_data(random_coordinates[i], random_types)
        )
        overkill_descriptors = onp.array(
            overkill_generator.process_data(
                random_coordinates[i], random_types
            )
        )
        print(
            f"Effect of neighbor list on descriptors (#{i + 1})",
            la.norm(descriptors - overkill_descriptors)
        )
        print(f"\t- Norm of the descriptors:", la.norm(overkill_descriptors))
        vjp = vjp_generator(random_coordinates[i], random_vectors[i])
        overkill_vjp = overkill_vjp_generator(
            random_coordinates[i], random_vectors[i]
        )
        print(
            f"Effect of neighbor list on VJP (#{i + 1})",
            la.norm(vjp - overkill_vjp)
        )
        print(f"\t- Norm of the VJP:", la.norm(overkill_vjp))
