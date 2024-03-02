# Copyright 2019-2021 The NeuralIL contributors
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

from typing import Callable
from typing import Sequence

import jax
import jax.nn
import jax.numpy as jnp
import flax.linen
from neuralil.neural_network.bessel_descriptors import center_at_atoms
import itertools
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Sequence

def pairwise(iterable):
    """Reimplementation of Python 3.10's itertools.pairwise."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


@jax.custom_jvp
def _sqrt(x):
    return jnp.sqrt(x)


@_sqrt.defjvp
def _sqrt_jvp(primals, tangents):
    (x,) = primals
    (xdot,) = tangents
    primal_out = _sqrt(x)
    tangent_out = jnp.where(x == 0.0, 0.0, 0.5 / primal_out) * xdot
    return (primal_out, tangent_out)


def _aux_function_f(t):
    "First auxiliary function used in the definition of the smooth bump."
    return jnp.where(t > 0.0, jnp.exp(-1.0 / jnp.where(t > 0.0, t, 1.0)), 0.0)


def _aux_function_g(t):
    "Second auxiliary function used in the definition of the smooth bump."
    f_of_t = _aux_function_f(t)
    return f_of_t / (f_of_t + _aux_function_f(1.0 - t))


def smooth_cutoff(r, r_switch, r_cut):
    """One-dimensional smooth cutoff function based on a smooth bump.

    Args:
        r: The radii at which the function must be evaluated.
        r_switch: The radius at which the function starts differing from 1.
        r_cut: The radius at which the function becomes exactly 0.
    """
    r_switch2 = r_switch * r_switch
    r_cut2 = r_cut * r_cut

    return 1.0 - _aux_function_g((r * r - r_switch2) / (r_cut2 - r_switch2))


def calc_morse_mixing_radii(radii, abd_probe, abd_source):
    """Compute the mixing radii between single-species Morse potentials.

    This radii are computed based on the repulsive part alone, as the solution
    of the equations

    r1 + r2 = radii
    phi_1'(2 * r1) = phi_2'(2 * r2)

    where phi' is the derivative of d * exp(-2 * a *(r - b)).
    Reference: J. Chem. Phys. 59 (1973) 2464.

    The function is intended to be used for two sets of atoms, all at once. We
    call the first set of atoms "probe" and the second one "source".

    Args:
        radii: The distances between atoms as an (n_probe, n_source) array.
        abd_probe: An (n_probe, 3) vector of Morse parameters, sorted as
            (a, b, d) in the notation of the equation above.
        abd_source: An (n_source, 3) vector of Morse parameters, sorted as
            (a, b, d) in the notation of the equation above.

    Returns:
        An (n_probe, n_source) array with the result.
    """
    a_probe = abd_probe[:, 0][:, jnp.newaxis]
    b_probe = abd_probe[:, 1][:, jnp.newaxis]
    d_probe = abd_probe[:, 2][:, jnp.newaxis]
    a_source = abd_source[:, 0][jnp.newaxis, :]
    b_source = abd_source[:, 1][jnp.newaxis, :]
    d_source = abd_source[:, 2][jnp.newaxis, :]
    return (
        a_source * radii
        + 0.25 * jnp.log(a_probe * d_probe / (a_source * d_source))
        + 0.5 * (a_probe * b_probe - a_source * b_source)
    ) / (a_probe + a_source)

class ResNetIdentity(flax.linen.Module):
    """Identity element of a regression deep residual network.

    Reference: Chen, D.; Hu, F.; Nian, G.; Yang, T.
    Deep Residual Learning for Nonlinear Regression. Entropy 22 (2020) 193.

    Args:
        width: The number of elements of the input and output.
        activation_function: The nonlinear activation function for each neuron,
            which is Swish by default.
        kernel_init: Initializer for the weight matrices (default:
            flax.linen.initializers.lecun_normal).
    """

    width: int
    activation_function: Callable = flax.linen.swish
    kernel_init: Callable = jax.nn.initializers.lecun_normal()

    @flax.linen.compact
    def __call__(self, input_signals):
        result_long = self.activation_function(
            flax.linen.LayerNorm()(
                flax.linen.Dense(
                    self.width,
                    kernel_init=self.kernel_init,
                )(input_signals)
            )
        )
        result_long = self.activation_function(
            flax.linen.LayerNorm()(
                flax.linen.Dense(
                    self.width,
                    kernel_init=self.kernel_init,
                )(result_long)
            )
        )
        result_long = flax.linen.LayerNorm()(
            flax.linen.Dense(
                self.width,
                kernel_init=self.kernel_init,
            )(result_long)
        )
        return self.activation_function(result_long + input_signals)


class ResNetDense(flax.linen.Module):
    """Dense element of a regression deep residual network.

    Reference: Chen, D.; Hu, F.; Nian, G.; Yang, T.
    Deep Residual Learning for Nonlinear Regression. Entropy 22 (2020) 193.

    Args:
        input_width: The number of elements of the input.
        output_width: The number of elements of the output.
        activation_function: The nonlinear activation function for each neuron,
            which is Swish by default.
        kernel_init: Initializer for the weight matrices (default:
            flax.linen.initializers.lecun_normal).
    """

    input_width: int
    output_width: int
    activation_function: Callable = flax.linen.swish
    kernel_init: Callable = jax.nn.initializers.lecun_normal()

    @flax.linen.compact
    def __call__(self, input_signals):
        result_long = self.activation_function(
            flax.linen.LayerNorm()(
                flax.linen.Dense(
                    self.input_width,
                    kernel_init=self.kernel_init,
                )(input_signals)
            )
        )
        result_long = self.activation_function(
            flax.linen.LayerNorm()(
                flax.linen.Dense(
                    self.input_width,
                    kernel_init=self.kernel_init,
                )(result_long)
            )
        )
        result_long = flax.linen.Dense(
            self.output_width,
            kernel_init=self.kernel_init,
        )(result_long)
        result_short = flax.linen.Dense(
            self.output_width,
            kernel_init=self.kernel_init,
        )(input_signals)
        # Skip the last layer normalization for the outlet, where it would
        # destroy the results.
        if self.output_width > 1:
            result_long = flax.linen.LayerNorm()(result_long)
            result_short = flax.linen.LayerNorm()(result_short)

        return self.activation_function(result_long + result_short)


class ResNetCore(flax.linen.Module):
    """Alternative to Core based on ResNet (deep network with bypasses).

    This model takes the descriptors of each atom (Bessel + embedding,
    concatenated or otherwise combined) as inputs and calculates that atom's
    contribution to the potential energy.
    Reference: Chen, D.; Hu, F.; Nian, G.; Yang, T.
    Deep Residual Learning for Nonlinear Regression. Entropy 22 (2020) 193.
    Compared to the basic Core, width-preserving layers are replaced with
    ResNetIdentity, while other layers are replaced with ResNetDense.

    Args:
        layer_widths: The sequence of layer widths, excluding the output
            layer, which always has a width equal to one.
        activation_function: The nonlinear activation function for each neuron,
            which is Swish by default.
        kernel_init: Initializer for the weight matrices (default:
            flax.linen.initializers.lecun_normal).
    """

    layer_widths: Sequence[int]
    activation_function: Callable = flax.linen.swish
    kernel_init: Callable = jax.nn.initializers.lecun_normal()

    def setup(self):
        total_widths = self.layer_widths + (1,)
        identity_counter = 0
        dense_counter = 0
        res_layers = []
        for i_width, o_width in pairwise(total_widths):
            if i_width == o_width:
                identity_counter += 1
                name = f"ResNetIdentity_{identity_counter}_{i_width}"
                res_layers.append(
                    ResNetIdentity(
                        i_width,
                        self.activation_function,
                        self.kernel_init,
                        name=name,
                    )
                )
            else:
                dense_counter += 1
                name = f"ResNetDense_{dense_counter}_{i_width}_to_{o_width}"
                res_layers.append(
                    ResNetDense(
                        i_width,
                        o_width,
                        self.activation_function,
                        self.kernel_init,
                        name=name,
                    )
                )
        self.res_layers = res_layers

    def __call__(self, descriptors):
        result = descriptors
        for layer in self.res_layers:
            result = layer(result)
        return result

class CoreRelu(flax.linen.Module):

    layer_widths: Sequence[int]
    activation_function: Callable = flax.linen.relu

    @flax.linen.compact
    def __call__(self, descriptors):
        result = self.activation_function(
            flax.linen.Dense(self.layer_widths[0])(descriptors))
        for w in self.layer_widths[1:]:
            result = self.activation_function(flax.linen.LayerNorm()(
                flax.linen.Dense(w, use_bias=False)(result)))
        return self.activation_function(flax.linen.Dense(1)(result))

class ReluModel(flax.linen.Module):

    n_types: int
    embed_d: int
    descriptor_generator: Callable
    corerelu_model: flax.linen.Module
    mixer: Callable = lambda d, e: jnp.concatenate(
        [d.reshape((d.shape[0], -1)), e], axis=1)

    def setup(self):
        self.embed = flax.linen.Embed(self.n_types, self.embed_d)
        self.denormalizer = flax.linen.Dense(1)
        self._gradient_calculator = jax.checkpoint(
            jax.grad(self.calc_potential_energy, argnums=0))

    def calc_potential_energy_from_descriptors(self, combined_inputs):
        contributions = self.corerelu_model(combined_inputs)
        contributions = self.denormalizer(contributions)
        return jnp.squeeze(contributions.sum(axis=0))

    def calc_atomic_energies(self, positions, types, cell):
        descriptors = self.descriptor_generator(positions, types, cell)
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        results = self.corerelu_model(combined_inputs)
        results = self.denormalizer(results)
        return results

    def calc_potential_energy(self, positions, types, cell):
        contributions = self.calc_atomic_energies(positions, types, cell)
        return jnp.squeeze(contributions.sum(axis=0))

    def calc_forces(self, positions, types, cell):
        return -self._gradient_calculator(positions, types, cell)


class ReluElectrostaticModel(flax.linen.Module):
    """Implementation of the charge equilibration scheme in [1], 
    based on atomic electronegativities predicted by a neural network
    with a similar architecture as DynamicsModel. The widths of the
    charge densities (sigmas) should be provided and are typically
    taken to be the covalent radii. The atomic hardness is a 
    learnable parameter.
    
    References:
    [1] Rappe, A. K. & Goddard, W. A. J. Phys. Chem. 95, 3358 (1991).

    Args:
        core_model: The model that takes all the descriptors and returns an
            atomic electronegativities
        n_types: Number of atomic types in the system
        sigmas: Sequence of widths of the element-specific Gaussian 
            charge densities 
    """
    corerelu_model: Sequence[int]
    n_types: int
    sigmas: Sequence[float]

    def setup(self):
        self.denormalizer = flax.linen.Dense(1)
        self.hardness = self.param('hardness', flax.linen.initializers.ones,
                                   (self.n_types, ))

    def calc_electronegativities_from_descriptors(self, combined_inputs):
        """Compute the electronegativities from mixed descriptors

        Args:
            combined_inputs: mixed descriptors

        Returns:
            The n_atoms electronegativities
        """
        results = self.corerelu_model(combined_inputs)
        results = self.denormalizer(results)
        return jnp.squeeze(results)

    def _get_charges(self, combined_inputs, positions, types, cell):
        electronegativities = self.calc_electronegativities_from_descriptors(
            combined_inputs)
        sigmas = self.sigmas
        sigmas_expanded = jnp.take(jnp.array(sigmas), types)

        hardness_expanded = jnp.take(self.hardness, types)

        sq_sigmas = jnp.squeeze(sigmas_expanded * sigmas_expanded)
        gammas = jnp.sqrt(sq_sigmas[:, jnp.newaxis] + sq_sigmas)

        _, r_ij = center_at_atoms(positions, cell)

        diag = (hardness_expanded + 1. / (sigmas_expanded * jnp.sqrt(jnp.pi)))

        offdiag_ij = (jax.lax.erf(r_ij / (jnp.sqrt(2) * gammas)) /(jnp.where(r_ij == 0, 1, r_ij)))
        a_ij = jnp.diag(diag) + offdiag_ij

        a_ij = jnp.vstack((a_ij, jnp.ones(a_ij.shape[0])))
        a_ij = jnp.hstack((a_ij, [[1]] * a_ij.shape[1] + [[0]]))
        electronegativities_sol = jnp.append(electronegativities, 0)

        charges = jax.scipy.linalg.solve(a_ij,
                                         (-1 * electronegativities_sol))[:-1]
        return offdiag_ij, charges

    def get_charges(self, combined_inputs, positions, types, cell):
        # Passthrough function to access charges
        return self._get_charges(combined_inputs, positions, types, cell)[1]

    @flax.linen.compact
    def __call__(self, combined_inputs, positions, types, cell):
        """Compute electrostatic contributions to potential energy

        Args:
            combined_inputs : Mixed descriptors
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The n_atoms electrostatic contributions to the energy
        """
        sigmas = self.sigmas
        sigmas_expanded = jnp.take(jnp.array(sigmas), types)
        offdiag, charges = self._get_charges(combined_inputs, positions, types, cell)

        self_contribution = ((charges * charges) /
                             (2 * sigmas_expanded * jnp.sqrt(jnp.pi))).sum()
        pair_contribution = jnp.dot(jnp.dot(offdiag, charges), charges) / 2.
        return jnp.squeeze(self_contribution + pair_contribution)


class ReluDynamicsModelWithCharges(flax.linen.Module):
    """Wrapper model around the core layers and the electrostatic model
     to calculate energies and forces.

    The class does not provide a __call__ method, forcing the user to choose
    what to evaluate (forces, energies or both).

    Args:
        n_types: The number of atomic types in the system.
        embed_d: The dimension of the embedding vector to be mixed with the
            descriptors.
        descriptor_generator: The function mapping the atomic coordinates,
            types and cell size to descriptors.
        core_model: The model that takes all the descriptors and returns an
            atomic contribution to the energy.
        electrostatic_model: The model that takes all the descriptors and returns
            atomic electronegativities
        mixer: The function that takes the Bessel descriptors and the
            embedding vectors and creates the input descriptors for the core
            model. The default choice just concatenates them.
    """
    n_types: int
    embed_d: int
    descriptor_generator: Callable
    corerelu_model: flax.linen.Module
    reluelectrostatic_model: flax.linen.Module
    mixer: Callable = lambda d, e: jnp.concatenate(
        [d.reshape((d.shape[0], -1)), e], axis=1)

    def setup(self):
        # These neurons create the embedding vector.
        self.embed = flax.linen.Embed(self.n_types, self.embed_d)
        # This linear layer centers and scales the energy after the core
        # has done its job.
        self.denormalizer = flax.linen.Dense(1)
        # The checkpointing strategy can be reconsidered to achieve different
        # tradeoffs between memory and CPU usage.
        self._gradient_calculator = jax.checkpoint(
            jax.grad(self.calc_potential_energy, argnums=0))

    def calc_atomic_energies(self, positions, types, cell):
        """Compute the atomic contributions to the potential energy.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The n_atoms contributions to the energy.
        """
        descriptors = self.descriptor_generator(positions, types, cell)
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        results = self.corerelu_model(combined_inputs)
        results = self.denormalizer(results)
        electrostatic_energy = self.reluelectrostatic_model(
            combined_inputs, positions, types, cell)
        return results + electrostatic_energy

    def calc_potential_energy(self, positions, types, cell):
        """Compute the total potential energy of the system.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The sum of all atomic contributions to the potential energy.
        """
        contributions = self.calc_atomic_energies(positions, types, cell)
        return jnp.squeeze(contributions.sum(axis=0))

    def calc_forces(self, positions, types, cell):
        """Compute the force on each atom.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The (n_atoms, 3) vector containing all the forces.
        """
        return -self._gradient_calculator(positions, types, cell)

    def calc_charges(self, positions, types, cell):
        """Computes atomic charges 
        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The n_atoms vector containing all the charges.
        """
        descriptors = self.descriptor_generator(positions, types, cell)
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        charges = self.reluelectrostatic_model.get_charges(combined_inputs,
                                                       positions, types, cell)
        return charges

    def calc_electronegativities(self, positions, types, cell):
        """Computes atomic electronegativities

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The n_atoms vector containing all the electronegativities.
        """
        descriptors = self.descriptor_generator(positions, types, cell)
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        electronegativities = (
            self.reluelectrostatic_model.calc_electronegativities_from_descriptors(
                combined_inputs))
        return electronegativities




class ReluModel(flax.linen.Module):

    n_types: int
    embed_d: int
    descriptor_generator: Callable
    corerelu_model: flax.linen.Module
    mixer: Callable = lambda d, e: jnp.concatenate(
        [d.reshape((d.shape[0], -1)), e], axis=1)

    def setup(self):
        self.embed = flax.linen.Embed(self.n_types, self.embed_d)
        self.denormalizer = flax.linen.Dense(1)
        self._gradient_calculator = jax.checkpoint(
            jax.grad(self.calc_potential_energy, argnums=0))

    def calc_potential_energy_from_descriptors(self, combined_inputs):
        contributions = self.corerelu_model(combined_inputs)
        contributions = self.denormalizer(contributions)
        return jnp.squeeze(contributions.sum(axis=0))

    def calc_atomic_energies(self, positions, types, cell):
        descriptors = self.descriptor_generator(positions, types, cell)
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        results = self.corerelu_model(combined_inputs)
        results = self.denormalizer(results)
        return results

    def calc_potential_energy(self, positions, types, cell):
        contributions = self.calc_atomic_energies(positions, types, cell)
        return jnp.squeeze(contributions.sum(axis=0))

    def calc_forces(self, positions, types, cell):
        return -self._gradient_calculator(positions, types, cell)

class ReluElectrostaticModel(flax.linen.Module):
    """Implementation of the charge equilibration scheme in [1], 
    based on atomic electronegativities predicted by a neural network
    with a similar architecture as DynamicsModel. The widths of the
    charge densities (sigmas) should be provided and are typically
    taken to be the covalent radii. The atomic hardness is a 
    learnable parameter.
    
    References:
    [1] Rappe, A. K. & Goddard, W. A. J. Phys. Chem. 95, 3358 (1991).

    Args:
        core_model: The model that takes all the descriptors and returns an
            atomic electronegativities
        n_types: Number of atomic types in the system
        sigmas: Sequence of widths of the element-specific Gaussian 
            charge densities 
    """
    corerelu_model: Sequence[int]
    n_types: int
    sigmas: Sequence[float]

    def setup(self):
        self.denormalizer = flax.linen.Dense(1)
        self.hardness = self.param('hardness', flax.linen.initializers.ones,
                                   (self.n_types, ))

    def calc_electronegativities_from_descriptors(self, combined_inputs):
        """Compute the electronegativities from mixed descriptors

        Args:
            combined_inputs: mixed descriptors

        Returns:
            The n_atoms electronegativities
        """
        results = self.corerelu_model(combined_inputs)
        results = self.denormalizer(results)
        return jnp.squeeze(results)

    def _get_charges(self, combined_inputs, positions, types, cell):
        electronegativities = self.calc_electronegativities_from_descriptors(
            combined_inputs)
        sigmas = self.sigmas
        sigmas_expanded = jnp.take(jnp.array(sigmas), types)

        hardness_expanded = jnp.take(self.hardness, types)

        sq_sigmas = jnp.squeeze(sigmas_expanded * sigmas_expanded)
        gammas = jnp.sqrt(sq_sigmas[:, jnp.newaxis] + sq_sigmas)

        _, r_ij = center_at_atoms(positions, cell)

        diag = (hardness_expanded + 1. / (sigmas_expanded * jnp.sqrt(jnp.pi)))

        offdiag_ij = (jax.lax.erf(r_ij / (jnp.sqrt(2) * gammas)) /(jnp.where(r_ij == 0, 1, r_ij)))
        a_ij = jnp.diag(diag) + offdiag_ij

        a_ij = jnp.vstack((a_ij, jnp.ones(a_ij.shape[0])))
        a_ij = jnp.hstack((a_ij, [[1]] * a_ij.shape[1] + [[0]]))
        electronegativities_sol = jnp.append(electronegativities, 0)

        charges = jax.scipy.linalg.solve(a_ij,
                                         (-1 * electronegativities_sol))[:-1]
        return offdiag_ij, charges

    def get_charges(self, combined_inputs, positions, types, cell):
        # Passthrough function to access charges
        return self._get_charges(combined_inputs, positions, types, cell)[1]

    @flax.linen.compact
    def __call__(self, combined_inputs, positions, types, cell):
        """Compute electrostatic contributions to potential energy

        Args:
            combined_inputs : Mixed descriptors
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The n_atoms electrostatic contributions to the energy
        """
        sigmas = self.sigmas
        sigmas_expanded = jnp.take(jnp.array(sigmas), types)
        offdiag, charges = self._get_charges(combined_inputs, positions, types, cell)

        self_contribution = ((charges * charges) /
                             (2 * sigmas_expanded * jnp.sqrt(jnp.pi))).sum()
        pair_contribution = jnp.dot(jnp.dot(offdiag, charges), charges) / 2.
        return jnp.squeeze(self_contribution + pair_contribution)


class ReluDynamicsModelWithCharges(flax.linen.Module):
    """Wrapper model around the core layers and the electrostatic model
     to calculate energies and forces.

    The class does not provide a __call__ method, forcing the user to choose
    what to evaluate (forces, energies or both).

    Args:
        n_types: The number of atomic types in the system.
        embed_d: The dimension of the embedding vector to be mixed with the
            descriptors.
        descriptor_generator: The function mapping the atomic coordinates,
            types and cell size to descriptors.
        core_model: The model that takes all the descriptors and returns an
            atomic contribution to the energy.
        electrostatic_model: The model that takes all the descriptors and returns
            atomic electronegativities
        mixer: The function that takes the Bessel descriptors and the
            embedding vectors and creates the input descriptors for the core
            model. The default choice just concatenates them.
    """
    n_types: int
    embed_d: int
    descriptor_generator: Callable
    corerelu_model: flax.linen.Module
    reluelectrostatic_model: flax.linen.Module
    mixer: Callable = lambda d, e: jnp.concatenate(
        [d.reshape((d.shape[0], -1)), e], axis=1)

    def setup(self):
        # These neurons create the embedding vector.
        self.embed = flax.linen.Embed(self.n_types, self.embed_d)
        # This linear layer centers and scales the energy after the core
        # has done its job.
        self.denormalizer = flax.linen.Dense(1)
        # The checkpointing strategy can be reconsidered to achieve different
        # tradeoffs between memory and CPU usage.
        self._gradient_calculator = jax.checkpoint(
            jax.grad(self.calc_potential_energy, argnums=0))

    def calc_atomic_energies(self, positions, types, cell):
        """Compute the atomic contributions to the potential energy.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The n_atoms contributions to the energy.
        """
        descriptors = self.descriptor_generator(positions, types, cell)
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        results = self.corerelu_model(combined_inputs)
        results = self.denormalizer(results)
        electrostatic_energy = self.reluelectrostatic_model(
            combined_inputs, positions, types, cell)
        return results + electrostatic_energy

    def calc_potential_energy(self, positions, types, cell):
        """Compute the total potential energy of the system.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The sum of all atomic contributions to the potential energy.
        """
        contributions = self.calc_atomic_energies(positions, types, cell)
        return jnp.squeeze(contributions.sum(axis=0))

    def calc_forces(self, positions, types, cell):
        """Compute the force on each atom.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The (n_atoms, 3) vector containing all the forces.
        """
        return -self._gradient_calculator(positions, types, cell)

    def calc_charges(self, positions, types, cell):
        """Computes atomic charges 
        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The n_atoms vector containing all the charges.
        """
        descriptors = self.descriptor_generator(positions, types, cell)
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        charges = self.reluelectrostatic_model.get_charges(combined_inputs,
                                                       positions, types, cell)
        return charges

    def calc_electronegativities(self, positions, types, cell):
        """Computes atomic electronegativities

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The n_atoms vector containing all the electronegativities.
        """
        descriptors = self.descriptor_generator(positions, types, cell)
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        electronegativities = (
            self.reluelectrostatic_model.calc_electronegativities_from_descriptors(
                combined_inputs))
        return electronegativities


class BasicMLP(flax.linen.Module):
    layer_widths: Sequence[int]
    activation_function: Callable = flax.linen.swish

    @flax.linen.compact
    def __call__(self, descriptors):
        result = self.activation_function(
            flax.linen.Dense(self.layer_widths[0])(descriptors))
        for w in self.layer_widths[1:]:
            result = self.activation_function(flax.linen.LayerNorm()(
                flax.linen.Dense(w, use_bias=False)(result)))
        return result


class Core(flax.linen.Module):
    """Multilayer perceptron with LayerNorm lying at the core of the model.

    This model takes the descriptors of each atom (Bessel + embedding,
    concatenated or otherwise combined) as inputs and calculates that atom's
    contribution to the potential energy. LayerNorm is applied at each layer
    except the first and the last ones.

    Args:
        layer_widths: The sequence of layer widths, excluding the output
            layer, which always has a width equal to one.
        activation_function: The nonlinear activation function for each neuron,
            which is Swish by default.
    """
    layer_widths: Sequence[int]
    activation_function: Callable = flax.linen.swish

    @flax.linen.compact
    def __call__(self, descriptors):
        result = self.activation_function(
            flax.linen.Dense(self.layer_widths[0])(descriptors))
        for w in self.layer_widths[1:]:
            result = self.activation_function(flax.linen.LayerNorm()(
                flax.linen.Dense(w, use_bias=False)(result)))
        return self.activation_function(flax.linen.Dense(1)(result))




class NeuralILCore(flax.linen.Module):
    """Legacy core architecture used in NeuralIL.

    Just like Core, this model takes the descriptors of each atom (Bessel +
    embedding, concatenated or otherwise combined) as inputs and calculates
    that atom's contribution to the potential energy. However, it does not apply
    any normalization but instead uses the SELU self-normalizing activation
    function. For historical reasons, it also appends an extra Swish layer
    with a width of one.

    Args:
        layer_widths: The sequence of layer widths, excluding the two output
            layer, which always have a width equal to one.
    """
    layer_widths: Sequence[int]

    @flax.linen.compact
    def __call__(self, descriptors):
        result = descriptors
        for w in self.layer_widths:
            result = jax.nn.selu(flax.linen.Dense(w)(result))
        result = jax.nn.selu(flax.linen.Dense(1)(result))
        result = flax.linen.swish(flax.linen.Dense(1)(result))
        return result


class DynamicsModel(flax.linen.Module):
    """Wrapper model around the core layers to calculate energies and forces.

    The class does not provide a __call__ method, forcing the user to choose
    what to evaluate (forces, energies or both).

    Args:
        n_types: The number of atomic types in the system.
        embed_d: The dimension of the embedding vector to be mixed with the
            descriptors.
        descriptor_generator: The function mapping the atomic coordinates,
            types and cell size to descriptors.
        core_model: The model that takes all the descriptors and returns an
            atomic contribution to the energy.
        mixer: The function that takes the Bessel descriptors and the
            embedding vectors and creates the input descriptors for the core
            model. The default choice just concatenates them.
    """
    n_types: int
    embed_d: int
    descriptor_generator: Callable
    core_model: flax.linen.Module
    mixer: Callable = lambda d, e: jnp.concatenate(
        [d.reshape((d.shape[0], -1)), e], axis=1)

    def setup(self):
        # These neurons create the embedding vector.
        self.embed = flax.linen.Embed(self.n_types, self.embed_d)
        # This linear layer centers and scales the energy after the core
        # has done its job.
        self.denormalizer = flax.linen.Dense(1)
        # The checkpointing strategy can be reconsidered to achieve different
        # tradeoffs between memory and CPU usage.
        self._gradient_calculator = jax.grad(self.calc_potential_energy, argnums=0)

    def calc_potential_energy_from_descriptors(self, combined_inputs):
        contributions = self.core_model(combined_inputs)
        contributions = self.denormalizer(contributions)
        return jnp.squeeze(contributions.sum(axis=0))

    def calc_atomic_energies(self, positions, types, cell):
        """Compute the atomic contributions to the potential energy.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The n_atoms contributions to the energy.
        """
        descriptors = self.descriptor_generator(positions, types, cell)
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        results = self.core_model(combined_inputs)
        results = self.denormalizer(results)
        return results

    def calc_potential_energy(self, positions, types, cell):
        """Compute the total potential energy of the system.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The sum of all atomic contributions to the potential energy.
        """
        contributions = self.calc_atomic_energies(positions, types, cell)
        return jnp.squeeze(contributions.sum(axis=0))

    def calc_forces(self, positions, types, cell):
        """Compute the force on each atom.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The (n_atoms, 3) vector containing all the forces.
        """
        return -self._gradient_calculator(positions, types, cell)



class ElectrostaticModel(flax.linen.Module):
    """Implementation of the charge equilibration scheme in [1], 
    based on atomic electronegativities predicted by a neural network
    with a similar architecture as DynamicsModel. The widths of the
    charge densities (sigmas) should be provided and are typically
    taken to be the covalent radii. The atomic hardness is a 
    learnable parameter.
    
    References:
    [1] Rappe, A. K. & Goddard, W. A. J. Phys. Chem. 95, 3358 (1991).

    Args:
        core_model: The model that takes all the descriptors and returns an
            atomic electronegativities
        n_types: Number of atomic types in the system
        sigmas: Sequence of widths of the element-specific Gaussian 
            charge densities 
    """
    core_model: Sequence[int]
    n_types: int
    sigmas: Sequence[float]

    def setup(self):
        self.denormalizer = flax.linen.Dense(1)
        self.hardness = self.param('hardness', flax.linen.initializers.ones,
                                   (self.n_types, ))

    def calc_electronegativities_from_descriptors(self, combined_inputs):
        """Compute the electronegativities from mixed descriptors

        Args:
            combined_inputs: mixed descriptors

        Returns:
            The n_atoms electronegativities
        """
        results = self.core_model(combined_inputs)
        results = self.denormalizer(results)
        return jnp.squeeze(results)

    def _get_charges(self, combined_inputs, positions, types, cell):
        electronegativities = self.calc_electronegativities_from_descriptors(
            combined_inputs)
        sigmas = self.sigmas
        sigmas_expanded = jnp.take(jnp.array(sigmas), types)

        hardness_expanded = jnp.take(self.hardness, types)

        sq_sigmas = jnp.squeeze(sigmas_expanded * sigmas_expanded)
        gammas = jnp.sqrt(sq_sigmas[:, jnp.newaxis] + sq_sigmas)

        _, r_ij = center_at_atoms(positions, cell)

        diag = (hardness_expanded + 1. / (sigmas_expanded * jnp.sqrt(jnp.pi)))

        offdiag_ij = (jax.lax.erf(r_ij / (jnp.sqrt(2) * gammas)) /
                      (jnp.where(r_ij == 0, 1, r_ij)))

        a_ij = jnp.diag(diag) + offdiag_ij

        a_ij = jnp.vstack((a_ij, jnp.ones(a_ij.shape[0])))
        a_ij = jnp.hstack((a_ij, [[1]] * a_ij.shape[1] + [[0]]))
        electronegativities_sol = jnp.append(electronegativities, 0)

        charges = jax.scipy.linalg.solve(a_ij,
                                         (-1 * electronegativities_sol))[:-1]
        return offdiag_ij, charges

    def get_charges(self, combined_inputs, positions, types, cell):
        # Passthrough function to access charges
        return self._get_charges(combined_inputs, positions, types, cell)[1]

    @flax.linen.compact
    def __call__(self, combined_inputs, positions, types, cell):
        """Compute electrostatic contributions to potential energy

        Args:
            combined_inputs : Mixed descriptors
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The n_atoms electrostatic contributions to the energy
        """
        sigmas = self.sigmas
        sigmas_expanded = jnp.take(jnp.array(sigmas), types)
        offdiag, charges = self._get_charges(combined_inputs, positions, types,
                                             cell)

        self_contribution = ((charges * charges) /
                             (2 * sigmas_expanded * jnp.sqrt(jnp.pi))).sum()
        pair_contribution = jnp.dot(jnp.dot(offdiag, charges), charges) / 2.
        return jnp.squeeze(self_contribution + pair_contribution)


class DynamicsModelWithCharges(flax.linen.Module):
    """Wrapper model around the core layers and the electrostatic model
     to calculate energies and forces.

    The class does not provide a __call__ method, forcing the user to choose
    what to evaluate (forces, energies or both).

    Args:
        n_types: The number of atomic types in the system.
        embed_d: The dimension of the embedding vector to be mixed with the
            descriptors.
        descriptor_generator: The function mapping the atomic coordinates,
            types and cell size to descriptors.
        core_model: The model that takes all the descriptors and returns an
            atomic contribution to the energy.
        electrostatic_model: The model that takes all the descriptors and returns
            atomic electronegativities
        mixer: The function that takes the Bessel descriptors and the
            embedding vectors and creates the input descriptors for the core
            model. The default choice just concatenates them.
    """
    n_types: int
    embed_d: int
    descriptor_generator: Callable
    core_model: flax.linen.Module
    electrostatic_model: flax.linen.Module
    mixer: Callable = lambda d, e: jnp.concatenate(
        [d.reshape((d.shape[0], -1)), e], axis=1)

    def setup(self):
        # These neurons create the embedding vector.
        self.embed = flax.linen.Embed(self.n_types, self.embed_d)
        # This linear layer centers and scales the energy after the core
        # has done its job.
        self.denormalizer = flax.linen.Dense(1)
        # The checkpointing strategy can be reconsidered to achieve different
        # tradeoffs between memory and CPU usage.
        self._gradient_calculator = jax.grad(self.calc_potential_energy, argnums=0)

    def calc_atomic_energies(self, positions, types, cell):
        """Compute the atomic contributions to the potential energy.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The n_atoms contributions to the energy.
        """
        descriptors = self.descriptor_generator(positions, types, cell)
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        results = self.core_model(combined_inputs)
        results = self.denormalizer(results)
        electrostatic_energy = self.electrostatic_model(
            combined_inputs, positions, types, cell)
        return results + electrostatic_energy

    def calc_ele_energy(self, positions, types, cell):
        descriptors = self.descriptor_generator(positions, types, cell)
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        electrostatic_energy = self.electrostatic_model(
            combined_inputs, positions, types, cell)
        return electrostatic_energy  

    def calc_potential_energy(self, positions, types, cell):
        """Compute the total potential energy of the system.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The sum of all atomic contributions to the potential energy.
        """
        contributions = self.calc_atomic_energies(positions, types, cell)
        return jnp.squeeze(contributions.sum(axis=0))

    def calc_forces(self, positions, types, cell):
        """Compute the force on each atom.

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The (n_atoms, 3) vector containing all the forces.
        """
        return -self._gradient_calculator(positions, types, cell)

    def calc_charges(self, positions, types, cell):
        """Computes atomic charges 
        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The n_atoms vector containing all the charges.
        """
        descriptors = self.descriptor_generator(positions, types, cell)
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        charges = self.electrostatic_model.get_charges(combined_inputs,
                                                       positions, types, cell)
        return charges

    def calc_electronegativities(self, positions, types, cell):
        """Computes atomic electronegativities

        Args:
            positions: The (n_atoms, 3) vector with the Cartesian coordinates
                of each atom.
            types: The atomic types, codified as integers from 0 to n_atoms - 1.
            cell: The (3, 3) matrix representing the simulation box ir periodic
                boundary conditions are in effect, or None otherwise.

        Returns:
            The n_atoms vector containing all the electronegativities.
        """
        descriptors = self.descriptor_generator(positions, types, cell)
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        electronegativities = (
            self.electrostatic_model.calc_electronegativities_from_descriptors(
                combined_inputs))
        return electronegativities



