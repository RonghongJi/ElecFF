from typing import Callable
from typing import Sequence

import jax
import jax.nn
import jax.numpy as jnp
import flax.linen
from neuralil.neural_network.bessel_descriptors import center_at_atoms
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Sequence

class Core_two(flax.linen.Module):

    layer_widths: Sequence[int]
    activation_function: Callable = flax.linen.swish

    @flax.linen.compact
    def __call__(self, descriptors):
        result = self.activation_function(
            flax.linen.Dense(self.layer_widths[0])(descriptors))
        for w in self.layer_widths[1:]:
            result = self.activation_function(flax.linen.LayerNorm()(
                flax.linen.Dense(w, use_bias=False)(result)))
        return self.activation_function(flax.linen.Dense(2)(result))


class NoElec(flax.linen.Module):

    n_types: int
    embed_d: int
    descriptor_generator: Callable
    core_model: flax.linen.Module
    mixer: Callable = lambda d, e: jnp.concatenate(
        [d.reshape((d.shape[0], -1)), e], axis=1)

    def setup(self):

        self.embed = flax.linen.Embed(self.n_types, self.embed_d)
        self.denormalizer = flax.linen.Dense(2)
        self._gradient_calculator = jax.grad(self.calc_potential_energy_two, argnums=0)

    def calc_potential_energy_from_descriptors(self, combined_inputs):
        contributions = self.core_model(combined_inputs)
        contributions = self.denormalizer(contributions)[..., 0]
        return jnp.squeeze(contributions.sum(axis=0))

    def calc_atomic_energies(self, positions, types, cell):

        descriptors = self.descriptor_generator(positions, types, cell)
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        results = self.core_model(combined_inputs)
        results = self.denormalizer(results)
        return results

    def calc_potential_energy(self, positions, types, cell):

        contributions = self.calc_atomic_energies(positions, types, cell)[..., 0]
        return jnp.squeeze(contributions.sum(axis=0))

    def calc_potential_energy_two(self, positions, types, cell):

        contributions = self.calc_atomic_energies(positions, types, cell)[..., 1]
        return jnp.squeeze(contributions.sum(axis=0))
    def calc_forces(self, positions, types, cell):

        return -self._gradient_calculator(positions, types, cell)



class ElectrostaticModel(flax.linen.Module):

    core_model: Sequence[int]
    n_types: int
    sigmas: Sequence[float]

    def setup(self):
        self.denormalizer = flax.linen.Dense(1)
        self.hardness = self.param('hardness', flax.linen.initializers.ones,
                                   (self.n_types, ))

    def calc_electronegativities_from_descriptors(self, combined_inputs):

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

        return self._get_charges(combined_inputs, positions, types, cell)[1]

    @flax.linen.compact
    def __call__(self, combined_inputs, positions, types, cell):

        sigmas = self.sigmas
        sigmas_expanded = jnp.take(jnp.array(sigmas), types)
        offdiag, charges = self._get_charges(combined_inputs, positions, types,
                                             cell)

        self_contribution = ((charges * charges) /
                             (2 * sigmas_expanded * jnp.sqrt(jnp.pi))).sum()
        pair_contribution = jnp.dot(jnp.dot(offdiag, charges), charges) / 2.
        return jnp.squeeze(self_contribution + pair_contribution)


class ElecFF(flax.linen.Module):

    n_types: int
    embed_d: int
    descriptor_generator: Callable
    core_model: flax.linen.Module
    electrostatic_model: flax.linen.Module
    mixer: Callable = lambda d, e: jnp.concatenate(
        [d.reshape((d.shape[0], -1)), e], axis=1)

    def setup(self):

        self.embed = flax.linen.Embed(self.n_types, self.embed_d)

        self.denormalizer = flax.linen.Dense(2)

        self._gradient_calculator = jax.grad(self.calc_potential_energy_two, argnums=0)

    def calc_potential_energy_from_descriptors(self, combined_inputs):
        contributions = self.core_model(combined_inputs)
        contributions = self.denormalizer(contributions)[..., 0]
        return jnp.squeeze(contributions.sum(axis=0))

    def calc_atomic_energies(self, positions, types, cell):

        descriptors = self.descriptor_generator(positions, types, cell)
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        results = self.core_model(combined_inputs)
        results = self.denormalizer(results)
        return results

    def calc_potential_energy(self, positions, types, cell):

        contributions = self.calc_atomic_energies(positions, types, cell)[..., 0]
        return jnp.squeeze(contributions.sum(axis=0))

    def calc_potential_energy_two(self, positions, types, cell):

        contributions = self.calc_atomic_energies(positions, types, cell)[..., 1]
        return jnp.squeeze(contributions.sum(axis=0))
    def calc_forces(self, positions, types, cell):

        return -self._gradient_calculator(positions, types, cell)

    def calc_charges(self, positions, types, cell):

        descriptors = self.descriptor_generator(positions, types, cell)
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        charges = self.electrostatic_model.get_charges(combined_inputs,
                                                       positions, types, cell)
        return charges

    def calc_electronegativities(self, positions, types, cell):

        descriptors = self.descriptor_generator(positions, types, cell)
        embeddings = self.embed(types)
        combined_inputs = self.mixer(descriptors, embeddings)
        electronegativities = (
            self.electrostatic_model.calc_electronegativities_from_descriptors(
                combined_inputs))
        return electronegativities