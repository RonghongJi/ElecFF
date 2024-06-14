#!/usr/bin/env python

# Importing necessary libraries
import pathlib
import json
import pickle
import random
import collections
import sys
sys.path.append("/home/rhji/train/src/neuralil/neural_network")

import numpy as onp
import tqdm.auto

import jax

# Setting JAX to use CPU
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import flax
import flax.optim
import flax.jax_utils
import flax.serialization

from bessel_descriptors import (
    get_max_number_of_neighbors
)
from bessel_descriptors import PowerSpectrumGenerator
from model import Core_two as Core
from model import ElecFF
from model import ElectrostaticModel

import matplotlib
import matplotlib.pyplot as plt

import scipy
import scipy.stats

# Setting random seed
random.seed(2024)

# Constants and parameters
N_PAIR = 6
TRAINING_FRACTION = .9
R_CUT = 4
N_MAX = 4
EMBED_D = 2

# Function to calculate potential energy using the model
@jax.jit
def calculate_energy(params, positions, types, cell):
    return dynamics_model.apply(
        params,
        positions,
        types,
        cell,
        method=ElecFF.calc_potential_energy
    )


# Function to compute error contributions (MSE and MAE)
@jax.jit
def error_contributions(params, positions, types, cell, energy, forces):
    pred_energy = dynamics_model.apply(
        params,
        positions,
        types,
        cell,
        method=ElecFF.calc_potential_energy
    )
    pred_forces = dynamics_model.apply(
        params, positions, types, cell, method=ElecFF.calc_forces
    )
    delta_energy = energy - pred_energy
    delta_forces = forces - pred_forces
    mse_contribution_energy = delta_energy * delta_energy
    mae_contribution_energy = jnp.fabs(delta_energy)
    mse_contribution_forces = (delta_forces * delta_forces).mean()
    mae_contribution_forces = jnp.fabs(delta_forces).mean()
    return (
        mse_contribution_energy,
        mae_contribution_energy,
        mse_contribution_forces,
        mae_contribution_forces,
    )

# Function to evaluate model performance on a batch
def eval_step(
    params, positions_batch, types_batch, cells_batch, energies_batch,
    forces_batch
):
    total_squared_error_energy = 0.
    total_absolute_error_energy = 0.
    total_squared_error_forces = 0.
    total_absolute_error_forces = 0.
    for p, t, c, e, f in zip(
        positions_batch, types_batch, cells_batch, energies_batch, forces_batch
    ):
        se_e, ae_e, se_f, ae_f = error_contributions(params, p, t, c, e, f)
        total_squared_error_energy += se_e
        total_absolute_error_energy += ae_e
        total_squared_error_forces += se_f
        total_absolute_error_forces += ae_f
    return (
        jnp.sqrt(total_squared_error_energy.sum() / len(positions_batch)) /
        positions_batch.shape[1], total_absolute_error_energy.sum() /
        len(positions_batch) / positions_batch.shape[1],
        jnp.sqrt(total_squared_error_forces.sum() / len(positions_batch)),
        total_absolute_error_forces.sum() / len(positions_batch)
    )

# Masses and types of ions
mass_cation = [14.007, 14.007, 12.011, 1.008, 12.011, 1.008, 12.011, 1.008, 12.011, 1.008, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 1.008]
mass_anion = [10.811, 18.998, 18.998, 18.998, 18.998]
masses = jnp.array(N_PAIR * mass_cation + N_PAIR * mass_anion)

type_cation = ["N", "N", "C", "H", "C", "H", "C", "H", "C", "H", "H", "H", "C", "H", "H", "C", "H", "H", "C", "H", "H", "C", "H", "H", "H"]
type_anion = ["B", "F", "F", "F", "F"]
types = N_PAIR * type_cation + N_PAIR * type_anion
unique_types = sorted(set(types))
type_dict = collections.OrderedDict()
for i, k in enumerate(unique_types):
    type_dict[k] = i
types = jnp.array([type_dict[i] for i in types])
n_atoms = len(types)

# Reading JSON file to retrieve data
print("- Reading the JSON file")
cells = []
positions = []
energies = []
forces = []

with open("[bmim][bf4]_cleaned_3000.json", "r") as json_f:
    for line in json_f:
        json_data = json.loads(line)
        cells.append(jnp.diag(jnp.array(json_data["Cell_size"])))
        positions.append(json_data["Positions"])
        energies.append(json_data["Energy"])
        forces.append(json_data["Forces"])

print("- Done")

# Extracting additional information from JSON
n_atoms = json_data["Natoms"]
box_size = json_data["Cell_size"][0]
n_configurations = len(positions)
types = [types for i in range(n_configurations)]

# Randomizing order of configurations
order = list(range(n_configurations))
random.shuffle(order)
cells = jnp.array([cells[i] for i in order])
positions = jnp.array([positions[i] for i in order])
energies = jnp.array([energies[i] for i in order])
types = jnp.array([types[i] for i in order])
forces = jnp.array([forces[i] for i in order])

print(f"- {n_configurations} configurations are available")

# Splitting data into training and validation sets
n_training = int(TRAINING_FRACTION * n_configurations)
n_validation = n_configurations - n_training
print(f"\t- {n_training} will be used for training")
print(f"\t- {n_validation} will be used for validation")
n_types = int(types.max()) + 1

# Generating power spectrum descriptors
max_neighbors = 1
for p, c in zip(positions, cells):
    max_neighbors = max(
        max_neighbors,
        get_max_number_of_neighbors(jnp.asarray(p), R_CUT, jnp.asarray(c))
    )
print("- Maximum number of neighbors that must be considered: ", max_neighbors)
pipeline = PowerSpectrumGenerator(N_MAX, R_CUT, n_types, max_neighbors)

descriptors = pipeline(positions[0], types[0], cells[0])
n_descriptors = descriptors.shape[1] * descriptors.shape[2]

# Splitting data into training and validation sets
cells_train = cells[:n_training]
positions_train = positions[:n_training]
types_train = types[:n_training]
energies_train = energies[:n_training]
forces_train = forces[:n_training]

cells_validate = cells[n_training:]
positions_validate = positions[n_training:]
types_validate = types[n_training:]
energies_validate = energies[n_training:]
forces_validate = forces[n_training:]

# Calculating average model energy during training
average_model_energy_train = jnp.mean(
    jnp.array(
        [
            calculate_energy(optimizer.target, p, t, c)
            for (p, t, c) in zip(positions_train, types_train, cells_train)
        ]
    )
)

# Initializing models and optimizer
average_dft_energy_train = energies_train.mean()
core_model = Core([256, 128, 128, 64, 32, 16, 16, 16])

covalent_radii = {'C': 0.76, 'N': 0.71, 'H': 0.31, 'B': 0.82, 'F':0.64}
sigmas = jnp.asarray([covalent_radii[key] for key in type_dict])

core_model_electrostatic = Core([16, 16, 16])
electrostatic_model = ElectrostaticModel(
    core_model_electrostatic, n_types, sigmas
)
dynamics_model = ElecFF(
    n_types, EMBED_D, pipeline.process_data, core_model, electrostatic_model
)

# Initializing parameters and optimizer from a saved state
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
params = dynamics_model.init(
    init_rng,
    positions_train[0],
    types_train[0],
    cells_train[0],
    method=ElecFF.calc_forces
)

optimizer_def = flax.optim.Adam(learning_rate=1.)
optimizer = optimizer_def.create(params)

pickle_file = "bmimbf4_charge_3000.pickle"
with open(pickle_file, "rb") as f:
    state_dict = pickle.load(f)
    optimizer = flax.serialization.from_state_dict(optimizer, state_dict)

# Evaluating model performance on training and validation sets
rmse_e_train, mae_e_train, rmse_f_train, mae_f_train = eval_step(
    optimizer.target, positions_train, types_train, cells_train, energies_train,
    forces_train
)

rmse_e_validate, mae_e_validate, rmse_f_validate, mae_f_validate = eval_step(
    optimizer.target, positions_validate, types_validate, cells_validate,
    energies_validate, forces_validate
)

# Converting [a.u.] to meV/atom and meV/Å
e_conversion_factor = 27211 / n_atoms
f_conversion_factor = 27211 / box_size

# Scaling errors to physical units
rmse_e_train *= e_conversion_factor
mae_e_train *= e_conversion_factor
rmse_f_train *= f_conversion_factor
mae_f_train *= f_conversion_factor

rmse_e_validate *= e_conversion_factor
mae_e_validate *= e_conversion_factor
rmse_f_validate *= f_conversion_factor
mae_f_validate *= f_conversion_factor

# Printing out results
print("[bmim][bf4]_3000:")
print(
    f"TRAIN:\n"
    f"\tRMSE = {rmse_e_train:.5f} meV/atom, {rmse_f_train:.5f} meV/Å\n"
    f"\tMAE = {mae_e_train:.5f} meV/atom, {mae_f_train:.5f} meV/Å"
)
print(
    f"VALIDATION:\n"
    f"\tRMSE = {rmse_e_validate:.5f} meV/atom, {rmse_f_validate:.5f} meV/Å\n"
    f"\tMAE = {mae_e_validate:.5f} meV/atom, {mae_f_validate:.5f} meV/Å"
)