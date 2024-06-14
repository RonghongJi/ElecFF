#!/usr/bin/env python

import pathlib
import json
import pickle
import random
import collections
import sys
sys.path.append("/home/rhji/train/src/neuralil/neural_network")

import tqdm
import tqdm.auto
import jax
import jax.nn
import jax.numpy as jnp
import flax
import flax.optim
import flax.jax_utils
import flax.serialization

# Import custom modules and functions
from bessel_descriptors import (
    get_max_number_of_neighbors
)
from bessel_descriptors import PowerSpectrumGenerator
from model import Core_two as Core
from model import ElecFF
from model import ElectrostaticModel

# Set random seed for reproducibility
random.seed(2024)

# Constants
N_PAIR = 6
TRAINING_FRACTION = .9
R_CUT = 4
N_MAX = 4
EMBED_D = 2
N_EPOCHS = 501
N_BATCH = 8
LOG_COSH_PARAMETER_energy = 1e2
LOG_COSH_PARAMETER_force = 1e1

# Create log-cosh loss function
def create_log_cosh(parameter):
    if parameter <= 0.:
        raise ValueError("parameter must be positive")

    def nruter(x):
        return (
            (jax.nn.softplus(2. * parameter * x) - jnp.log(2)) / parameter - x
        )

    return nruter

# Define huber loss function
def huber_loss(y_true, y_pred, delta):
    residual = jnp.abs(y_true - y_pred)
    squared_loss = 0.5 * jnp.square(residual)
    linear_loss = delta * residual - 0.5 * jnp.square(delta)
    loss = jnp.where(residual < delta, squared_loss, linear_loss)
    loss = jnp.abs(loss) + 1e-6 
    return jnp.mean(loss)

# Create learning rate schedule using one cycle policy
def create_onecycle_schedule(lr_min, lr_max, lr_final, steps_per_epoch):
    cycle_length = int(.9 * steps_per_epoch)

    def learning_rate_fn(step):
        slope = 2. * (lr_max - lr_min) / cycle_length
        if step < cycle_length // 2:
            return lr_min + slope * step
        elif step < cycle_length:
            return lr_max - slope * (step - cycle_length // 2)
        else:
            return lr_final

    return learning_rate_fn

# Define training step
@jax.jit
def train_step_sum(optimizer_sum, positions_batch, types_batch, cells_batch, energies_batch, forces_batch, learning_rate):
    def calc_loss_sum(params):
        def contribution_energy(positions, types, cell, energy):
            pred = dynamics_model.apply(
                params,
                positions,
                types,
                cell,
                method=ElecFF.calc_potential_energy
            )
            delta = energy - pred
            return huber_loss(energy, pred, 0.02)
        
        def contribution_force(positions, types, cell, forces):
            pred = dynamics_model.apply(
                params,
                positions,
                types,
                cell,
                method=ElecFF.calc_forces
            )
            delta = forces - pred
            return log_cosh_force(delta).mean()

        energy_loss = jnp.mean(jax.vmap(jax.checkpoint(contribution_energy))(
            positions_batch, types_batch, cells_batch, energies_batch
        ), axis=0) / positions_batch.shape[1]

        force_loss = jnp.mean(jax.vmap(jax.checkpoint(contribution_force))(
            positions_batch, types_batch, cells_batch, forces_batch
        ), axis=0)
        
        energy_weight = 1
        force_weight = 100

        weighted_loss = energy_weight * energy_loss + force_weight * force_loss

        return weighted_loss

    calc_loss_and_grad_sum = jax.value_and_grad(calc_loss_sum)
    loss_val_sum, loss_grad_sum = calc_loss_and_grad_sum(optimizer_sum.target)
    optimizer_sum = optimizer_sum.apply_gradient(loss_grad_sum, learning_rate=learning_rate)
    return loss_val_sum, optimizer_sum

# Define function to compute error contributions for evaluation
@jax.jit
def error_contributions(params, positions, types, cell, forces):
    pred = dynamics_model.apply(
        params, positions, types, cell, method=ElecFF.calc_forces
    )
    mse_contribution = ((forces - pred) * (forces - pred)).mean()
    mae_contribution = jnp.fabs(forces - pred).mean()
    return (mse_contribution, mae_contribution)

# Define evaluation step
def eval_step(params, positions_batch, types_batch, cells_batch, forces_batch):
    total_squared_error = 0.
    total_absolute_error = 0.
    for p, t, c, f in zip(positions_batch, types_batch, cells_batch, forces_batch):
        squared_error, absolute_error = error_contributions(params, p, t, c, f)
        total_squared_error += squared_error
        total_absolute_error += absolute_error
    return (
        jnp.sqrt(total_squared_error.sum() / len(positions_batch)),
        total_absolute_error.sum() / len(positions_batch)
    )

# Define function to train one epoch
def train_epoch_sum(optimizer_sum, batch_size, epoch, rng):
    steps_per_epoch = n_training // batch_size

    perms = jax.random.permutation(rng, n_training)
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))
    learning_schedule = create_onecycle_schedule(
        1e-5, 1e-4, 1e-6, steps_per_epoch
    )

    with tqdm.auto.trange(len(perms)) as t:
        t.set_description(f"EPOCH #{epoch + 1}")
        for iperm in t:
            perm = perms[iperm].sort()
            positions_batch = positions_train[perm, ...]
            types_batch = types_train[perm, ...]
            cells_batch = cells_train[perm, ...]
            energies_batch = energies_train[perm, ...]
            forces_batch = forces_train[perm, ...]
            learning_rate = learning_schedule(iperm)
            loss, optimizer_sum = train_step_sum(
                optimizer_sum, positions_batch, types_batch, cells_batch,
                energies_batch, forces_batch, learning_rate
            )
            t.set_postfix(loss=loss, learning_rate=learning_rate)

    return optimizer_sum

# Create log-cosh loss functions for energy and force
log_cosh_energy = create_log_cosh(LOG_COSH_PARAMETER_energy)
log_cosh_force = create_log_cosh(LOG_COSH_PARAMETER_force)

# Define masses and types for cations and anions
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

# Read data from JSON file
print("- Reading the JSON file")
cells = []
positions = []
energies = []
forces = []
with open((pathlib.Path(__file__).parent / "[bmim][bf4]_cleaned_3000.json").resolve(), "r") as json_f:
    for line in json_f:
        json_data = json.loads(line)
        cells.append(jnp.diag(jnp.array(json_data["Cell_size"])))
        positions.append(json_data["Positions"])
        energies.append(json_data["Energy"])
        forces.append(json_data["Forces"])
print("- Done")

n_configurations = len(positions)
types = [types for _ in range(n_configurations)]

# Shuffle data
order = list(range(n_configurations))
random.shuffle(order)
cells = jnp.array([cells[i] for i in order])
positions = jnp.array([positions[i] for i in order])
energies = jnp.array([energies[i] for i in order])
types = jnp.array([types[i] for i in order])
forces = jnp.array([forces[i] for i in order])


# Split data into training and validation sets
n_training = int(TRAINING_FRACTION * n_configurations)
n_validation = n_configurations - n_training

n_types = int(types.max()) + 1

# Determine the maximum number of neighbors
max_neighbors = 1
for p, c in zip(positions, cells):
    max_neighbors = max(
        max_neighbors,
        get_max_number_of_neighbors(jnp.asarray(p), R_CUT, jnp.asarray(c))
    )

# Initialize descriptor generator
pipeline = PowerSpectrumGenerator(N_MAX, R_CUT, n_types, max_neighbors)

# Generate descriptors
descriptors = pipeline(positions[0], types[0], cells[0])
n_descriptors = descriptors.shape[1] * descriptors.shape[2]

# Split data into training and validation sets
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

# Initialize models
core_model = Core([256, 128, 128, 64, 32, 16, 16, 16])
covalent_radii = {'C': 0.76, 'N': 0.71, 'H': 0.31, 'B': 0.82, 'F': 0.64}
sigmas = jnp.asarray([covalent_radii[key] for key in type_dict])

core_model_electrostatic = Core([16, 16, 16])
electrostatic_model = ElectrostaticModel(core_model_electrostatic, n_types, sigmas)
dynamics_model = ElecFF(n_types, EMBED_D, pipeline.process_data, core_model, electrostatic_model)

# Initialize parameters
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
params = dynamics_model.init(
    init_rng,
    positions_train[0],
    types_train[0],
    cells_train[0],
    method=ElecFF.calc_forces
)

# Initialize optimizer
optimizer_def = flax.optim.Adam(learning_rate=1.)
optimizer = optimizer_def.create(params)

PICKLE_FILE = f"bmimbf4_charge_3000.pickle"

# Training loop
min_mae = jnp.inf
for i in range(N_EPOCHS):
    rng, epoch_rng = jax.random.split(rng)
    optimizer = train_epoch_sum(optimizer, N_BATCH, i, epoch_rng)
    rmse, mae = eval_step(
        optimizer.target, positions_validate, types_validate, cells_validate,
        forces_validate
    )

    dict_output = flax.serialization.to_state_dict(optimizer)
    if mae < min_mae:
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(dict_output, f, protocol=5)
        min_mae = mae
