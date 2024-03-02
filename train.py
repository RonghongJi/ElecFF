#!/usr/bin/env python

import pathlib
import json
import pickle
import random
import collections

import tqdm
import tqdm.auto
import jax
import jax.nn
import jax.numpy as jnp
import flax
import flax.optim
import flax.jax_utils
import flax.serialization

from bessel_descriptors import get_max_number_of_neighbors
from bessel_descriptors import PowerSpectrumGenerator
from model import Core
from model import CoreRelu
from model import DynamicsModel
from model import DynamicsModelWithCharges
from model import ElectrostaticModel
from model import ReluElectrostaticModel
from model import ReluDynamicsModelWithCharges
from model import ResNetCore


# ElecFF train

random.seed(2024)

N_PAIR = 6

TRAINING_FRACTION = .9
R_CUT = 4
N_MAX = 4
EMBED_D = 2

N_EPOCHS = 501
N_BATCH = 8

LOG_COSH_PARAMETER_energy = 1e2
LOG_COSH_PARAMETER_force = 1e1


def create_log_cosh(parameter):
    """Return a differentiable implementation of the log cosh loss.

    Args:
        parameter: The positive parameter determining the transition between
            the linear and quadratic regimes.

    Returns:
        A float->float function to compute a contribution to the loss.

    Raises:
        ValueError: If parameter is not positive.
    """
    if parameter <= 0.:
        raise ValueError("parameter must be positive")

    def nruter(x):
        return (
            (jax.nn.softplus(2. * parameter * x) - jnp.log(2)) / parameter - x
        )

    return nruter
    
def mae_loss(y_true, y_pred):
    absolute_error = jnp.abs(y_true - y_pred)
    loss = jnp.mean(absolute_error)
    return loss

def mse_loss(y_true, y_pred):
    return jnp.mean(jnp.square(y_true - y_pred))    
    
def huber_loss(y_true, y_pred, delta):
    residual = jnp.abs(y_true - y_pred)
    squared_loss = 0.5 * jnp.square(residual)
    linear_loss = delta * residual - 0.5 * jnp.square(delta)
    loss = jnp.where(residual < delta, squared_loss, linear_loss)
    loss = jnp.abs(loss)+ 1e-6 
    return jnp.mean(loss)


log_cosh_energy = create_log_cosh(LOG_COSH_PARAMETER_energy)
log_cosh_force = create_log_cosh(LOG_COSH_PARAMETER_force)


mass_cation = [
    14.007, 14.007, 12.011, 1.008, 12.011, 1.008, 12.011, 1.008, 12.011, 1.008, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 1.008]
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

print("- Reading the JSON file")
cells = []
positions = []
energies = []
forces = []
with open(
    (pathlib.Path(__file__).parent /
     "[bmim][bf4]_cleaned_3000.json").resolve(), "r"
) as json_f:
    for line in json_f:
        json_data = json.loads(line)
        cells.append(jnp.diag(jnp.array(json_data["Cell_size"])))
        positions.append(json_data["Positions"])
        energies.append(json_data["Energy"])
        forces.append(json_data["Forces"])
print("- Done")

n_configurations = len(positions)
types = [types for i in range(n_configurations)]

order = list(range(n_configurations))
random.shuffle(order)
cells = jnp.array([cells[i] for i in order])
positions = jnp.array([positions[i] for i in order])
energies = jnp.array([energies[i] for i in order])
types = jnp.array([types[i] for i in order])
forces = jnp.array([forces[i] for i in order])

print(f"- {n_configurations} configurations are available")

n_training = int(TRAINING_FRACTION * n_configurations)
n_validation = n_configurations - n_training
print(f"\t- {n_training} will be used for training")
print(f"\t- {n_validation} will be used for validation")
n_types = int(types.max()) + 1

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


core_model = Core([256, 128, 128, 64, 32, 16, 16, 16])
covalent_radii = {'C': 0.76, 'N': 0.71, 'H': 0.31, 'B': 0.82, 'F':0.64}
sigmas = jnp.asarray([covalent_radii[key] for key in type_dict])

core_model_electrostatic = Core([16, 16, 16])
electrostatic_model = ElectrostaticModel(
    core_model_electrostatic, n_types, sigmas
)
dynamics_model = DynamicsModelWithCharges(
    n_types, EMBED_D, pipeline.process_data, core_model, electrostatic_model
)

rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
params = dynamics_model.init(
    init_rng,
    positions_train[0],
    types_train[0],
    cells_train[0],
    method=DynamicsModelWithCharges.calc_forces
)


optimizer_def = flax.optim.Adam(learning_rate=1.)
optimizer = optimizer_def.create(params)


@jax.jit
def train_step_sum(
    optimizer_sum, positions_batch, types_batch, cells_batch, energies_batch, forces_batch,
    learning_rate
):
    def calc_loss_sum(params):
        def contribution_energy(positions, types, cell, energy):
            pred = dynamics_model.apply(
                params,
                positions,
                types,
                cell,
                method=DynamicsModelWithCharges.calc_potential_energy
            )
            delta = energy - pred
            return huber_loss(energy, pred, 0.02)
        
        def contribution_force(positions, types, cell, forces):
            pred = dynamics_model.apply(
                params,
                positions,
                types,
                cell,
                method=DynamicsModelWithCharges.calc_forces
            )
            delta = forces - pred
            return log_cosh_force(delta).mean()

        energy_loss = jnp.mean(jax.vmap(jax.checkpoint(contribution_energy))
        (positions_batch, types_batch, cells_batch, energies_batch),
        axis=0) / positions_batch.shape[1]
        force_loss = jnp.mean(jax.vmap(jax.checkpoint(contribution_force))
        (positions_batch, types_batch, cells_batch, forces_batch), axis=0)
        
        energy_weight = 1
        force_weight = 100

        weighted_loss = energy_weight * energy_loss + force_weight * force_loss

        return weighted_loss

    calc_loss_and_grad_sum = jax.value_and_grad(calc_loss_sum)
    loss_val_sum, loss_grad_sum = calc_loss_and_grad_sum(optimizer_sum.target)
    optimizer_sum = optimizer_sum.apply_gradient(loss_grad_sum, learning_rate=learning_rate)
    return loss_val_sum, optimizer_sum


@jax.jit
def error_contributions(params, positions, types, cell, forces):
    pred = dynamics_model.apply(
        params, positions, types, cell, method=DynamicsModelWithCharges.calc_forces
    )
    mse_contribution = ((forces - pred) * (forces - pred)).mean()
    mae_contribution = jnp.fabs(forces - pred).mean()
    return (mse_contribution, mae_contribution)


def eval_step(params, positions_batch, types_batch, cells_batch, forces_batch):
    total_squared_error = 0.
    total_absolute_error = 0.
    for p, t, c, f in zip(
        positions_batch, types_batch, cells_batch, forces_batch
    ):
        squared_error, absolute_error = error_contributions(params, p, t, c, f)
        total_squared_error += squared_error
        total_absolute_error += absolute_error
    return (
        jnp.sqrt(total_squared_error.sum() / len(positions_batch)),
        total_absolute_error.sum() / len(positions_batch)
    )


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


PICKLE_FILE = f"ElecFF.pickle"

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
            print("- Saving the most recent state")
            pickle.dump(dict_output, f, protocol=5)
        min_mae = mae

pickle_file = "ElecFF.pickle"
with open(pickle_file, "rb") as f:
    state_dict = pickle.load(f)
    optimizer = flax.serialization.from_state_dict(optimizer, state_dict)


@jax.jit
def calculate_energy(params, positions, types, cell):
    return dynamics_model.apply(
        params,
        positions,
        types,
        cell,
        method=DynamicsModelWithCharges.calc_potential_energy
    )


average_model_energy_train = jnp.mean(
    jnp.array(
        [
            calculate_energy(optimizer.target, p, t, c)
            for (p, t, c) in zip(positions_train, types_train, cells_train)
        ]
    )
)


@jax.jit
def error_contributions(params, positions, types, cell, energy, forces):
    pred_energy = dynamics_model.apply(
        params,
        positions,
        types,
        cell,
        method=DynamicsModelWithCharges.calc_potential_energy
    )
    pred_forces = dynamics_model.apply(
        params, positions, types, cell, method=DynamicsModelWithCharges.calc_forces
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


rmse_e_train, mae_e_train, rmse_f_train, mae_f_train = eval_step(
    optimizer.target, positions_train, types_train, cells_train, energies_train,
    forces_train
)

rmse_e_validate, mae_e_validate, rmse_f_validate, mae_f_validate = eval_step(
    optimizer.target, positions_validate, types_validate, cells_validate,
    energies_validate, forces_validate
)

hartree_to_mev = 27211

rmse_e_train = (rmse_e_train * hartree_to_mev) / n_atoms
mae_e_train = (mae_e_train * hartree_to_mev) / n_atoms
rmse_e_validate = (rmse_e_validate * hartree_to_mev) / n_atoms
mae_e_validate = (mae_e_validate * hartree_to_mev) / n_atoms
rmse_f_train = (rmse_f_train * hartree_to_mev) / cells
mae_f_train = (mae_f_train * hartree_to_mev) / cells
rmse_f_validate = (rmse_f_validate * hartree_to_mev) / cells
mae_f_validate = (mae_f_validate * hartree_to_mev) / cells

print("ElecFF:")
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


# NoElec train

random.seed(2024)

N_PAIR = 6

TRAINING_FRACTION = .9
R_CUT = 4
N_MAX = 4
EMBED_D = 2

N_EPOCHS = 501
N_BATCH = 8


LOG_COSH_PARAMETER_energy = 1e2
LOG_COSH_PARAMETER_force = 1e1  


def create_log_cosh(parameter):
    """Return a differentiable implementation of the log cosh loss.

    Args:
        parameter: The positive parameter determining the transition between
            the linear and quadratic regimes.

    Returns:
        A float->float function to compute a contribution to the loss.

    Raises:
        ValueError: If parameter is not positive.
    """
    if parameter <= 0.:
        raise ValueError("parameter must be positive")

    def nruter(x):
        return (
            (jax.nn.softplus(2. * parameter * x) - jnp.log(2)) / parameter - x
        )

    return nruter

    
def mae_loss(y_true, y_pred):
    absolute_error = jnp.abs(y_true - y_pred)
    loss = jnp.mean(absolute_error)
    return loss

def mse_loss(y_true, y_pred):
    return jnp.mean(jnp.square(y_true - y_pred))    
    
def huber_loss(y_true, y_pred, delta):
    residual = jnp.abs(y_true - y_pred)
    squared_loss = 0.5 * jnp.square(residual)
    linear_loss = delta * residual - 0.5 * jnp.square(delta)
    loss = jnp.where(residual < delta, squared_loss, linear_loss)
    loss = jnp.abs(loss)+ 1e-6 
    return jnp.mean(loss)

log_cosh_energy = create_log_cosh(LOG_COSH_PARAMETER_energy)
log_cosh_force = create_log_cosh(LOG_COSH_PARAMETER_force)


mass_cation = [
    14.007, 14.007, 12.011, 1.008, 12.011, 1.008, 12.011, 1.008, 12.011, 1.008, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 1.008]
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

print("- Reading the JSON file")
cells = []
positions = []
energies = []
forces = []
with open(
    (pathlib.Path(__file__).parent /
     "[bmim][bf4]_cleaned_3000.json").resolve(), "r"
) as json_f:
    for line in json_f:
        json_data = json.loads(line)
        cells.append(jnp.diag(jnp.array(json_data["Cell_size"])))
        positions.append(json_data["Positions"])
        energies.append(json_data["Energy"])
        forces.append(json_data["Forces"])
print("- Done")

n_configurations = len(positions)
types = [types for i in range(n_configurations)]

order = list(range(n_configurations))
random.shuffle(order)
cells = jnp.array([cells[i] for i in order])
positions = jnp.array([positions[i] for i in order])
energies = jnp.array([energies[i] for i in order])
types = jnp.array([types[i] for i in order])
forces = jnp.array([forces[i] for i in order])

print(f"- {n_configurations} configurations are available")

n_training = int(TRAINING_FRACTION * n_configurations)
n_validation = n_configurations - n_training
print(f"\t- {n_training} will be used for training")
print(f"\t- {n_validation} will be used for validation")
n_types = int(types.max()) + 1

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


core_model = Core([256, 128, 128, 64, 32, 16, 16, 16])
dynamics_model = DynamicsModel(
    n_types,
    EMBED_D,
    pipeline.process_data,
    core_model,
)

rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
params = dynamics_model.init(
    init_rng,
    positions_train[0],
    types_train[0],
    cells_train[0],
    method=DynamicsModel.calc_potential_energy
)

optimizer_def = flax.optim.Adam(learning_rate=1.)
optimizer_sum = optimizer_def.create(params)

@jax.jit
def train_step_sum(
    optimizer_sum, positions_batch, types_batch, cells_batch, energies_batch, forces_batch,
    learning_rate
):
    def calc_loss_sum(params):
        def contribution_energy(positions, types, cell, energy):
            pred = dynamics_model.apply(
                params,
                positions,
                types,
                cell,
                method=DynamicsModel.calc_potential_energy
            )
            delta = energy - pred
            return huber_loss(energy, pred, 0.02)
        
        def contribution_force(positions, types, cell, forces):
            pred = dynamics_model.apply(
                params,
                positions,
                types,
                cell,
                method=DynamicsModel.calc_forces
            )
            delta = forces - pred
            return log_cosh_force(delta).mean()

        energy_loss = jnp.mean(jax.vmap(jax.checkpoint(contribution_energy))
        (positions_batch, types_batch, cells_batch, energies_batch),
        axis=0) / positions_batch.shape[1]
        force_loss = jnp.mean(jax.vmap(jax.checkpoint(contribution_force))
        (positions_batch, types_batch, cells_batch, forces_batch), axis=0)
        
        energy_weight = 1
        force_weight = 100

        weighted_loss = energy_weight * energy_loss + force_weight * force_loss

        return weighted_loss

    calc_loss_and_grad_sum = jax.value_and_grad(calc_loss_sum)
    loss_val_sum, loss_grad_sum = calc_loss_and_grad_sum(optimizer_sum.target)
    optimizer_sum = optimizer_sum.apply_gradient(loss_grad_sum, learning_rate=learning_rate)
    return loss_val_sum, optimizer_sum


@jax.jit
def error_contributions(params, positions, types, cell, energies):
    pred = dynamics_model.apply(
        params,
        positions,
        types,
        cell,
        method=DynamicsModel.calc_potential_energy
    )
    mse_contribution = ((energies - pred) * (energies - pred))
    mae_contribution = jnp.fabs(energies - pred)
    return (mse_contribution, mae_contribution)


def eval_step(
    params, positions_batch, types_batch, cells_batch, energies_batch
):
    total_squared_error = 0.
    total_absolute_error = 0.
    for p, t, c, e in zip(
        positions_batch, types_batch, cells_batch, energies_batch
    ):
        squared_error, absolute_error = error_contributions(params, p, t, c, e)
        total_squared_error += squared_error
        total_absolute_error += absolute_error
    return (
        jnp.sqrt(total_squared_error.sum() / positions_batch.shape[0]) /
        positions_batch.shape[1], total_absolute_error.sum() /
        positions_batch.shape[0] / positions_batch.shape[1]
    )


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


PICKLE_FILE = f"NoElec.pickle"

min_mae = jnp.inf
for i in range(N_EPOCHS):
    rng, epoch_rng = jax.random.split(rng)
    optimizer_sum = train_epoch_sum(optimizer_sum, N_BATCH, i, epoch_rng)
    rmse, mae = eval_step(
        optimizer_sum.target, positions_validate, types_validate, cells_validate,
        energies_validate
    )

    # Save the state only if the validation MAE is minimal.
    dict_output = flax.serialization.to_state_dict(optimizer_sum)
    if mae < min_mae:
        with open(PICKLE_FILE, "wb") as f:
            print("- Saving the most recent state")
            pickle.dump(dict_output, f, protocol=5)
        min_mae = mae

pickle_file = "NoElec.pickle"
with open(pickle_file, "rb") as f:
    state_dict = pickle.load(f)
    optimizer = flax.serialization.from_state_dict(optimizer, state_dict)

@jax.jit
def calculate_energy(params, positions, types, cell):
    return dynamics_model.apply(
        params,
        positions,
        types,
        cell,
        method=DynamicsModel.calc_potential_energy
    )



@jax.jit
def error_contributions(params, positions, types, cell, energy, forces):
    pred_energy = dynamics_model.apply(
        params,
        positions,
        types,
        cell,
        method=DynamicsModel.calc_potential_energy
    )
    pred_forces = dynamics_model.apply(
        params, positions, types, cell, method=DynamicsModel.calc_forces
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


rmse_e_train, mae_e_train, rmse_f_train, mae_f_train = eval_step(
    optimizer.target, positions_train, types_train, cells_train, energies_train,
    forces_train
)

rmse_e_validate, mae_e_validate, rmse_f_validate, mae_f_validate = eval_step(
    optimizer.target, positions_validate, types_validate, cells_validate,
    energies_validate, forces_validate
)

hartree_to_mev = 27211

rmse_e_train = (rmse_e_train * hartree_to_mev) / n_atoms
mae_e_train = (mae_e_train * hartree_to_mev) / n_atoms
rmse_e_validate = (rmse_e_validate * hartree_to_mev) / n_atoms
mae_e_validate = (mae_e_validate * hartree_to_mev) / n_atoms
rmse_f_train = (rmse_f_train * hartree_to_mev) / cells
mae_f_train = (mae_f_train * hartree_to_mev) / cells
rmse_f_validate = (rmse_f_validate * hartree_to_mev) / cells
mae_f_validate = (mae_f_validate * hartree_to_mev) / cells

print("NoElec:")
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

# ReluNet train

random.seed(2024)

N_PAIR = 6

TRAINING_FRACTION = .9
R_CUT = 4
N_MAX = 4
EMBED_D = 2

N_EPOCHS = 501
N_BATCH = 8

LOG_COSH_PARAMETER_energy = 1e2
LOG_COSH_PARAMETER_force = 1e1


def create_log_cosh(parameter):
    """Return a differentiable implementation of the log cosh loss.

    Args:
        parameter: The positive parameter determining the transition between
            the linear and quadratic regimes.

    Returns:
        A float->float function to compute a contribution to the loss.

    Raises:
        ValueError: If parameter is not positive.
    """
    if parameter <= 0.:
        raise ValueError("parameter must be positive")

    def nruter(x):
        return (
            (jax.nn.softplus(2. * parameter * x) - jnp.log(2)) / parameter - x
        )

    return nruter

def huber_loss(y_true, y_pred, delta):
    residual = jnp.abs(y_true - y_pred)
    squared_loss = 0.5 * jnp.square(residual)
    linear_loss = delta * residual - 0.5 * jnp.square(delta)
    loss = jnp.where(residual < delta, squared_loss, linear_loss)
    loss = jnp.abs(loss)+ 1e-6 
    return jnp.mean(loss)

log_cosh_energy = create_log_cosh(LOG_COSH_PARAMETER_energy)
log_cosh_force = create_log_cosh(LOG_COSH_PARAMETER_force)

mass_cation = [
    14.007, 14.007, 12.011, 1.008, 12.011, 1.008, 12.011, 1.008, 12.011, 1.008, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 1.008]
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

print("- Reading the JSON file")
cells = []
positions = []
energies = []
forces = []
with open(
    (pathlib.Path(__file__).parent /
     "[bmim][bf4]_cleaned_3000.json").resolve(), "r"
) as json_f:
    for line in json_f:
        json_data = json.loads(line)
        cells.append(jnp.diag(jnp.array(json_data["Cell_size"])))
        positions.append(json_data["Positions"])
        energies.append(json_data["Energy"])
        forces.append(json_data["Forces"])
print("- Done")

n_configurations = len(positions)
types = [types for i in range(n_configurations)]

order = list(range(n_configurations))
random.shuffle(order)
cells = jnp.array([cells[i] for i in order])
positions = jnp.array([positions[i] for i in order])
energies = jnp.array([energies[i] for i in order])
types = jnp.array([types[i] for i in order])
forces = jnp.array([forces[i] for i in order])

print(f"- {n_configurations} configurations are available")

n_training = int(TRAINING_FRACTION * n_configurations)
n_validation = n_configurations - n_training
print(f"\t- {n_training} will be used for training")
print(f"\t- {n_validation} will be used for validation")
n_types = int(types.max()) + 1

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

# %%
average_dft_energy_train = energies_train.mean()

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


instance_code = "ReluNet"

core_model = CoreRelu([256, 128, 128, 64, 32, 16, 16, 16])
covalent_radii = {'C': 0.76, 'N': 0.71, 'H': 0.31, 'B': 0.82, 'F':0.64}
sigmas = jnp.asarray([covalent_radii[key] for key in type_dict])

core_model_electrostatic = CoreRelu([16, 16, 16])
electrostatic_model = ReluElectrostaticModel(
    core_model_electrostatic, n_types, sigmas
)
dynamics_model = ReluDynamicsModelWithCharges(
    n_types, EMBED_D, pipeline.process_data, core_model, electrostatic_model
)

rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
params = dynamics_model.init(
    init_rng,
    positions_train[0],
    types_train[0],
    cells_train[0],
    method=ReluDynamicsModelWithCharges.calc_forces
)

optimizer_def = flax.optim.Adam(learning_rate=1.)
optimizer = optimizer_def.create(params)


@jax.jit
def train_step_sum(
    optimizer_sum, positions_batch, types_batch, cells_batch, energies_batch, forces_batch,
    learning_rate
):
    def calc_loss_sum(params):
        def contribution_energy(positions, types, cell, energy):
            pred = dynamics_model.apply(
                params,
                positions,
                types,
                cell,
                method=ReluDynamicsModelWithCharges.calc_potential_energy
            )
            delta = energy - pred
            return huber_loss(energy, pred, 0.02)
        
        def contribution_force(positions, types, cell, forces):
            pred = dynamics_model.apply(
                params,
                positions,
                types,
                cell,
                method=ReluDynamicsModelWithCharges.calc_forces
            )
            delta = forces - pred
            return log_cosh_force(delta).mean()

        energy_loss = jnp.mean(jax.vmap(jax.checkpoint(contribution_energy))
        (positions_batch, types_batch, cells_batch, energies_batch),
        axis=0) / positions_batch.shape[1]
        force_loss = jnp.mean(jax.vmap(jax.checkpoint(contribution_force))
        (positions_batch, types_batch, cells_batch, forces_batch), axis=0)
        
        energy_weight = 1
        force_weight = 100

        weighted_loss = energy_weight * energy_loss + force_weight * force_loss

        return weighted_loss

    calc_loss_and_grad_sum = jax.value_and_grad(calc_loss_sum)
    loss_val_sum, loss_grad_sum = calc_loss_and_grad_sum(optimizer_sum.target)
    optimizer_sum = optimizer_sum.apply_gradient(loss_grad_sum, learning_rate=learning_rate)
    return loss_val_sum, optimizer_sum


@jax.jit
def error_contributions(params, positions, types, cell, forces):
    pred = dynamics_model.apply(
        params, positions, types, cell, method=ReluDynamicsModelWithCharges.calc_forces
    )
    mse_contribution = ((forces - pred) * (forces - pred)).mean()
    mae_contribution = jnp.fabs(forces - pred).mean()
    return (mse_contribution, mae_contribution)


def eval_step(params, positions_batch, types_batch, cells_batch, forces_batch):
    total_squared_error = 0.
    total_absolute_error = 0.
    for p, t, c, f in zip(
        positions_batch, types_batch, cells_batch, forces_batch
    ):
        squared_error, absolute_error = error_contributions(params, p, t, c, f)
        total_squared_error += squared_error
        total_absolute_error += absolute_error
    return (
        jnp.sqrt(total_squared_error.sum() / len(positions_batch)),
        total_absolute_error.sum() / len(positions_batch)
    )


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


PICKLE_FILE = f"ReluNet.pickle"

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
            print("- Saving the most recent state")
            pickle.dump(dict_output, f, protocol=5)
        min_mae = mae

pickle_file = "ReluNet.pickle"
with open(pickle_file, "rb") as f:
    state_dict = pickle.load(f)
    optimizer = flax.serialization.from_state_dict(optimizer, state_dict)


@jax.jit
def calculate_energy(params, positions, types, cell):
    return dynamics_model.apply(
        params,
        positions,
        types,
        cell,
        method=ReluDynamicsModelWithCharges.calc_potential_energy
    )


average_model_energy_train = jnp.mean(
    jnp.array(
        [
            calculate_energy(optimizer.target, p, t, c)
            for (p, t, c) in zip(positions_train, types_train, cells_train)
        ]
    )
)


@jax.jit
def error_contributions(params, positions, types, cell, energy, forces):
    pred_energy = dynamics_model.apply(
        params,
        positions,
        types,
        cell,
        method=ReluDynamicsModelWithCharges.calc_potential_energy
    )
    pred_forces = dynamics_model.apply(
        params, positions, types, cell, method=ReluDynamicsModelWithCharges.calc_forces
    )
    delta_energy = (
        (pred_energy - average_model_energy_train) -
        (energy - average_dft_energy_train)
    )
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


rmse_e_train, mae_e_train, rmse_f_train, mae_f_train = eval_step(
    optimizer.target, positions_train, types_train, cells_train, energies_train,
    forces_train
)

rmse_e_validate, mae_e_validate, rmse_f_validate, mae_f_validate = eval_step(
    optimizer.target, positions_validate, types_validate, cells_validate,
    energies_validate, forces_validate
)

hartree_to_mev = 27211

rmse_e_train = (rmse_e_train * hartree_to_mev) / n_atoms
mae_e_train = (mae_e_train * hartree_to_mev) / n_atoms
rmse_e_validate = (rmse_e_validate * hartree_to_mev) / n_atoms
mae_e_validate = (mae_e_validate * hartree_to_mev) / n_atoms
rmse_f_train = (rmse_f_train * hartree_to_mev) / cells
mae_f_train = (mae_f_train * hartree_to_mev) / cells
rmse_f_validate = (rmse_f_validate * hartree_to_mev) / cells
mae_f_validate = (mae_f_validate * hartree_to_mev) / cells

print("ReluNet:")
print(
    f"TRAIN:\n"
    f"\tRMSE = {rmse_e_train} meV/atom, {rmse_f_train} meV/Å\n"
    f"\tMAE = {mae_e_train} meV/atom, {mae_f_train} meV/Å"
)
print(
    f"VALIDATION:\n"
    f"\tRMSE = {rmse_e_validate} meV/atom, {rmse_f_validate} meV/Å\n"
    f"\tMAE = {mae_e_validate} meV/atom, {mae_f_validate} meV/Å"
)

# SameLoss train

random.seed(2024)

N_PAIR = 6

TRAINING_FRACTION = .9
R_CUT = 4
N_MAX = 4
EMBED_D = 2

N_EPOCHS = 501
N_BATCH = 8

LOG_COSH_PARAMETER_energy = 1e2
LOG_COSH_PARAMETER_force = 1e1


def create_log_cosh(parameter):
    """Return a differentiable implementation of the log cosh loss.

    Args:
        parameter: The positive parameter determining the transition between
            the linear and quadratic regimes.

    Returns:
        A float->float function to compute a contribution to the loss.

    Raises:
        ValueError: If parameter is not positive.
    """
    if parameter <= 0.:
        raise ValueError("parameter must be positive")

    def nruter(x):
        return (
            (jax.nn.softplus(2. * parameter * x) - jnp.log(2)) / parameter - x
        )

    return nruter


log_cosh_energy = create_log_cosh(LOG_COSH_PARAMETER_energy)
log_cosh_force = create_log_cosh(LOG_COSH_PARAMETER_force)


mass_cation = [
    14.007, 14.007, 12.011, 1.008, 12.011, 1.008, 12.011, 1.008, 12.011, 1.008, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 1.008]
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

print("- Reading the JSON file")
cells = []
positions = []
energies = []
forces = []
with open(
    (pathlib.Path(__file__).parent /
     "[bmim][bf4]_cleaned_3000.json").resolve(), "r"
) as json_f:
    for line in json_f:
        json_data = json.loads(line)
        cells.append(jnp.diag(jnp.array(json_data["Cell_size"])))
        positions.append(json_data["Positions"])
        energies.append(json_data["Energy"])
        forces.append(json_data["Forces"])
print("- Done")

n_configurations = len(positions)
types = [types for i in range(n_configurations)]

order = list(range(n_configurations))
random.shuffle(order)
cells = jnp.array([cells[i] for i in order])
positions = jnp.array([positions[i] for i in order])
energies = jnp.array([energies[i] for i in order])
types = jnp.array([types[i] for i in order])
forces = jnp.array([forces[i] for i in order])

print(f"- {n_configurations} configurations are available")

n_training = int(TRAINING_FRACTION * n_configurations)
n_validation = n_configurations - n_training
print(f"\t- {n_training} will be used for training")
print(f"\t- {n_validation} will be used for validation")
n_types = int(types.max()) + 1

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


instance_code = "SameLoss"

core_model = Core([256, 128, 128, 64, 32, 16, 16, 16])
covalent_radii = {'C': 0.76, 'N': 0.71, 'H': 0.31, 'B': 0.82, 'F':0.64}
sigmas = jnp.asarray([covalent_radii[key] for key in type_dict])

core_model_electrostatic = Core([16, 16, 16])
electrostatic_model = ElectrostaticModel(
    core_model_electrostatic, n_types, sigmas
)
dynamics_model = DynamicsModelWithCharges(
    n_types, EMBED_D, pipeline.process_data, core_model, electrostatic_model
)

rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
params = dynamics_model.init(
    init_rng,
    positions_train[0],
    types_train[0],
    cells_train[0],
    method=DynamicsModelWithCharges.calc_forces
)

optimizer_def = flax.optim.Adam(learning_rate=1.)
optimizer = optimizer_def.create(params)


@jax.jit
def train_step_sum(
    optimizer_sum, positions_batch, types_batch, cells_batch, energies_batch, forces_batch,
    learning_rate
):
    def calc_loss_sum(params):
        def contribution_energy(positions, types, cell, energy):
            pred = dynamics_model.apply(
                params,
                positions,
                types,
                cell,
                method=DynamicsModelWithCharges.calc_potential_energy
            )
            delta = energy - pred
            return log_cosh_energy(delta)
        
        def contribution_force(positions, types, cell, forces):
            pred = dynamics_model.apply(
                params,
                positions,
                types,
                cell,
                method=DynamicsModelWithCharges.calc_forces
            )
            delta = forces - pred
            return log_cosh_force(delta).mean()

        energy_loss = jnp.mean(jax.vmap(jax.checkpoint(contribution_energy))
        (positions_batch, types_batch, cells_batch, energies_batch),
        axis=0) / positions_batch.shape[1]
        force_loss = jnp.mean(jax.vmap(jax.checkpoint(contribution_force))
        (positions_batch, types_batch, cells_batch, forces_batch), axis=0)
        
        energy_weight = 1
        force_weight = 100

        weighted_loss = energy_weight * energy_loss + force_weight * force_loss

        return weighted_loss


    calc_loss_and_grad_sum = jax.value_and_grad(calc_loss_sum)
    loss_val_sum, loss_grad_sum = calc_loss_and_grad_sum(optimizer_sum.target)
    optimizer_sum = optimizer_sum.apply_gradient(loss_grad_sum, learning_rate=learning_rate)
    return loss_val_sum, optimizer_sum


@jax.jit
def error_contributions(params, positions, types, cell, forces):
    pred = dynamics_model.apply(
        params, positions, types, cell, method=DynamicsModelWithCharges.calc_forces
    )
    mse_contribution = ((forces - pred) * (forces - pred)).mean()
    mae_contribution = jnp.fabs(forces - pred).mean()
    return (mse_contribution, mae_contribution)


def eval_step(params, positions_batch, types_batch, cells_batch, forces_batch):
    total_squared_error = 0.
    total_absolute_error = 0.
    for p, t, c, f in zip(
        positions_batch, types_batch, cells_batch, forces_batch
    ):
        squared_error, absolute_error = error_contributions(params, p, t, c, f)
        total_squared_error += squared_error
        total_absolute_error += absolute_error
    return (
        jnp.sqrt(total_squared_error.sum() / len(positions_batch)),
        total_absolute_error.sum() / len(positions_batch)
    )


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


PICKLE_FILE = f"SameLoss.pickle"

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
            print("- Saving the most recent state")
            pickle.dump(dict_output, f, protocol=5)
        min_mae = mae

pickle_file = "SameLoss.pickle"
with open(pickle_file, "rb") as f:
    state_dict = pickle.load(f)
    optimizer = flax.serialization.from_state_dict(optimizer, state_dict)


@jax.jit
def calculate_energy(params, positions, types, cell):
    return dynamics_model.apply(
        params,
        positions,
        types,
        cell,
        method=DynamicsModelWithCharges.calc_potential_energy
    )


average_model_energy_train = jnp.mean(
    jnp.array(
        [
            calculate_energy(optimizer.target, p, t, c)
            for (p, t, c) in zip(positions_train, types_train, cells_train)
        ]
    )
)


@jax.jit
def error_contributions(params, positions, types, cell, energy, forces):
    pred_energy = dynamics_model.apply(
        params,
        positions,
        types,
        cell,
        method=DynamicsModelWithCharges.calc_potential_energy
    )
    pred_forces = dynamics_model.apply(
        params, positions, types, cell, method=DynamicsModelWithCharges.calc_forces
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


rmse_e_train, mae_e_train, rmse_f_train, mae_f_train = eval_step(
    optimizer.target, positions_train, types_train, cells_train, energies_train,
    forces_train
)

rmse_e_validate, mae_e_validate, rmse_f_validate, mae_f_validate = eval_step(
    optimizer.target, positions_validate, types_validate, cells_validate,
    energies_validate, forces_validate
)


hartree_to_mev = 27211

rmse_e_train = (rmse_e_train * hartree_to_mev) / n_atoms
mae_e_train = (mae_e_train * hartree_to_mev) / n_atoms
rmse_e_validate = (rmse_e_validate * hartree_to_mev) / n_atoms
mae_e_validate = (mae_e_validate * hartree_to_mev) / n_atoms
rmse_f_train = (rmse_f_train * hartree_to_mev) / cells
mae_f_train = (mae_f_train * hartree_to_mev) / cells
rmse_f_validate = (rmse_f_validate * hartree_to_mev) / cells
mae_f_validate = (mae_f_validate * hartree_to_mev) / cells

print("SameLoss:")
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

# SkipConnection train

random.seed(2024)

N_PAIR = 6

TRAINING_FRACTION = .9
R_CUT = 4
N_MAX = 4
EMBED_D = 2

N_EPOCHS = 501
N_BATCH = 8

LOG_COSH_PARAMETER_energy = 1e2
LOG_COSH_PARAMETER_force = 1e1


def create_log_cosh(parameter):
    """Return a differentiable implementation of the log cosh loss.

    Args:
        parameter: The positive parameter determining the transition between
            the linear and quadratic regimes.

    Returns:
        A float->float function to compute a contribution to the loss.

    Raises:
        ValueError: If parameter is not positive.
    """
    if parameter <= 0.:
        raise ValueError("parameter must be positive")

    def nruter(x):
        return (
            (jax.nn.softplus(2. * parameter * x) - jnp.log(2)) / parameter - x
        )

    return nruter
    
def mae_loss(y_true, y_pred):
    absolute_error = jnp.abs(y_true - y_pred)
    loss = jnp.mean(absolute_error)
    return loss

def mse_loss(y_true, y_pred):
    return jnp.mean(jnp.square(y_true - y_pred))    
    
def huber_loss(y_true, y_pred, delta):
    residual = jnp.abs(y_true - y_pred)
    squared_loss = 0.5 * jnp.square(residual)
    linear_loss = delta * residual - 0.5 * jnp.square(delta)
    loss = jnp.where(residual < delta, squared_loss, linear_loss)
    loss = jnp.abs(loss)+ 1e-6 
    return jnp.mean(loss)


log_cosh_energy = create_log_cosh(LOG_COSH_PARAMETER_energy)
log_cosh_force = create_log_cosh(LOG_COSH_PARAMETER_force)


mass_cation = [
    14.007, 14.007, 12.011, 1.008, 12.011, 1.008, 12.011, 1.008, 12.011, 1.008, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 12.011, 1.008, 1.008, 1.008]
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

print("- Reading the JSON file")
cells = []
positions = []
energies = []
forces = []
with open(
    (pathlib.Path(__file__).parent /
     "[bmim][bf4]_cleaned_3000.json").resolve(), "r"
) as json_f:
    for line in json_f:
        json_data = json.loads(line)
        cells.append(jnp.diag(jnp.array(json_data["Cell_size"])))
        positions.append(json_data["Positions"])
        energies.append(json_data["Energy"])
        forces.append(json_data["Forces"])
print("- Done")

n_configurations = len(positions)
types = [types for i in range(n_configurations)]

order = list(range(n_configurations))
random.shuffle(order)
cells = jnp.array([cells[i] for i in order])
positions = jnp.array([positions[i] for i in order])
energies = jnp.array([energies[i] for i in order])
types = jnp.array([types[i] for i in order])
forces = jnp.array([forces[i] for i in order])

print(f"- {n_configurations} configurations are available")

n_training = int(TRAINING_FRACTION * n_configurations)
n_validation = n_configurations - n_training
print(f"\t- {n_training} will be used for training")
print(f"\t- {n_validation} will be used for validation")
n_types = int(types.max()) + 1

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


core_model = ResNetCore([256, 128, 128, 64, 32, 16, 16, 16])
covalent_radii = {'C': 0.76, 'N': 0.71, 'H': 0.31, 'B': 0.82, 'F':0.64}
sigmas = jnp.asarray([covalent_radii[key] for key in type_dict])

core_model_electrostatic = Core([16, 16, 16])
electrostatic_model = ElectrostaticModel(
    core_model_electrostatic, n_types, sigmas
)
dynamics_model = DynamicsModelWithCharges(
    n_types, EMBED_D, pipeline.process_data, core_model, electrostatic_model
)

rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
params = dynamics_model.init(
    init_rng,
    positions_train[0],
    types_train[0],
    cells_train[0],
    method=DynamicsModelWithCharges.calc_forces
)


optimizer_def = flax.optim.Adam(learning_rate=1.)
optimizer = optimizer_def.create(params)


@jax.jit
def train_step_sum(
    optimizer_sum, positions_batch, types_batch, cells_batch, energies_batch, forces_batch,
    learning_rate
):
    def calc_loss_sum(params):
        def contribution_energy(positions, types, cell, energy):
            pred = dynamics_model.apply(
                params,
                positions,
                types,
                cell,
                method=DynamicsModelWithCharges.calc_potential_energy
            )
            delta = energy - pred
            return huber_loss(energy, pred, 0.02)
        
        def contribution_force(positions, types, cell, forces):
            pred = dynamics_model.apply(
                params,
                positions,
                types,
                cell,
                method=DynamicsModelWithCharges.calc_forces
            )
            delta = forces - pred
            return log_cosh_force(delta).mean()

        energy_loss = jnp.mean(jax.vmap(jax.checkpoint(contribution_energy))
        (positions_batch, types_batch, cells_batch, energies_batch),
        axis=0) / positions_batch.shape[1]
        force_loss = jnp.mean(jax.vmap(jax.checkpoint(contribution_force))
        (positions_batch, types_batch, cells_batch, forces_batch), axis=0)
        
        energy_weight = 1
        force_weight = 100

        weighted_loss = energy_weight * energy_loss + force_weight * force_loss

        return weighted_loss

    calc_loss_and_grad_sum = jax.value_and_grad(calc_loss_sum)
    loss_val_sum, loss_grad_sum = calc_loss_and_grad_sum(optimizer_sum.target)
    optimizer_sum = optimizer_sum.apply_gradient(loss_grad_sum, learning_rate=learning_rate)
    return loss_val_sum, optimizer_sum


@jax.jit
def error_contributions(params, positions, types, cell, forces):
    pred = dynamics_model.apply(
        params, positions, types, cell, method=DynamicsModelWithCharges.calc_forces
    )
    mse_contribution = ((forces - pred) * (forces - pred)).mean()
    mae_contribution = jnp.fabs(forces - pred).mean()
    return (mse_contribution, mae_contribution)


def eval_step(params, positions_batch, types_batch, cells_batch, forces_batch):
    total_squared_error = 0.
    total_absolute_error = 0.
    for p, t, c, f in zip(
        positions_batch, types_batch, cells_batch, forces_batch
    ):
        squared_error, absolute_error = error_contributions(params, p, t, c, f)
        total_squared_error += squared_error
        total_absolute_error += absolute_error
    return (
        jnp.sqrt(total_squared_error.sum() / len(positions_batch)),
        total_absolute_error.sum() / len(positions_batch)
    )


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


PICKLE_FILE = f"SkipConnection.pickle"

min_mae = jnp.inf
for i in range(N_EPOCHS):
    rng, epoch_rng = jax.random.split(rng)
    optimizer = train_epoch_sum(optimizer, N_BATCH, i, epoch_rng)
    rmse, mae = eval_step(
        optimizer.target, positions_validate, types_validate, cells_validate,
        forces_validate
    )

    # Save the state only if the validation MAE is minimal.
    dict_output = flax.serialization.to_state_dict(optimizer)
    if mae < min_mae:
        with open(PICKLE_FILE, "wb") as f:
            print("- Saving the most recent state")
            pickle.dump(dict_output, f, protocol=5)
        min_mae = mae

pickle_file = "SkipConnection.pickle"
with open(pickle_file, "rb") as f:
    state_dict = pickle.load(f)
    optimizer = flax.serialization.from_state_dict(optimizer, state_dict)


@jax.jit
def calculate_energy(params, positions, types, cell):
    return dynamics_model.apply(
        params,
        positions,
        types,
        cell,
        method=DynamicsModelWithCharges.calc_potential_energy
    )


average_model_energy_train = jnp.mean(
    jnp.array(
        [
            calculate_energy(optimizer.target, p, t, c)
            for (p, t, c) in zip(positions_train, types_train, cells_train)
        ]
    )
)


@jax.jit
def error_contributions(params, positions, types, cell, energy, forces):
    pred_energy = dynamics_model.apply(
        params,
        positions,
        types,
        cell,
        method=DynamicsModelWithCharges.calc_potential_energy
    )
    pred_forces = dynamics_model.apply(
        params, positions, types, cell, method=DynamicsModelWithCharges.calc_forces
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


rmse_e_train, mae_e_train, rmse_f_train, mae_f_train = eval_step(
    optimizer.target, positions_train, types_train, cells_train, energies_train,
    forces_train
)

rmse_e_validate, mae_e_validate, rmse_f_validate, mae_f_validate = eval_step(
    optimizer.target, positions_validate, types_validate, cells_validate,
    energies_validate, forces_validate
)


hartree_to_mev = 27211

rmse_e_train = (rmse_e_train * hartree_to_mev) / n_atoms
mae_e_train = (mae_e_train * hartree_to_mev) / n_atoms
rmse_e_validate = (rmse_e_validate * hartree_to_mev) / n_atoms
mae_e_validate = (mae_e_validate * hartree_to_mev) / n_atoms
rmse_f_train = (rmse_f_train * hartree_to_mev) / cells
mae_f_train = (mae_f_train * hartree_to_mev) / cells
rmse_f_validate = (rmse_f_validate * hartree_to_mev) / cells
mae_f_validate = (mae_f_validate * hartree_to_mev) / cells

print("SkipConnection:")
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