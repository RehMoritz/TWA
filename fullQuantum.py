import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import warnings
import sys

import util
import plots


def coherent_state(alpha, cutoff):
    k = jnp.arange(cutoff)
    state = jnp.exp(jnp.log(alpha) * k - 0.5 * util.stirlingSeries_logFac(k) - jnp.abs(alpha)**2 / 2)
    return state


def get_observables(state, actions, N):
    obs = {}
    obs["sx2"] = np.sqrt(np.real(np.sum(np.conj(state) * apply_Sx2(state, actions)))) / np.sqrt(N)
    obs["qyz2"] = np.sqrt(np.real(np.sum(np.conj(state) * apply_Qyz2(state, actions)))) / np.sqrt(N)
    obs["N_0"] = np.real(np.sum(np.conj(state) * actions["counters"]["counter_0"] * state)) / N
    obs["N_p"] = np.real(np.sum(np.conj(state) * actions["counters"]["counter_1"] * state)) / N
    obs["N"] = obs["N_0"] + 2 * obs["N_p"]
    return obs


def get_Sx2(cutoffs, counters):
    effect_0 = np.sqrt(np.arange(cutoffs[0]) + 1) * np.sqrt(np.arange(cutoffs[0]) + 2)
    effect_1 = np.arange(cutoffs[1])
    effect = effect_0[:, None] * effect_1[None, :]

    effect_0 = np.sqrt(np.arange(cutoffs[0])) * np.nan_to_num(np.sqrt(np.arange(cutoffs[0]) - 1))
    effect_1 = np.arange(cutoffs[1]) + 1
    effect_conj = effect_0[:, None] * effect_1[None, :]

    N = (counters["counter_1"] * (1 + counters["counter_0"]) +
         (1 + counters["counter_1"]) * counters["counter_0"])

    return {"effect": effect, "effect_conj": effect_conj, "N": N}


def apply_Sx2(state, actions):
    state_roll = np.roll(state, (-2, 1), axis=(0, 1))
    state_roll_conj = np.roll(state, (2, -1), axis=(0, 1))

    state_roll[-2:] = 0
    state_roll[:, 0] = 0
    state_roll_conj[:, -1:] = 0
    state_roll_conj[:2] = 0

    new_state = actions["sx2"]["effect"] * state_roll
    new_state += actions["sx2"]["effect_conj"] * state_roll_conj
    new_state += actions["sx2"]["N"] * state
    return new_state


def get_Qyz2(cutoffs, counters):
    effect_0 = np.sqrt(np.arange(cutoffs[0]) + 1) * np.sqrt(np.arange(cutoffs[0]) + 2)
    effect_1 = np.arange(cutoffs[1])
    effect = - effect_0[:, None] * effect_1[None, :]

    effect_0 = np.sqrt(np.arange(cutoffs[0])) * np.nan_to_num(np.sqrt(np.arange(cutoffs[0]) - 1))
    effect_1 = np.arange(cutoffs[1]) + 1
    effect_conj = - effect_0[:, None] * effect_1[None, :]

    N = (counters["counter_1"] * (1 + counters["counter_0"]) +
         (1 + counters["counter_1"]) * counters["counter_0"])

    return {"effect": effect, "effect_conj": effect_conj, "N": N}


def apply_Qyz2(state, actions):
    state_roll = np.roll(state, (-2, 1), axis=(0, 1))
    state_roll_conj = np.roll(state, (2, -1), axis=(0, 1))

    state_roll[-2:] = 0
    state_roll[:, 0] = 0
    state_roll_conj[:, -1:] = 0
    state_roll_conj[:2] = 0

    new_state = actions["qyz2"]["effect"] * state_roll
    new_state += actions["qyz2"]["effect_conj"] * state_roll_conj
    new_state += actions["qyz2"]["N"] * state

    return new_state


def get_counters(cutoffs):
    """
    Returns the action of the counters of the two modes on the state.
    """
    counter_0 = np.arange(cutoffs[0])[:, None] * np.ones(cutoffs)
    counter_1 = np.arange(cutoffs[1])[None, :] * np.ones(cutoffs)
    return {"counter_0": counter_0, "counter_1": counter_1}


def get_actions(cutoffs, couplings):
    """
    Get all actions on the fock states
    """
    actions = {}

    interactions = get_interaction_matrices(cutoffs)
    actions["interact"] = couplings["g"] * interactions["effect"]
    actions["interact_conj"] = couplings["g"] * interactions["effect_conj"]

    counters = get_counters(cutoffs)
    actions["counters"] = counters
    actions["N"] = (couplings["g"] * (counters["counter_0"] - 0.5) * (2 * counters["counter_1"] - 1)
                    + couplings["q"] * (2 * counters["counter_1"] - 1))

    actions["sx2"] = get_Sx2(cutoffs, counters)
    actions["qyz2"] = get_Qyz2(cutoffs, counters)

    return actions


def get_interaction_matrices(cutoffs):
    """
    Returns the interaction matrices for fixed cutoffs.
    """
    effect_0 = np.sqrt(np.arange(cutoffs[0]) + 1) * np.sqrt(np.arange(cutoffs[0]) + 2)
    effect_1 = np.arange(cutoffs[1])
    effect = effect_0[:, None] * effect_1[None, :]

    effect_0 = np.sqrt(np.arange(cutoffs[0])) * np.nan_to_num(np.sqrt(np.arange(cutoffs[0]) - 1))
    effect_1 = np.arange(cutoffs[1]) + 1
    effect_conj = effect_0[:, None] * effect_1[None, :]

    return {"effect": effect, "effect_conj": effect_conj}


def mode_interact(state, effect, effect_conj):
    """
    Args:
        * state
        * effect
        * effect_conj

    Returns: time_derivative
    """

    state_roll = np.roll(state, (-2, 1), axis=(0, 1))
    state_roll_conj = np.roll(state, (2, -1), axis=(0, 1))

    state_roll[-2:] = 0
    state_roll[:, 0] = 0
    state_roll_conj[:, -1:] = 0
    state_roll_conj[:2] = 0

    time_der = effect * state_roll
    time_der += effect_conj * state_roll_conj
    return time_der


def state_time_derivative(state, actions):
    state_der = mode_interact(state, actions["interact"], actions["interact_conj"])
    state_der += actions["N"] * state
    return state_der


def step(state, actions, dt):
    """
    4th-order RK scheme.
    """
    k1 = -1j * state_time_derivative(state, actions)
    k2 = -1j * state_time_derivative(state + dt * 0.5 * k1, actions)
    k3 = -1j * state_time_derivative(state + dt * 0.5 * k2, actions)
    k4 = -1j * state_time_derivative(state + dt * k3, actions)
    return state + dt * (k1 + 2. * k2 + 2. * k3 + k4) / 6.


if __name__ == '__main__':
    state_mode = ['Fock', 'coherent'][0]
    N = 100
    couplings = {"g": - 1 / N, "q": 1 - 1 / (2 * N)}
    t_end = 10
    t = 0
    dt = 1e-3

    cutoffs = (101, 101)  # zero mode, plus 1 mode
    state = np.zeros(cutoffs)
    if state_mode == 'Fock':
        state[N, 0] = 1
    elif state_mode == 'coherent':
        state[:, 0] = coherent_state(np.sqrt(N), cutoffs[0])
    else:
        warnings.warn('State not recognized.')

    actions = get_actions(cutoffs, couplings)

    # init_state = state
    # print(state)
    # state = apply_Sx2(state, actions)
    # print(state)
    # print(np.sum(np.conj(init_state) * state))
    # exit()

    observables = {}
    while t < t_end:
        print(f"> t = {t}")
        obs = get_observables(state, actions, N)
        obs["t"] = t
        for key, val in obs.items():
            try:
                observables[key].append(val)
            except:
                observables[key] = [val]
            print(f"> \t{key} = {observables[key][-1]}")

        state = step(state, actions, dt)

        t = t + dt

    plots.plot_observables(observables, name='fullQ')
    plt.show()
