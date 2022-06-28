import numpy as np
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
jax.config.update('jax_enable_x64', True)
from functools import partial

import plots
import util


def gauss_logProb(x):
    """
    logarithmic probability of a standard normal dist.
    """
    # factor of 2 because the (complex) distribution is essentially 2D
    return -0.5 * 2 * jnp.log(2 * jnp.pi) - 0.5 * jnp.abs(x)**2


def poisson_logProb(x, lam=1000):
    """
    logarithmic probability of a poisson dist with parameter lambda = 1000
    """
    return x * jnp.log(lam) - lam - util.ramanujan_logFac(x)


def beamsplit(sample, key):
    mixer = jnp.kron(jnp.array([[1, 1], [1, -1]]), jnp.eye(3)) / jnp.sqrt(2)

    noise = jax.random.normal(key, shape=(sample.shape[0], 2))
    noise = (noise[:, 0] + 1j * noise[:, 1]) / 2
    sample = jnp.concatenate((sample, noise))
    return mixer @ sample


def get_Sx(sample):
    Sx = 1 / jnp.sqrt(2) * jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    return jnp.real(jnp.conj(sample) @ Sx @ sample)


def get_Qyz(sample):
    Qyz = 1.j / jnp.sqrt(2) * jnp.array([[0, -1, 0], [1, 0, 1], [0, -1, 0]])
    return jnp.real(jnp.conj(sample) @ Qyz @ sample)


def float_modulo(t, dt, dt_obs, eps=1e-5):
    """
    for a given time t, checks whether t < n * dt_obs < t + dt for some integer n
    """
    n = round(t / dt_obs)
    return np.abs(t - n * dt_obs) < eps * dt


@jax.jit
def get_observables(samples, key, N):
    """
    Observable utility
    """
    keys = jax.random.split(key, num=samples[0].shape[0])
    samples_split = jax.vmap(beamsplit, in_axes=(0, 0))(samples[0], keys)

    sx = jax.vmap(get_Sx)(samples[0]) / jnp.sqrt(N)
    sx_mean = jnp.mean(sx)
    sx_std = jnp.std(sx)

    qyz = jax.vmap(get_Qyz)(samples[0]) / jnp.sqrt(N)
    qyz_mean = jnp.mean(qyz)
    qyz_std = jnp.std(qyz)

    sx_a = jax.vmap(get_Sx)(samples_split[:, :3]) / jnp.sqrt(N)
    sx_a_mean = jnp.mean(sx_a)
    sx_a_std = jnp.std(sx_a)

    qyz_a = jax.vmap(get_Qyz)(samples_split[:, :3]) / jnp.sqrt(N)
    qyz_a_mean = jnp.mean(qyz_a)
    qyz_a_std = jnp.std(qyz_a)

    sx_b = jax.vmap(get_Sx)(samples_split[:, 3:]) / jnp.sqrt(N)
    sx_b_mean = jnp.mean(sx_b)
    sx_b_std = jnp.std(sx_b)

    qyz_b = jax.vmap(get_Qyz)(samples_split[:, 3:]) / jnp.sqrt(N)
    qyz_b_mean = jnp.mean(qyz_b)
    qyz_b_std = jnp.std(qyz_b)

    N = jnp.mean(jnp.abs(samples[0])**2, axis=0) / N

    return {"sx": sx, "sx_mean": sx_mean, "sx_std": sx_std,
            "qyz": qyz, "qyz_mean": qyz_mean, "qyz_std": qyz_std,
            "sx_a": sx_a, "sx_a_mean": sx_a_mean, "sx_a_std": sx_a_std,
            "qyz_a": qyz_a, "qyz_a_mean": qyz_a_mean, "qyz_a_std": qyz_a_std,
            "sx_b": sx_b, "sx_b_mean": sx_b_mean, "sx_b_std": sx_b_std,
            "qyz_b": qyz_b, "qyz_b_mean": qyz_b_mean, "qyz_b_std": qyz_b_std,
            "N_m": N[0], "N_0": N[1], "N_p": N[2]}


def hamiltonian(a_conj, a, params):
    """
    Returns the Hamiltonian evaluated at a and a_conj for parameters specified by params.
    """
    N = a * a_conj

    H = (params["g"] * (a_conj[0] * a_conj[2] * a[1]**2 + a[0] * a[2] * a_conj[1]**2
                        + (N[1] - 0.5) * (N[2] + N[0] - 1)
                        + 0.5 * (N[2] - N[0])**2)
         + params["q"] * (N[2] + N[0]))
    return jnp.real(H)


def derivative(ham_grad, probs, a, a_conj, params, dt):
    grad = ham_grad(a_conj, a, params)

    def f(a_tmp): return a_tmp + dt * ham_grad(jnp.conj(a_tmp), a_tmp, params)
    prob_grad = jax.jvp(f, (a,), (jnp.complex128(jnp.exp(probs))[0],))

    return grad, prob_grad


def integrate_single_coord(sample_coord, sample_prob, flow, params, dt):
    """
    4th-order RK-scheme
    """
    k1 = -1j * flow(jnp.conj(sample_coord), sample_coord, params)
    k2 = -1j * flow(jnp.conj(sample_coord) + dt * 0.5 * jnp.conj(k1), sample_coord + dt * 0.5 * k1, params)
    k3 = -1j * flow(jnp.conj(sample_coord) + dt * 0.5 * jnp.conj(k2), sample_coord + dt * 0.5 * k2, params)
    k4 = -1j * flow(jnp.conj(sample_coord) + dt * jnp.conj(k3), sample_coord + dt * k3, params)
    return [sample_coord + dt * (k1 + 2. * k2 + 2. * k3 + k4) / 6., sample_prob]


@partial(jax.jit, static_argnums=(1,))
def integrate(samples, flow, params, dt):
    """
    integrate all points by a single time step dt
    """
    samples = jax.vmap(integrate_single_coord, in_axes=(0, 0, None, None, None))(samples[0], samples[1], flow, params, dt)
    return samples


def stepper(t_end, samples, params, dt=1e-3, dt_obs=1e-1):
    """
    Integrates sample positions from time t=0 to t=t_end while recording observables.
    """
    t = 0
    key = jax.random.PRNGKey(0)

    """
    The evolution of a 3 component vector a is given by the Poisson bracket:
    i\partial_t a = {a, H} = \frac{\partial H}{\partial a^*}
    """
    flow = jax.grad(hamiltonian, argnums=0)

    obs_dict = {"t": []}
    while t < t_end:
        if float_modulo(t, dt, dt_obs):
            key, key_to_use = jax.random.split(key)
            obs_dict["t"].append(t)
            observables = get_observables(samples, key_to_use, params["N"])
            for (obs_key, obs_val) in observables.items():
                try:
                    obs_dict[obs_key].append(obs_val)
                except:
                    obs_dict[obs_key] = []
                    obs_dict[obs_key].append(obs_val)

        samples = integrate(samples, flow, params, dt)
        t = t + dt
        print(f"> t = {t}")
        print(f">\t sx_mean = {obs_dict['sx_a_mean'][-1]}")
        print(f">\t sx_std = {obs_dict['sx_a_std'][-1]}")
        print(f">\t qyz_mean = {obs_dict['qyz_a_mean'][-1]}")
        print(f">\t qyz_std = {obs_dict['qyz_a_std'][-1]}")

    for (key, val) in obs_dict.items():
        obs_dict[key] = np.array(val)
    return obs_dict


if __name__ == "__main__":
    params = {"N": 1000}
    params["g"] = -1 / params["N"]
    params["q"] = -params["g"] * (params["N"] - 0.5)
    keys = jax.random.split(jax.random.PRNGKey(1))
    N_samples = int(2 * 1e5)
    t_end = 1e1
    dt = 1e-2
    dt_obs = 1e-1
    wdir = './'

    sample_coords = jax.random.normal(keys[0], shape=(N_samples, 3, 2))
    sample_coords = (sample_coords[..., 0] + 1j * sample_coords[..., 1]) / 2
    # sample_coords = sample_coords.at[:, 1].set(jnp.sqrt(jax.random.poisson(keys[1], lam=params["N"], shape=(N_samples,))))
    sample_coords = sample_coords.at[:, 1].set(jnp.sqrt(params["N"]))
    sample_probs = gauss_logProb(sample_coords)
    sample_probs = sample_probs.at[:, 1].set(poisson_logProb(sample_coords[:, 1]**2, lam=params["N"]))
    samples = [sample_coords, sample_probs]

    beamsplit(samples[0][0], jax.random.PRNGKey(1))

    obs = stepper(t_end, samples, params, dt=dt, dt_obs=dt_obs)
    util.store_data(wdir, obs)
    plots.plot_observables(obs, name='twa', keys=["sx_mean", "sx_std", "qyz_mean", "qyz_std", "N_m", "N_0", "N_p"])
    plots.animate(obs, mode='scatter')
    plt.show()
