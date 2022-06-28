import os
import h5py
import numpy as np
import jax.numpy as jnp
import collections
import scipy.special


def ramanujan_logFac(k):
    """
    approximate log(k!) by first answer of https://math.stackexchange.com/questions/138194/approximating-log-of-factorial
    """
    return k * jnp.log(k) - k + jnp.log(k * (1 + 4 * k * (1 + 2 * k))) / 6 + jnp.log(jnp.pi) / 2


def stirlingSeries_logFac(k, cut=9):
    """
    approximate log(k!) by https://en.wikipedia.org/wiki/Stirling%27s_approximation#:~:text=Writing%20Stirling%27s%20series%20in%20the%20form
    """
    if isinstance(k, collections.abc.Iterable):
        res = np.log(scipy.special.factorial(k[:cut]))
        k = k[cut:]
        res = np.append(res, k * np.log(k) - k + 0.5 * np.log(2 * np.pi * k) + 1 / (12 * k) - 1 / (360 * k**3))  # + 1 / (1260 * k**5))

        mask = np.isinf(res)
        res[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), res[~mask])
    else:
        res = k * jnp.log(k) - k + 0.5 * jnp.log(2 * jnp.pi * k) + 1 / (12 * k) - 1 / (360 * k**3) + 1 / (1260 * k**5)
    return res


def store_data(wdir, obs, name='data.hdf5'):
    try:
        os.makedirs(wdir)
    except OSError:
        print("Creation of the directory %s failed" % wdir)

    with h5py.File(wdir + 'data.hdf5', 'w') as f:
        for (key, val) in obs.items():
            f.create_dataset(key, data=val)
