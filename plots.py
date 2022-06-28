import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import h5py
import warnings
import sys


def plot_observables(obs, name='twa_evol', keys=[]):
    fig, ax = plt.subplots(nrows=2, figsize=(12, 6))

    for (obs_key, obs_val) in obs.items():
        if obs_key == "t":
            continue
        if keys != []:
            if obs_key not in keys:
                continue
        if obs_key in ["N_m", "N_0", "N_p", "N"]:
            ax[0].plot(obs["t"], obs_val, label=obs_key)
        else:
            ax[1].plot(obs["t"], obs_val, label=obs_key)

    ax[0].grid()
    ax[0].legend()
    ax[0].set_xlabel(r'$t$')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_xlabel(r'$t$')

    fig.tight_layout()
    fig.savefig(name + '.pdf')


def animate(obs, mode='hist', delta_t=1e-1, every_nth=1, fps=15):
    # every_nth is the factor to which data points from the time series to plot
    fig = plt.figure(figsize=(6, 6))

    sx = obs["sx"]
    qyz = obs["qyz"]
    if mode == 'hist':
        warnings.warn('hist does not work as intended. Flips expectation values.')
        snapshots = np.array([np.histogram2d(sx, qyz, bins=(20, 20), range=[[-1, 1], [-1, 1]])[0]
                              for (sx, qyz) in zip(sx, qyz)])
        im = plt.imshow(snapshots[0].T, extent=(-1, 1, -1, 1), interpolation='none', aspect='auto')
    elif mode == 'scatter':
        scat = plt.scatter(sx[0], qyz[0], alpha=0.1, marker='x')
        plt.xlim([-40, 40])
        plt.ylim([-40, 40])

    plt.xlabel(r'$S_{x}/\sqrt{N}$')
    plt.ylabel(r'$Q_{yz}/\sqrt{N}$')
    txt = plt.text(0.8, 0.9, r'$t = 0.00$', fontsize=14, weight='bold', transform=fig.transFigure)
    plt.tight_layout()

    def animate_func(i):
        i = i * every_nth
        print(f'Animation progress: {i / obs["t"].shape[0]}')

        txt.set_text(rf'$t = {obs["t"][i]:.2f}$')
        if mode == 'hist':
            im.set_array(snapshots[i].T)
        elif mode == 'scatter':
            scat.set_offsets(np.array([sx[i], qyz[i]]).T)

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=obs["t"].shape[0] // every_nth,
        interval=1000 / fps,  # in ms
    )

    anim.save('twa_evol.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])


if __name__ == '__main__':
    wdir = './'
    obs = {}
    with h5py.File(wdir + 'data.hdf5', 'r') as f:
        for (key, val) in f.items():
            obs[key] = np.array(val)

    animate(obs, mode='scatter', shift=True)
    plt.show()
