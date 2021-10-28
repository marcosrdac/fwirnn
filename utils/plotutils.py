import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':
    from filters import seis_gain
else:
    from .filters import seis_gain


def plot_velocities(v_e,
                    loss_grad,
                    v_0,
                    v_true=None,
                    dx=1,
                    dz=1,
                    title=None,
                    figname=None,
                    figsize=(8, 5),
                    xlabel='x (km)',
                    ylabel='z (km)',
                    cbar_label='v (m/s)',
                    show=None,
                    dpi=300):
    fig, axes = plt.subplots(
        2,
        2,
        # sharex=True,
        # sharey=True,
        figsize=figsize,
        dpi=dpi)

    if title:
        fig.suptitle(title)

    extent = (0, dx * v_e.shape[1], dz * v_e.shape[0], 0)

    vmin, vmax = np.min(v_e), np.max(v_e)
    avmax = np.max(np.abs((vmin, vmax)))
    axes.flat[0].set_title('$v_e$')
    im = axes.flat[0].imshow(v_e, vmin=vmin, vmax=vmax, extent=extent)
    cbar = fig.colorbar(im, ax=axes.flat[0])
    cbar.set_label(cbar_label)

    axes.flat[1].set_title('$v$')
    im = axes.flat[1].imshow(v_true, vmin=vmin, vmax=vmax, extent=extent)
    cbar = fig.colorbar(im, ax=axes.flat[1])
    cbar.set_label(cbar_label)

    vmin, vmax = np.min(loss_grad), np.max(loss_grad)
    avmax = np.max(np.abs((vmin, vmax)))
    if not np.isfinite(avmax):
        avmax = 0
    axes.flat[3].set_title(r'$\nabla$ loss($v_e$)')
    abs_loss_grad = np.abs(loss_grad)
    loss_grad_abs_mean = np.mean(abs_loss_grad)
    loss_grad_abs_std = np.std(abs_loss_grad)
    loss_grad_thresh = loss_grad_abs_mean - .5 * loss_grad_abs_std
    loss_grad_thresh = loss_grad_thresh if loss_grad_thresh > 1e-8 else 1e-8
    im = axes.flat[3].imshow(loss_grad,
                             vmin=-avmax,
                             vmax=avmax,
                             extent=extent,
                             norm=mpl.colors.SymLogNorm(loss_grad_thresh,
                                                        base=10),
                             cmap='seismic')
    cbar = fig.colorbar(im, ax=axes.flat[3])
    cbar.set_label(cbar_label)

    delta = (v_e - v_0)
    vmin, vmax = np.min(delta), np.max(delta)
    avmax = np.max(np.abs((vmin, vmax)))
    axes.flat[2].set_title('$v_e - v_0$')
    im = axes.flat[2].imshow(delta,
                             vmin=-avmax,
                             vmax=avmax,
                             cmap='seismic',
                             extent=extent)
    cbar = fig.colorbar(im, ax=axes.flat[2])
    cbar.set_label(cbar_label)

    for ax in axes.flat:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()

    if figname:
        fig.savefig(figname)
    else:
        show = True if show is None else show

    fig.tight_layout()
    if show:
        plt.show()
    plt.close(fig)


def plot_seismograms(seis_i,
                     seis_t,
                     dt=0.015,
                     dx=1,
                     a=.8,
                     b=0.001,
                     xlabel='x (km)',
                     ylabel='t (s)',
                     title=None,
                     figname=None,
                     figsize=(8, 5),
                     seis_unit='Ampliude',
                     show=None,
                     dpi=300):
    fig, axes = plt.subplots(
        1,
        3,
        # sharex=True,
        # sharey=True,
        figsize=figsize,
        dpi=dpi)

    if title:
        fig.suptitle(title)

    extent = (0, dx * seis_i.shape[1], dt * seis_i.shape[0], 0)

    axes.flat[0].set_title('Predicted')
    seis_i_gain = seis_gain(seis_i, dt=dt, a=a, b=b)
    seis_t_gain = seis_gain(seis_t, dt=dt, a=a, b=b)
    delta_gain = seis_gain(seis_i - seis_t, dt=dt, a=a, b=b)
    vmax = np.max([np.max(np.abs(s)) for s in (
        seis_i_gain,
        seis_t_gain,
    )])

    if np.isfinite(vmax):
        vmin = -vmax
    else:
        vmin = vmax = None

    im = axes.flat[0].imshow(seis_i,
                             extent=extent,
                             vmin=vmin,
                             vmax=vmax,
                             cmap='gray',
                             aspect='auto')

    axes.flat[1].set_title('Ground-truth')
    im = axes.flat[1].imshow(seis_t,
                             extent=extent,
                             vmin=vmin,
                             vmax=vmax,
                             cmap='gray',
                             aspect='auto')

    axes.flat[2].set_title('Difference')
    im = axes.flat[2].imshow(delta_gain,
                             extent=extent,
                             vmin=vmin,
                             vmax=vmax,
                             cmap='gray',
                             aspect='auto')

    for ax in axes.flat:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()

    if figname:
        fig.savefig(figname)
    else:
        show = True if show is None else show

    fig.tight_layout()
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    img_orig = np.random.rand(200, 301) - .5
    img = gaussian_filter(img_orig, 10)

    dx = 8  # km
    dz = 2  # km

    # plot_velocities(img, img_orig, img, img_orig, dz=.3, dx=.5, show=True)
    plot_seismograms(img, img_orig, dt=0.015, dx=.3, show=True)
