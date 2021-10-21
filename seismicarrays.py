import numpy as np
from numbers import Number
from typing import Union, Sequence


def make_default_array(shape, signal):
    nz, nx = shape

    srcsgns = np.asarray([
        signal,  # shot 0
    ])
    srcposs = np.asarray([
        (0, nx // 2, 0),  # shot 0
    ])
    recposs = np.asarray([(0, x) for x in range(nx)])
    return srcsgns, srcposs, recposs


def make_default_arrays_long(shape, signal):
    nz, nx = shape
    batch_srcsgns = np.asarray([
        [  # batch 0
            signal,  # shot 0
        ],
        [  # batch 1
            signal,  # shot 0
        ],
        [  # batch 2
            signal,  # shot 0
        ],
    ])

    batch_srcposs = np.asarray([
        [  # batch 0
            (0, 0, 0),  # shot 0
        ],
        [  # batch 1
            (0, nx // 2, 0),  # shot 0
        ],
        [  # batch 2
            (0, nx - 1, 0),  # shot 0
        ],
    ])

    batch_recposs = np.asarray([
        [  # batch 0
            (50, nx // 2 - 1),  # shot 0
            (50, nx // 2 + 0),  # shot 0
            (50, nx // 2 + 1),  # shot 0
        ],
        [  # batch 1
            (50, nx // 2 - 1),  # shot 0
            (50, nx // 2 + 0),  # shot 0
            (50, nx // 2 + 1),  # shot 0
        ],
        [  # batch 2
            (50, nx // 2 - 1),  # shot 0
            (50, nx // 2 + 0),  # shot 0
            (50, nx // 2 + 1),  # shot 0
        ],
    ])

    batch_recposs = np.asarray([
        [  # batch 0
            (20, 0 + i) for i in range(-50, 51)
        ],
        [  # batch 1
            (20, nx // 2 + i) for i in range(-50, 51)
        ],
        [  # batch 2
            (20, nx - 1 + i) for i in range(-50, 51)
        ],
    ])

    return batch_srcsgns, batch_srcposs, batch_recposs


def make_default_arrays(shape, srcsgn):
    nz, nx = shape
    batch_srcsgns = np.asarray([[srcsgn] for i in range(13)])
    batch_srcposs = np.asarray([[(0, i * nx // 24, 0)]
                                for i in range(6, 6 + 13)])
    batch_recposs = np.asarray([[(0, x) for x in range(nx)]
                                for i in range(13)])
    return batch_srcsgns, batch_srcposs, batch_recposs


def parse_geometry(geometry: str,
                   array=None,
                   ns=1,
                   ss=1,
                   rr=1,
                   l_rr=None,
                   r_rr=None):
    def to_int_or_float(x):
        try:
            return int(x)
        except ValueError:
            return float(x)

    geometry = [to_int_or_float(x) for x in geometry.split('-')]
    assert len(geometry) >= 3
    if array:
        if array in ['endon-left', 'split-spread']:
            idx_0 = 2
        else:
            idx_0 = 0
    else:
        if geometry[0] == 0:
            array = 'endon-right'
            idx_0 = 0
        else:
            idx_0 = 2
            if len(geometry) == 3:
                array = 'endon-left'
            elif len(geometry) == 5:
                array = 'split-spread'
            else:
                raise ValueError('Geometry not understood')

    geometry = {idx - idx_0: x for idx, x in enumerate(geometry)}

    l_maxoff = geometry.get(-2, 0)
    l_minoff = geometry.get(-1, 0)
    r_minoff = geometry.get(1, 0)
    r_maxoff = geometry.get(2, 0)
    l_max_span = l_maxoff - l_minoff
    r_max_span = r_maxoff - r_minoff
    l_rr = l_rr or rr
    r_rr = r_rr or rr

    l_span_to_rr = int(l_max_span / l_rr)
    r_span_to_rr = int(r_max_span / r_rr)

    l_span = l_rr * l_span_to_rr
    r_span = r_rr * r_span_to_rr

    l_nr = l_span_to_rr + 1 if l_max_span else 0
    r_nr = r_span_to_rr + 1 if r_max_span else 0

    l_maxoff_corr = l_minoff + l_span
    r_maxoff_corr = r_minoff + r_span

    sx_0 = l_maxoff_corr

    return {
        'array': array,
        'src-src': ss,
        'sx_0': sx_0,
        'endon-left': {
            'min-offset': l_minoff,
            'max-offset': l_maxoff_corr,
            'rec-rec': l_rr,
            'n_rec': l_nr,
        },
        'endon-right': {
            'min-offset': r_minoff,
            'max-offset': r_maxoff_corr,
            'rec-rec': r_rr,
            'n_rec': r_nr,
        }
    }


def calc_max_nshots(parsed_geometry, nx, dx=1, dz=1):
    ss = parsed_geometry['src-src']
    l_maxoff = parsed_geometry['endon-left']['max-offset']
    r_maxoff = parsed_geometry['endon-right']['max-offset']
    interval = nx * dx - r_maxoff - l_maxoff
    return int(np.ceil(interval / ss))


def make_array(
        parsed_geometry=None,
        srcsgn=None,
        dx=1,
        dz=None,
        nx=None,
        nz=None,  # needed?
        nshots=None,
        sz=0,
        rz=0,
        all_recs=False,
        **geometry_kwargs):

    if not parsed_geometry:
        parsed_geometry = parse_geometry(**geometry_kwargs)

    dz = dz or dx

    sx_0 = parsed_geometry['sx_0']
    ss = parsed_geometry['src-src']
    l_minoff = parsed_geometry['endon-left']['min-offset']
    r_minoff = parsed_geometry['endon-right']['min-offset']
    l_rr = parsed_geometry['endon-left']['rec-rec']
    l_nr = parsed_geometry['endon-left']['n_rec']
    r_rr = parsed_geometry['endon-right']['rec-rec']
    r_nr = parsed_geometry['endon-right']['n_rec']

    true_srccrds_0 = np.asarray([sz, sx_0, 0])

    if nshots is None:
        if nx:
            nshots = calc_max_nshots(parsed_geometry, nx, dx=dx, dz=dz)
        else:
            nshots = 1

    true_srccrds = [[tuple(true_srccrds_0 + np.asarray([0, s * ss, 0]))]
                    for s in range(nshots)]

    grid_srccrds = [[(int(round(sz / dz)), int(round(sx / dx)), int(st))
                     for sz, sx, st in srccrds] for srccrds in true_srccrds]

    if all_recs:
        grid_reccrds = [[
            (rz, rx) for rx in range(nx)
            if not -l_minoff / dx < rx - srccrds[0][1] < r_minoff / dx
        ] for srccrds in grid_srccrds]
        true_reccrds = [[(rz * dz, rx * dx) for rz, rx in reccrd]
                        for reccrd in grid_reccrds]
    else:
        l_rx = l_rr * np.arange(0, l_nr)
        r_rx = r_rr * np.arange(0, r_nr) + r_minoff + sx_0
        rx_0 = np.concatenate([l_rx, r_rx])

        true_reccrds_0 = np.empty((len(rx_0), 2))
        true_reccrds_0[:, 0] = rz
        true_reccrds_0[:, 1] = rx_0

        true_reccrds = [
            tuple((true_reccrds_0 + np.asarray([0, s * ss])).tolist())
            for s in range(nshots)
        ]
        grid_reccrds = [[(int(round(sz / dz)), int(round(sx / dx)))
                         for sz, sx in crds] for crds in true_reccrds]

    if srcsgn is not None:
        srcsgns = np.asarray([[srcsgn] for s in range(nshots)])
    else:
        srcsgns = [[[]] for s in range(nshots)]

    return srcsgns, grid_srccrds, grid_reccrds, true_srccrds, true_reccrds


def test_make_array(parsed_geometry=None,
                    v=None,
                    shape=None,
                    grid=True,
                    dz=None,
                    dx=None,
                    all_recs=False,
                    aspect='auto',
                    **geometry_kwargs):

    if not parsed_geometry:
        parsed_geometry = parse_geometry(**geometry_kwargs)

    if shape is None:
        if v is None:
            shape = nz, nx = 12, 21
        else:
            shape = nz, nx = v.shape

    if v is None:
        v = np.random.random(nz * nx).reshape(shape)

    dx = dx or 10.
    dz = dz or dx

    srcsgns, grid_srccrds, grid_reccrds, true_srccrds, true_reccrds = make_array(
        parsed_geometry,
        signal,
        nshots=None,
        nx=nx,
        dx=dx,
        all_recs=all_recs,
    )

    if grid:
        srccrds = grid_srccrds
        reccrds = grid_reccrds
        extent = (0, nx, nz, 0)
        # xticklabels = np.arange(0, nx)
        # yticklabels = np.arange(nz-1, -1, -1)
        pdz = 1
    else:
        srccrds = true_srccrds
        reccrds = true_reccrds
        extent = (0, nx * dx, nz * dz, 0)
        # xticklabels = dx * np.arange(0, nx)
        # yticklabels = dz * np.arange(nz-1, -1, -1)
        pdz = dz

    reccrds_x = [[crd[1] + .5 * pdz for crd in crds] for crds in reccrds]
    reccrds_z = [[crd[0] + (i + .5) * pdz for crd in crds]
                 for i, crds in enumerate(reccrds)]

    srccrds_x = [[crd[1] + .5 * pdz for crd in crds] for crds in srccrds]
    srccrds_z = [[crd[0] + (i + .5) * pdz for crd in crds]
                 for i, crds in enumerate(srccrds)]

    z_max = int(pdz / 2 + np.max([np.max(srccrds_z), np.max(reccrds_z)]))

    plt.imshow(v, extent=extent, cmap='gray', aspect=aspect)
    plt.ylim(z_max, 0)
    for x, z in zip(reccrds_x, reccrds_z):
        plt.scatter(x, z, s=200, c='green', marker='v')

    for x, z in zip(srccrds_x, srccrds_z):
        plt.scatter(x, z, s=200, c='red', marker='*')
    plt.show()


if __name__ == '__main__':
    from os.path import isfile
    from utils.discarrays import discarray, todiscarray
    import matplotlib.pyplot as plt

    marmousi_path = '/home/marcosrdac/cld/Dropbox/home/pro/0/awm/awm2d/models/marmousi.bin'

    signal = [1, 2]

    parsed_geometry = parse_geometry(
        geometry='50.-30.-0-30.-50.',
        ss=10.,
        rr=20.,
    )

    print(parsed_geometry)

    test_make_array(parsed_geometry, all_recs=False, grid=False)

    if isfile(marmousi_path):
        v = discarray(marmousi_path, mode='r', order='F',
                      dtype=float).astype(np.float32)

        test_make_array(geometry='2600.0-520.0-0-520-2600',
                        ss=520,
                        rr=260,
                        v=v,
                        dx=260,
                        dz=86,
                        all_recs=False,
                        grid=False)
