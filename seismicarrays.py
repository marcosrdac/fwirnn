import numpy as np


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
    batch_srcsgns = np.asarray([[srcsgn] for i in range(7)])
    batch_srcposs = np.asarray([[(0, i * nx // 12, 0)]
                                for i in range(3, 3 + 7)])
    batch_recposs = np.asarray([[(0, x) for x in range(nx)] for i in range(7)])
    return batch_srcsgns, batch_srcposs, batch_recposs


def make_default_arrays(shape, srcsgn):
    nz, nx = shape
    batch_srcsgns = np.asarray([[srcsgn] for i in range(13)])
    batch_srcposs = np.asarray([[(0, i * nx // 24, 0)]
                                for i in range(6, 6 + 13)])
    batch_recposs = np.asarray([[(0, x) for x in range(nx)]
                                for i in range(13)])
    return batch_srcsgns, batch_srcposs, batch_recposs


if __name__ == '__main__':
    batch_srcsgns, batch_srcposs, batch_recposs = make_default_arrays(
        (70, 120), [1, 2, 3])
    print(len(batch_srcsgns), len(batch_srcposs), len(batch_recposs))
    print(batch_srcposs)
