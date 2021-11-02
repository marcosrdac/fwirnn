import numpy as np
import pickle
import yaml


def discarray(filename, mode='r', dtype=float, shape=None, order='C'):
    file_mode = f'{mode[0]}b{mode[1:]}'
    if not isinstance(shape, tuple):
        shape = (shape, )
    with open(filename, file_mode) as io:
        if 'w' in mode:
            ndims_shape = np.array((len(shape), *shape), dtype=np.int64)
            ndims_shape.tofile(io)
        if 'r' in mode:
            ndims = np.fromfile(io, dtype=np.int64, count=1)[0]
            shape = tuple(np.fromfile(io, dtype=np.int64, count=ndims))
        offset = io.tell()
        arr = np.memmap(io,
                        mode=mode,
                        dtype=dtype,
                        shape=shape,
                        offset=offset,
                        order=order)
    return arr


def discarray_to(filename, arr, order='C'):
    disk_arr = discarray(filename,
                         'w+',
                         dtype=arr.dtype,
                         shape=arr.shape,
                         order=order)
    disk_arr[...] = arr[...]


def pickle_to(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def unpickle_from(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def yaml_to(path, data):
    with open(path, 'w') as f:
        f.write(yaml.dump(data))


def unyaml_from(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


yaml.safe_load(yaml.dump({'a': 3}))

if __name__ == '__main__':
    from os import remove

    filepath = '/tmp/myfile.pkl'
    var = 'A string object.'

    pickle_to(filepath, var)
    print(unpickle_from(filepath))

    yaml_to(filepath, var)
    print(unyaml_from(filepath))

    remove(filepath)
