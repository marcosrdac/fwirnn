from os.path import splitext, basename


def with_same_basename(filename, fns):
    name = splitext(filename)[0]
    for fn in fns:
        if splitext(fn)[0] == name:
            return fn
    return None
