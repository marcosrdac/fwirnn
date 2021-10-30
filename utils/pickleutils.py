import pickle


def pickle_to(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def unpickle_from(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    from os import remove

    filepath = '/tmp/myfile.pkl'
    var = 'A string object.'

    pickle_to(filepath, var)
    print(unpickle_from(filepath))

    remove(filepath)
