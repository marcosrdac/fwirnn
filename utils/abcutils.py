class AccumulatingDict(dict):
    def __iadd__(self, other):
        assert isinstance(other, dict)
        for key in other:
            self[key] = self[key] + other[key] if key in self else other[key]
        return self

    def __isub__(self, other):
        assert isinstance(other, dict)
        for key in other:
            self[key] = self[key] - other[key] if key in self else -other[key]
        return self

    def __add__(self, other):
        assert isinstance(other, dict)
        for key in other:
            new = {**self}
            new[key] = self[key] + other[key] if key in self else other[key]
        return new

    def __sub__(self, other):
        assert isinstance(other, dict)
        for key in other:
            new = {**self}
            new[key] = self[key] - other[key] if key in self else -other[key]
        return new

    def append(self, other):
        for key in other:
            if key in self:
                self[key].append(other[key])
            else:
                self[key] = [other[key]]


if __name__ == '__main__':

    a = AccumulatingDict({1: 2, 2: 4})
    b = {'a': 3, 2: 5}
    print(a)
    print(a + b)

    a += b
    print(a)
    print(b)
    a -= b
    print(a)
    a -= a
    print(a)

    history = AccumulatingDict()
    new_metrics = {'acc': .8, 'f1-score': .7}
    print(history)
    history.append(new_metrics)
    print(history)
    history.append(new_metrics)
    print(history)

    d = {
        'a': history,
        'b': history,
    }
    print(type(d['a']))

    d = {k: dict(v) for k, v in d.items()}
    print(type(d['a']))

    print(type(AccumulatingDict({'a': d})))
