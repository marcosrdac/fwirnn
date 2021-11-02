from copy import copy
from dataclasses import dataclass
from itertools import count
from numbers import Number
from typing import Optional, Any, Dict, Iterator, Sequence


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


class StateInfo:
    '''
    '''
    ids: Dict[str, int]
    names: Dict[int, str]
    abbrvs: Dict[int, str]
    vals: Dict[int, Any]
    fmts: Dict[int, bool]
    eqs: Dict[int, str]
    show: Dict[int, bool]
    counter: Iterator

    def __init__(self):
        '''
        '''
        self.ids = {}
        self.names = {}
        self.abbrvs = {}
        self.vals = {}
        self.fmts = {}
        self.eqs = {}
        self.show = {}
        self.counter = count(0, 1)

    def __len__(self):
        return len(self.ids)

    def copy(self):
        return copy(self)

    def __getitem__(self, name):
        id = self.ids[name]
        return self.vals[id]

    def get(self, name, val=None):
        try:
            return self[name]
        except KeyError:
            return val

    def __setitem__(self, name, vals):
        if isinstance(vals, tuple):
            val = vals[0]
            info = vals[1]
            if not isinstance(info, dict):
                info = {'abbrv': vals[1]}
        else:
            val = vals
            info = {}

        if name in self.ids:
            id = self.ids[name]
            self.vals[id] = val
            if info:
                if info.get('abbrv'):
                    self.abbrvs[id] = info['abbrv']
                if info.get('fmt'):
                    self.fmts[id] = info['fmt']
                if info.get('eq'):
                    self.eqs[id] = info['eq'] if name else '='
                if info.get('show'):
                    self.show[id] = info['show']
        else:
            id = next(self.counter)
            self.ids[name] = id
            self.names[id] = name
            self.vals[id] = val

            self.abbrvs[id] = info.get('abbrv') or name[0].lower()
            self.eqs[id] = info.get('eq', ' = ') if name else '='
            fmt = info.get('fmt', '.4g' if isinstance(val, Number) else '')
            self.fmts[id] = fmt
            self.show[id] = info.get('show', True)

    def info(self, name):
        id = self.ids[name]
        return {
            'abbrv': self.abbrvs[id],
            'eq': self.eqs[id],
            'fmt': self.fmts[id],
            'show': self.show[id],
        }

    def __repr__(self):
        return self.print()

    def _reprs(self, *names, pretty=False):
        names = names or [n for n in self.ids if self.show[self.ids[n]]]
        reprs = []
        for name in names:
            id = self.ids.get(name)

            if id is None:
                continue

            if pretty:
                reprs.append(f'{self.names[id]}'  # .capitalize()
                             f'{self.eqs[id]}'
                             f'{self.vals[id]:{self.fmts[id]}}')
            else:
                reprs.append(
                    f'{self.abbrvs[id]}={self.vals[id]:{self.fmts[id]}}')
        return reprs

    def list(self, *names, **kwargs):
        return self._reprs(*names, **kwargs, pretty=False)

    def plist(self, *names, **kwargs):
        return self._reprs(*names, **kwargs, pretty=True)

    def print(self, *args, sep: str = ' ', **kwargs):
        reprs = self.list(*args, **kwargs)
        return sep.join(reprs)

    def filename(self, *args, sep: str = '_', **kwargs):
        return self.print(*args, sep=sep, **kwargs)

    def pprint(self, *args, sep: str = ', ', **kwargs):
        reprs = self.plist(*args, **kwargs)
        return sep.join(reprs)

    def remove(self, name):
        '''
        '''
        id = self.ids[name]
        del self.ids[name]
        del self.names[id]
        del self.vals[id]
        del self.abbrvs[id]
        del self.fmts[id]
        del self.eqs[id]
        del self.show[id]

    def update(self, other):
        new = self.copy()
        for name in other.ids:
            new[name] = other[name], other.info(name)
            print(new)
            print()
        return new


if __name__ == '__main__':
    def test1():
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

    def test2():
        infos = StateInfo()
        infos['epoch'] = 3, {'abbrv': 'e', 'show': True}
        infos['learning_rate'] = 2e8, 'lr'
        print(infos.print('test', 'epoch'))
        print(infos.pprint())
        print(infos)
        # infos.remove('test')
        print(infos)
        print(infos.pprint())
        # print(infos.copy())
        infos2 = StateInfo()
        infos2['a'] = 4, 'a'
        infos2['test'] = 9, 'y'

        infos3 = infos.update(infos2)
        print(infos3)

        print(infos3.print('epoch'))
        print(infos3.filename())


    test1()
    test2()
