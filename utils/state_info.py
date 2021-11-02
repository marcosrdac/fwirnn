from typing import Optional, Any, Dict, Iterator, Sequence
from numbers import Number
from itertools import count
from dataclasses import dataclass
from copy import copy


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
            id = self.ids[name]
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

    def pprint(self, *args, sep: str = ' ', **kwargs):
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

    def updated(self, other):
        new = self.copy()
        for name in other.ids:
            new[name] = other[name], other.info(name)
        return new


if __name__ == '__main__':
    infos = StateInfo()
    infos['epoch'] = 3, {'abbrv': 'e', 'show': False}
    infos['learning_rate'] = 2e8, 'learning_rate'
    # print(infos.print('test', 'epoch'))
    print(infos.pprint())
    print(infos)
    # infos.remove('test')
    print(infos)
    print(infos.pprint())
    print(infos.copy())
    infos2 = StateInfo()
    infos2['a'] = 4, 'a'
    infos2['test'] = 9, 'y'
    print(infos.updated(infos2))
