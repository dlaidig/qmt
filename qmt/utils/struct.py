# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import json
import gzip
import os
from collections.abc import Iterable
from copy import deepcopy

import scipy.io as spio
import numpy as np


class Struct:
    """
    Wrapper around a Python dict that makes it behave similar to Matlab structs, including support for .mat files.

    Dots are used as separators to easily create and access nested structures::

        data = Struct()
        data['foo.bar.baz'] = 42
    """
    def __init__(self, data=None, **kwargs):
        """
        Creates a new Struct object.

        :param data: dict, list, other Struct or any data type that can be converted to a dict
        :param kwargs: kwargs passed to the dict constuctor
        """
        if data is not None and kwargs:
            # Make sure Struct(data=1, dataB=2) works as expected.
            # Note that Struct(1, dataB=2) unfortunately works as well and is equivalent, which can only be solved with
            # positional-only parameters introduced in Python 3.8.
            self._data = dict(data=data, **kwargs)
        elif data is not None:
            if isinstance(data, list):
                self._data = data
            else:
                self._data = dict(data)
        else:
            self._data = dict(**kwargs)

    @classmethod
    def load(cls, filename, squeeze=True, verify=True):
        """
        Loads a mat/json/yaml file into a new Struct object.

        :param filename: filename of the mat/json/yaml file
        :param squeeze: whether unix matrix dimensions should be squeezed for mat files
        :param verify: enables verification of the length of compressed data segments for mat files
        :return: a new Struct object with the data from the file
        """

        filename = str(filename)
        if filename.endswith('.mat'):
            return cls(_loadmat(filename, squeeze, verify))
        elif filename.endswith(('.json', '.json.gz')):
            with _openAutoGzip(filename) as f:
                data = json.load(f)
            _createArrays(data)
            return cls(data)
        elif filename.endswith(('.yml', '.yaml', '.yml.gz', '.yaml.gz')):
            import yaml
            with _openAutoGzip(filename) as f:
                data = yaml.safe_load(f)
            _createArrays(data)
            return cls(data)
        elif filename.endswith(('.jsonl', '.jsonl.gz')):
            data = []
            with _openAutoGzip(filename) as f:
                for line in f:
                    data.append(json.loads(line))
            _createArrays(data)
            return cls(data)
        else:
            raise ValueError('invalid file extension')

    @classmethod
    def fromJson(cls, string):
        data = json.loads(string)
        _createArrays(data)
        return cls(data)

    def save(self, filename, compress=True, indent=None, sort_keys=False, makedirs=False):
        """
        Saves the data to a mat or json file.

        :param filename: filename of the output file (will be overwritten if it exists)
        :param compress: enable compression (for mat files)
        :param indent: enable indentation (for json files)
        :param sort_keys: sort keys (for json files)
        :param makedirs: recursively create parent directories if they do not exist
        :return: None
        """
        filename = str(filename)
        if makedirs:
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        if filename.endswith('.mat'):
            spio.savemat(filename, self._data, long_field_names=True, do_compression=compress, oned_as='column')
        elif filename.endswith('.json'):
            with open(filename, 'wb') as f:
                from qmt.utils.misc import toJson
                content = toJson(self.data, indent=indent, sort_keys=sort_keys)
                f.write(content)
        elif filename.endswith('.json.gz'):
            import gzip
            with gzip.open(filename, 'wb') as f:
                from qmt.utils.misc import toJson
                content = toJson(self.data, indent=indent, sort_keys=sort_keys)
                f.write(content)
        else:
            raise ValueError('invalid filename extension')

    def toJson(self, method='orjson', indent=None, sort_keys=False):
        from qmt.utils.misc import toJson
        return toJson(self.data, method, indent, sort_keys)

    @property
    def data(self):
        """
        Provides access to the underlying ``dict`` (or in some special cases ``list``) object holding the actual data.

        :return: dict (or list)
        """
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def __getitem__(self, key):
        key = str(key)
        keys = key.split('.')
        item = self._data
        for k in keys:
            try:
                item = item[int(k)] if isinstance(item, list) else item[k]
            except KeyError as e:
                raise KeyError(key) from e
        if isinstance(item, dict):
            return Struct(item)
        elif isinstance(item, list):
            return Struct(item)
        return item

    def __setitem__(self, key, value):
        keys = key.split('.')
        item = self._data
        for k in keys[:-1]:
            if k not in item:
                item[k] = {}
            item = item[int(k)] if isinstance(item, list) else item[k]

        if isinstance(value, Struct):
            value = value._data
        elif isinstance(value, Iterable) and not isinstance(value, (list, dict, str)):
            value = np.asarray(value)

        value = deepcopy(value)

        if isinstance(item, list):
            item[int(keys[-1])] = value
        else:
            item[keys[-1]] = value

    def __len__(self):
        return len(self.keys())

    def __repr__(self):
        s = json.dumps(self._data, indent=4, default=lambda x: '___TYPE___'+_formatType(x)+'___TYPE___', sort_keys=True)
        s = s.replace(' "___TYPE___', ' ')
        s = s.replace('___TYPE___"', '')
        return 'Struct ' + s

    def __delitem__(self, key):
        keys = key.split('.')
        item = self._data
        toCheck = []
        for k in keys[:-1]:
            toCheck.append(item)
            item = item[k]

        del item[keys[-1]]

        # delete empty dicts on higher levels of the hierarchy
        for item in reversed(toCheck):
            for key in list(item.keys()):
                val = item[key]
                if isinstance(val, dict) and len(val) == 0:
                    del item[key]

    def __contains__(self, item):
        return item in self.allKeys()

    def keys(self):
        """
        Returns the top-level keys.

        :return: iterable
        """
        return [str(i) for i in range(len(self._data))] if isinstance(self._data, list) else self._data.keys()

    def leafKeys(self):
        """
        Returns the keys of all leaf elements, i.e. all entries that are not Structs but hold actual data.

        :return: iterable
        """
        return self._keys(self._data, allKeys=False)

    def allKeys(self):
        """
        Returns all valid keys, including the leaf elements but also all intermediate Structs.

        :return: iterable
        """
        return self._keys(self._data, allKeys=True)

    def _keys(self, base, allKeys, prefix=''):
        keys = []
        for k in sorted(base.keys()):
            if isinstance(base[k], dict):
                if allKeys:
                    keys.append(prefix + k)
                keys.extend(self._keys(base[k], allKeys, prefix+k+'.'))
            else:
                keys.append(prefix+k)
        return keys

    def get(self, key, default=None):
        """
        Gets an item by key and returns a default value if the item does not exist.

        :param key: the key to look up
        :param default: default value to return if item does not exist
        :return: self[key] or default
        """
        try:
            return self[key]
        except KeyError:
            return default

    def createArrays(self):
        """Turn all lists that only contain numbers into numpy arrays."""
        _createArrays(self._data)

    def diff(self, ref, exact=False, rtol=1e-07, atol=0, verbose=False, plot=False):
        """
        Print differences between this struct and another given struct.

        :param ref: other Struct object
        :param exact: if set to true, rtol and atol are set to zero
        :param rtol: relative tolerance for comparing numpy arrays (with np.testing.assert_allclose)
        :param atol: absolute tolerance for comparing numpy arrays (with np.testing.assert_allclose)
        :param verbose: enables printing of more detailed output
        :param plot: if set to True, a plot is created for all numpy arrays that have the same shape but different data
        :return: None
        """
        import textwrap
        if exact:
            rtol = 0
            atol = 0

        changes = 0

        keys = set(self.leafKeys())
        refKeys = set(ref.leafKeys())

        def printVal(val, char, direct=False):
            if not direct:
                val = repr(val)
            print(textwrap.indent(val, f'  {char} ', lambda _: True))

        for k in sorted(keys | refKeys):
            if k not in refKeys:
                print(k)
                printVal(self[k], '+')
                changes += 1
                continue
            elif k not in keys:
                print(k)
                printVal(ref[k], '-')
                changes += 1
                continue

            valRef = ref[k]
            valData = self[k]
            if not isinstance(valRef, np.ndarray) or not isinstance(valData, np.ndarray):
                res = valRef != valData
                if np.any(res) if isinstance(res, np.ndarray) else res:
                    print(k)
                    printVal(valRef, '-')
                    printVal(valData, '+')
                    changes += 1
                elif verbose:
                    print(f'no difference for non-array value {k}')
            elif valRef.shape != valData.shape:
                print(k)
                printVal(valRef, '-')
                printVal(valRef.shape, '-')
                printVal(valData, '+')
                printVal(valData.shape, '+')
                printVal('shapes are different', '>', direct=True)
                changes += 1
            else:
                try:
                    np.testing.assert_allclose(valData, valRef, rtol=rtol, atol=atol, verbose=False)
                    if verbose:
                        print(f'no difference for array {k}')
                except AssertionError as e:
                    print(k)
                    printVal(valRef, '-')
                    printVal(valData, '+')
                    printVal(str(e).strip(), '>', direct=True)
                    changes += 1
                    if plot:
                        print('  ! see PLOT')
                        from matplotlib import pyplot as plt
                        fig = plt.figure(figsize=(16, 10))
                        axes = fig.subplots(3, 1)
                        axes[0].plot(valRef)
                        axes[1].plot(valData)
                        axes[2].plot(valData - valRef)
                        axes[0].set_title('ref')
                        axes[1].set_title('data')
                        axes[2].set_title('diff (data-ref)')
                        for ax in axes:
                            ax.grid()
                        plt.suptitle(k)
                        plt.tight_layout()

        print(f'found {changes} differences')


def _openAutoGzip(filename):
    if filename.endswith('.gz'):
        return gzip.open(filename, 'rt', encoding='utf-8')
    else:
        return open(filename, 'r', encoding='utf-8')


# http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
# answer by "mergen"
def _loadmat(filename, squeeze=True, verify=True):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=squeeze, verify_compressed_data_integrity=verify)
    data.pop('__globals__', None)  # delete if exists
    data.pop('__header__', None)
    data.pop('__version__', None)
    return _check_keys(data)


def _check_keys(d):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in d:
        if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
            d[key] = _todict(d[key])
        # for use with flatten=False
        elif isinstance(d[key], np.ndarray) and d[key].shape == (1, 1) and \
                isinstance(d[key][0, 0], spio.matlab.mio5_params.mat_struct):
            d[key] = _todict(d[key][0, 0])
        # for cell arrays of structs
        elif isinstance(d[key], np.ndarray) and len(d[key].shape) == 1 and d[key].dtype == object and \
                isinstance(d[key][0], spio.matlab.mio5_params.mat_struct):
            d[key] = [_todict(e) for e in d[key]]
    return d


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    d = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            d[strg] = _todict(elem)
        # for use with flatten=False
        elif isinstance(elem, np.ndarray) and elem.shape == (1, 1) and \
                isinstance(elem[0, 0], spio.matlab.mio5_params.mat_struct):
            d[strg] = _todict(elem[0, 0])
        # for cell arrays of structs
        elif isinstance(elem, np.ndarray) and len(elem.shape) == 1 and elem.dtype == object and \
                isinstance(elem[0], spio.matlab.mio5_params.mat_struct):
            d[strg] = [_todict(e) for e in elem]
        else:
            d[strg] = elem
    return d


def _formatType(obj):
    if isinstance(obj, np.ndarray):
        if len(obj.shape) <= 1 and obj.size <= 4 and obj.dtype in (np.float64, np.int64, np.int32, bool):
            return repr(obj)
        return '<array, {0}'.format('x'.join(str(n) for n in obj.shape)) + (f', dtype={obj.dtype!s}'
                                                                            if obj.dtype != np.float64 else '') + '>'
    return '<' + type(obj).__name__ + '>'


def _createArrays(data):
    """In nested dicts, recursively replace lists containing only numbers with numpy arrays."""
    if isinstance(data, dict):
        for k in data:
            if _createArrays(data[k]):
                data[k] = np.array(data[k], float)
    elif isinstance(data, list):
        if _listIsArray(data):
            return True
        for i in range(len(data)):
            if _createArrays(data[i]):
                data[i] = np.array(data[i], float)
    return False


def _listIsArray(item):
    return all(_listIsArray(e) if isinstance(e, list) else isinstance(e, (float, int)) for e in item)
