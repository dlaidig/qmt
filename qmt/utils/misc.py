# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import json
import orjson
import numpy as np


def orjson_default(obj):
    """Helper function to handle various edge cases in json encoding with orjson."""
    if isinstance(obj, np.ndarray):
        if not obj.flags['C_CONTIGUOUS']:
            return np.ascontiguousarray(obj)
        if obj.dtype in (np.uint8,):
            return obj.astype(np.uint64)
    if isinstance(obj, np.float64):
        return obj.item()
    print('warning: not handled by orjson default function:', repr(obj), type(obj))
    raise TypeError


class NumpyJSONEncoder(json.JSONEncoder):
    """
    JSON encoder class with support for numpy arrays. Usage: json.dumps(obj, cls=NumpyJSONEncoder)

    Based on https://github.com/mkocabas/VIBE/issues/49#issuecomment-606381400
    """
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def toJson(obj, method='orjson', indent=None, sort_keys=False):
    """
    Versatile and fast json-encoding function that can handle numpy array and qmt.Struct. Uses the fast orjson
    library by default.

    :param obj: object to be encoded (list, dict, or qmt.Struct)
    :param method: json encoder to use, values: 'orjson' (default) or 'json'
    :param indent: indent output (with orjson, indentation is always 2 spaces and enabled if indent evalues to True)
    :param sort_keys: enables sorting of dictionary keys
    :return: encoded json as bytes (with method=orjson, default) or string (with method=json)
    """
    from qmt.utils.struct import Struct
    if isinstance(obj, (dict, list)):
        if method == 'orjson':  # returns bytes!
            option = orjson.OPT_SERIALIZE_NUMPY
            if sort_keys:
                option |= orjson.OPT_SORT_KEYS
            if indent:  # orjson only supports one indent level
                option |= orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE
            return orjson.dumps(obj, default=orjson_default, option=option)
        elif method == 'json':  # returns str!
            return json.dumps(obj, indent=indent, sort_keys=sort_keys, cls=NumpyJSONEncoder)
        else:
            raise ValueError('invalid method')
    elif isinstance(obj, Struct):
        return obj.toJson()
    raise ValueError('invalid obj')


def setDefaults(params, defaults, mandatory=None, check=True):
    """
    Helper function to facilitate handling of multiple optional parameters that are passed in a dictionary. Default
    values are set for missing parameters and, if check is True, an error is raised if unexpected parameters are passed
    or some non-optional parameters are missing.

    The input dictionary is not modified.

    >>> qmt.setDefaults(dict(mandatory1=42, optional2=3), dict(optional1=1, optional2=2), ['mandatory1'])
    {'mandatory1': 42, 'optional2': 3, 'optional1': 1}

    :param params: dict with parameter values or None
    :param defaults: dict with default parameter values
    :param mandatory: optional list of mandatory parameter names that need to be contained in parames
    :param check: enables checking of the passed parameter names
    :return: dict with parameter values after applying default values
    """
    params = {} if params is None else params.copy()
    assert isinstance(params, dict)
    assert isinstance(defaults, dict)
    params.update((k, v) for k, v in defaults.items() if k not in params)
    if check:
        expected = set(defaults.keys())
        if mandatory:
            expected.update(mandatory)
        keys = set(params.keys())
        assert keys == expected, f'wrong set of params: {keys} != {expected}'
    return params


def startStopInd(booleanArray):
    """
    Determines start and end indices of true phases in a boolean array.

    >>> qmt.startStopInd([0, 0, 1, 1, 1, 0, 0, 1, 0, 0])
    (array([2, 7]), array([5, 8]))

    :param booleanArray: 1D boolean array
    :return:
        - starts: array with start indices of true phases
        - stops: array with stop indices of true phases
    """
    # https://stackoverflow.com/a/29853487/1971363
    booleanArray = np.asarray(booleanArray, bool)
    assert booleanArray.ndim == 1
    x1 = np.hstack([[False], booleanArray, [False]])
    d = np.diff(x1.astype(int))
    starts = np.where(d == 1)[0]
    stops = np.where(d == -1)[0]
    assert starts.shape == stops.shape
    return starts, stops
