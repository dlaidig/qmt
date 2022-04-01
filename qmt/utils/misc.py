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


def _parallelTarget(args):
    i, item = args
    res = item['fn'](*item.get('args', tuple()), **item.get('kwargs', {}))
    return i, res


def parallel(items, ordered=False, workers=None, chunksize=1, verbose=False):
    """
    Simple utility to perform parallel processing.

    This is a simple wrapper around the ``multiprocessing`` package and meant to make usage simpler for a common
    application scenario in which a list of function calls (with potentially different functions and parameters)
    is distributed to multiple worker processes.

    The main parameter is a list of items. Each item corresponds to one function call and is a dictionary that must
    have an ``fn`` key with the function to be called in the worker process. Arguments that are passed to the function
    can be specified by setting ``args`` to a tuple and/or by setting ``kwargs`` to a dict. Optionally, a ``name`` can
    be set for the item which is used only for the debug output. The item may contain other keys which are ignored by
    this function (but they will still be sent to the worker process, so adding large data should be avoided).

    This function is a generator that yields a tuple (i, item, res) for each element of items.
    ``i`` is the original index in the ``items`` list, ``item`` is ``items[i]`` and ``res`` is the return value of
    the executed function ``fn``. Unless ``ordered`` is set to True, results are not necessarily returned in the
    original order.

    The following minimal working example shows how to use this utility::

        import time

        def slowFunction(t):
            time.sleep(t)
            return t

        # create list of items to be processed
        items = [{'fn': slowFunction, 'kwargs': {'t': t}} for t in [5, 10, 4, 2, 8, 3]]

        for i, item, res in parallel(items, verbose=True):
            print(i, item, res)

    By default, the number of workers is automatically chosen to match the number of CPU cores. By setting ``workers``
    to 1, the use of ``multiprocessing`` is disabled, making debugging and profiling much easier.

    How to best split a slow task into multiple function calls (i.e., items) depends on the specific problem. A good
    rule of thumb is that each item should at least take a few seconds (to avoid that overhead plays a large role) while
    on the other hand the number of items should be significantly larger than the number of CPU cores (to avoid that the
    time in which some CPU cores are idle at the end plays a large role). Also, consider splitting the items in a way
    that makes combining the results as easy and fast as possible. In many cases, when processing experimental data with
    several trials/recordings, having one item per trial (and loading the trial data locally in the target function) is
    a good idea.

    :param items: list of items to process (each item is a dict as described above and corresponds to one function call)
    :param ordered: set to True to force the results to be returned in the same order as the items list
    :param workers: number of worker processes to use (when set to the default of None, the number of CPU cores is used)
    :param chunksize: chunk size passed to multiprocessing (the default of 1 works well unless there are lots of items)
    :param verbose: enables output of status information
    :return: generator that yields (i, item, res) for each element of items (in arbitrary order unless ordered==True)
    """

    import multiprocessing
    from contextlib import nullcontext

    if workers is None:
        workers = multiprocessing.cpu_count()

    if verbose:
        print(f'[parallel] processing {len(items)} items on {workers} workers')

    with multiprocessing.Pool(workers) if workers != 1 else nullcontext() as pool:
        if workers == 1:  # do not use multiprocessing to facilitate debugging/profiling
            iterator = map(_parallelTarget, enumerate(items))
        elif ordered:
            iterator = pool.imap(_parallelTarget, enumerate(items), chunksize=chunksize)
        else:
            iterator = pool.imap_unordered(_parallelTarget, enumerate(items), chunksize=chunksize)

        for doneCount, (i, res) in enumerate(iterator, start=1):
            if verbose:
                percentage = 100 * doneCount / len(items)
                nameStr = f'{items[i]["name"]}, ' if 'name' in items[i] else ''
                print(f'[parallel] item {i} done: {nameStr}{doneCount} of {len(items)}, {percentage:.0f} %')
            yield i, items[i], res
