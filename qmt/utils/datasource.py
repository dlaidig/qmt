# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import asyncio
import inspect
import json
import logging
import multiprocessing
import time

import numpy as np

import qmt

logger = logging.getLogger(__name__)


def _toStruct(data):
    if isinstance(data, qmt.Struct) or data is None:
        return data
    elif isinstance(data, dict):
        return qmt.Struct(data)
    elif isinstance(data, str) and data.startswith('{'):
        return qmt.Struct.fromJson(data)
    elif isinstance(data, str):
        return qmt.Struct.load(data)
    else:
        raise ValueError('invalid value for data')


class AbstractDataSource:
    """
    Base class for all data source classes used in the :class:`qmt.Webapp` online data processing loop.

    The data source provides samples that are then (optionally) processed and sent via the websocket. To implement a
    custom data source, you must implement either ``poll`` or ``apoll`` or ``samples``. Which method to implement
    depends on what works best for the given problem.
    """
    def __init__(self, Ts):
        """
        :param Ts: sampling time in seconds, or None if the data source does not produce samples at regular times
        """
        self.Ts = Ts
        self.pollTime = 0.01

    @property
    def Ts(self):
        """Sampling time in seconds, or None if the data source does not produce samples at regular times."""
        return self._Ts

    @Ts.setter
    def Ts(self, Ts):
        self._Ts = Ts

    async def setup(self):
        """
        This (async) method is called once at the start and can be used to perform setup that uses the asyncio loop.

        Data sources should not use ``asyncio.get_event_loop()`` in ``__init__`` because the event loop might still
        change.
        """
        pass

    def command(self, command):
        """
        This method is called when a command is received over the websocket. If the data source decides to handle a
        command, it should return True and this command will not be sent to the data processing block.

        Commands should for example be used to control playback of recorded data or initiate connecting to sensors.
        Command names for data sources should be chosen in a way that minimizes possible name clashes with commands
        used for data processing.

        The default implementation just returns False.

        :param command: command as a list
        :return: True if command was handled by the data source, else False
        """
        return False

    def setParams(self, params):
        """
        This method is called when a parameter dict is received over the websocket. Parameters are first passed to the
        data source and then to the data processing block.

        The default implementation does nothing.

        :param params: parameter dict
        :return: None
        """
        pass

    def poll(self):
        """
        This method is used to poll the data source. It should either return a sample or, if no sample is available,
        return None.

        :return: sample dict (must contain 't' with time in seconds) or None if no sample is available
        """
        return NotImplemented

    async def apoll(self):
        """
        Asynchronous poll method that returns the next sample. If no sample is available, the method must wait without
        blocking the event loop.

        :return: sample dict (must contain 't' with time in seconds)
        """
        while True:
            sample = self.poll()
            if sample is not None:
                return sample
            await asyncio.sleep(self.pollTime)

    async def samples(self):
        """
        Asynchronous generator that yields samples as they become available.
        """
        return NotImplemented

    def __aiter__(self):
        if inspect.isasyncgenfunction(self.samples):
            return self.samples()
        else:
            return self

    async def __anext__(self):
        sample = await self.apoll()
        return sample


class ProcessDataSource(AbstractDataSource):
    """
    This data source runs a function in a separate process using the multiprocessing module and allows for the
    integration of blocking processing code (e.g., simulations) that does not work well in combination with
    async code.

    The target function is started in a separate process as ``target(conn, Ts, *args, **kwargs)``, where ``conn`` is a
    :class:`ProcessDataSourceConnection` instance that can be used to communicate with the webapp.
    """

    def __init__(self, target, Ts, *args, pollTime=0.01, **kwargs):
        """
        :param target: target function to run in separate process
        :param Ts: sampling time in seconds (pass None if irregular or not relevant for further processing)
        :param args: positional arguments to pass to the target function
        :param pollTime: poll time for the multiprocessing pipe in seconds (only used on Windows)
        :param kwargs: keyword arguments to pass to the target function
        """
        super().__init__(Ts)
        self.pollTime = pollTime
        self.pipe, targetPipe = multiprocessing.Pipe()
        connection = ProcessDataSourceConnection(targetPipe)
        self.frameAvailable = None
        self.process = multiprocessing.Process(target=target, args=(connection, self.Ts) + args, kwargs=kwargs)
        self.process.start()

    async def setup(self):
        try:
            event = asyncio.Event()  # https://stackoverflow.com/a/62098165
            asyncio.get_event_loop().add_reader(self.pipe.fileno(), event.set)
            self.frameAvailable = event
        except NotImplementedError:
            # add_reader is not supported on Windows with the ProactorEventLoop:
            # https://docs.python.org/3/library/asyncio-platforms.html#asyncio-platform-support
            logger.info('ProcessDataSource: falling back to polling because add_reader is not supported')

    def command(self, command):
        """"""  # do not show in documentation
        self.pipe.send(['command', command])
        return True

    def setParams(self, params):
        """"""  # do not show in documentation
        self.pipe.send(['params', params])

    def poll(self):
        """"""  # do not show in documentation
        if not self.pipe.poll():
            return None
        return self.pipe.recv()

    async def __anext__(self):
        try:
            if self.frameAvailable is None:
                return await self.apoll()  # use polling
            else:
                self.frameAvailable.clear()
                if not self.pipe.poll():
                    await self.frameAvailable.wait()
                return self.pipe.recv()
        except asyncio.CancelledError:
            self.pipe.send(['close'])
            raise


class ProcessDataSourceConnection:
    """
    Helper for :class:`qmt.ProcessDataSource`. An instance of this class passed to the target process and can be
    used to communicate with the webapp.
    """
    def __init__(self, pipe):
        self._pipe = pipe
        self._params = {}
        self._commands = []
        self._isClosed = False
        self._raiseInterruptOnClose = False

    @property
    def raiseInterruptOnClose(self):
        """
        Set this property to True to raise a KeyboardInterrupt when the connection is closed.

        If this is not set, :meth:`isClosed` needs to be called regularly to detect when the connection is closed.
        """
        return self._raiseInterruptOnClose

    @raiseInterruptOnClose.setter
    def raiseInterruptOnClose(self, raiseInterrupt):
        self._raiseInterruptOnClose = raiseInterrupt

    def getParams(self, clear=False):
        """
        Returns a dictionary containing the received parameters.

        :param clear: If True, the parameter dictionary is cleared after returning the parameters (allows for the
            detection which parameters were sent since the last call to this function).
        :return: dict with parameters received over websocket
        """
        self._readPipe()
        params = self._params
        if clear:
            self._params = {}
        return params

    def getCommands(self):
        """
        Iterator to get new received commands.

        Commands are deleted from the internal FIFO once they are read. Regularly loop over this iterator to receive and
        handle all new commands.
        """
        self._readPipe()
        while self._commands:
            yield self._commands.pop(0)

    def isClosed(self):
        self._readPipe()
        return self._isClosed

    def sendSample(self, sample):
        """
        Sends a sample to the webapp via the websocket.

        :param sample: sample dict
        :return: None
        """
        assert isinstance(sample, dict)
        self._readPipe()
        try:
            self._pipe.send(sample)
        except ConnectionResetError:
            self._closed()

    def sendCommand(self, command):
        """
        Sends a command to the webapp via the websocket.

        :param command: command (as list)
        :return: None
        """
        assert isinstance(command, list)
        self._readPipe()
        try:
            self._pipe.send(command)
        except ConnectionResetError:
            self._closed()

    def _closed(self):
        self._isClosed = True
        if self.raiseInterruptOnClose:
            raise KeyboardInterrupt

    def _readPipe(self):
        try:
            while self._pipe.poll():
                msg = self._pipe.recv()
                if msg[0] == 'params':
                    self._params.update(msg[1])
                elif msg[0] == 'command':
                    self._commands.append(msg[1])
                elif msg[0] == 'close':
                    self._closed()
                else:
                    raise RuntimeError(f'invalid message received: {msg}')
        except ConnectionResetError:
            self._closed()
        except EOFError:
            self._closed()


class ClockDataSource(AbstractDataSource):
    """
    This data sources generates samples at a fixed interval, containing only the time.

    Use this class in combination with a data processing block to create animations.
    """
    def __init__(self, Ts):
        """
        :param Ts: sample time at which to generate samples
        """
        super().__init__(Ts)

    async def samples(self):
        """"""  # do not show in documentation
        t0 = time.monotonic()
        N = 0
        while True:
            t = round(N*self.Ts, 9)
            yield dict(t=t)
            N += 1
            await asyncio.sleep(t0 + t - time.monotonic())


class PlaybackDataSource(AbstractDataSource):
    """
    Data source that generates samples from recorded/generated data similar to the playback mode in webapps.

    The data struct must have a time vector named 't'. Samples are generated from all elements in the data that have
    the same length as 't'.
    """
    def __init__(self, data):  # NOTE: this could be extended with loop=False, autoPlay=True, speed=1.0, reverse=False
        """
        :param data: qmt.Struct, dict or filename of a file that can be loaded with qmt.Struct.load()
        """
        self.data = _toStruct(data)
        # self.loop = loop

        super().__init__(self._determineTs())

        self._setup()

    def _determineTs(self):
        diff = np.diff(self.data['t'])
        if np.allclose(diff[0], diff):
            return np.round(diff[0], 8)
        else:  # irregular sampling time
            return None

    def _setup(self):
        t = self.data['t']
        N, = t.shape  # must be 1D array
        self._keys = []
        for k in self.data.leafKeys():
            val = self.data[k]
            if isinstance(val, np.ndarray) and len(val.shape) and val.shape[0] == N:
                self._keys.append(k)

    async def samples(self):
        """"""  # do not show in documentation
        t = time.monotonic()
        pos = 0
        timeVector = self.data['t']

        while True:
            yield self._getSample(pos)
            pos += 1
            if pos >= timeVector.shape[0]:
                return
            t += timeVector[pos] - timeVector[pos-1]
            await asyncio.sleep(t - time.monotonic())

    def _getSample(self, pos):
        sample = qmt.Struct()
        for k in self._keys:
            sample[k] = self.data[k][pos]
        return sample.data


class DummyImuDataSource(ClockDataSource):
    """
    Data source that generates fake IMU data.

    For every virtual IMU, a fake sensor movement is generated from random orientations, and raw measurement data is
    derived from this. Noise and a constant bias are added. The generated data is not meant to be realistic in
    terms of error characteristics, but reasonable enough to be useful for some testing and development tasks where real
    hardware would otherwise be required.
    """
    def __init__(self, Ts, N=1):
        """
        :param Ts: sample time at which to generate samples
        :param N: number of virtual IMUs
        """
        super().__init__(Ts)
        self.N = N
        self.movementGenerators = [self.randomSensorMovement(Ts) for _ in range(N)]

        self.gyrBias = np.deg2rad([0.2, 0.1, -0.3])
        self.gyrNoiseStd = np.deg2rad(1)
        self.accBias = np.array([0.03, -0.01, 0.02], float)
        self.accNoiseStd = 0.2
        self.magBias = np.array([-10, 15, 5], float)
        self.magNoiseStd = 0.5
        self.accRef = np.array([0, 0, 9.81], float)
        magNorm = 49.8
        magDip = np.deg2rad(68)
        self.magRef = np.array([0, magNorm*np.cos(magDip), -magNorm*np.sin(magDip)], float)

    async def samples(self):
        lastQuat = [next(self.movementGenerators[i]) for i in range(self.N)]
        async for sample in super().samples():
            for i in range(self.N):
                quat = next(self.movementGenerators[i])

                gyr = qmt.quatToGyrStrapdown([lastQuat[i], quat], 1/self.Ts)[1]
                sample[f'gyr{i+1}'] = gyr + self.gyrBias + np.random.normal(0, self.gyrNoiseStd, (3,))

                acc = qmt.rotate(qmt.qinv(quat), self.accRef)
                sample[f'acc{i+1}'] = acc + self.accBias + np.random.normal(0, self.accNoiseStd, (3,))

                mag = qmt.rotate(qmt.qinv(quat), self.magRef)
                sample[f'mag{i+1}'] = mag + self.magBias + np.random.normal(0, self.magNoiseStd, (3,))

                sample[f'quat{i + 1}'] = quat
                # for real IMUs, this should be a serial number or something else to uniquely identify the device
                # (this can be used to store calibration parameters, for example)
                sample[f'sensorid{i+1}'] = f'dummy{i+1}'
                sample[f'battery{i+1}'] = 100

                lastQuat[i] = quat
            yield sample

    @staticmethod
    def randomSensorMovement(Ts):
        quats = qmt.randomQuat(1)
        while True:
            for q in quats:
                yield q

            # determine next random orientation
            q0 = quats[-1]
            angle = np.random.uniform(np.pi/6, np.pi)
            q1 = qmt.qmult(q0, qmt.randomQuat(angle=angle))

            # determine time vector with random length and sine easing function
            duration = np.random.uniform(1, 3)
            N = round(duration/Ts)
            t = np.linspace(0, 1, N)
            t = -(np.cos(np.pi * t) - 1) / 2

            # interpolate from previous orientation to the next orientation
            quats = qmt.slerp(q0, q1, t)


def dataSourceFromJson(jsonString, classes=None, autoImport=True):
    """
    Create a data source object from a JSON object containing the class name and parameters.

    The JSON object must contain a ``class`` entry with the name or path of a qmt.AbstractDataSource subclass. Custom
    classes can be defined in the optional ``classes`` parameter. Unless ``autoImport`` is set to false, external
    packages are automatically imported, i.e., the user can specify "my_package.MyCustomDataSource" and "my_package"
    will be automatically imported. This can be useful to decouple the processing code from the hardware backend, but
    the jsonString should not come from untrusted user input.

    This function is used by the ``qmt-webapp`` CLI tool, e.g.:

    .. code-block:: python

        qmt-webapp --datasource '{"class": "qmt.DummyImuDataSource", "Ts": 0.01, "N": 3}' /view/imubox

    :param jsonString: json string (or dict) defining the data source class and arguments
    :param classes: optional dictionary defining custom class names
    :param autoImport: enables automatic import of modules based on the given class name
    :return: data source object
    """

    kwargs = jsonString.copy() if isinstance(jsonString, dict) else json.loads(jsonString)
    name = kwargs.pop('class')
    if classes is not None and name in classes:
        cls = classes[name]
    elif name.startswith('qmt.'):
        cls = getattr(qmt, name[len('qmt.'):])
    elif autoImport and '.' in name:
        import importlib
        moduleName, className = name.rsplit('.', maxsplit=1)
        module = importlib.import_module(moduleName)
        cls = getattr(module, className)
    else:
        raise ValueError(f'invalid data source class name "{name}"')

    assert issubclass(cls, AbstractDataSource)
    return cls(**kwargs)
