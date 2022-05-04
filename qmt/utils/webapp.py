# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import asyncio
import inspect
import logging
import os
import signal
import sys
import time
from collections import defaultdict

import aiofiles

from qmt.blocks.core import Block
from qmt.utils.struct import Struct
from qmt.utils.datasource import AbstractDataSource, PlaybackDataSource, _toStruct
from qmt.utils.misc import toJson

logger = logging.getLogger(__name__)


def _handleSpecialPaths(path, prefixes):
    if path is None:
        return path
    for prefix, target in prefixes:
        if path.startswith(prefix):
            path = os.path.join(target, path[len(prefix):])
            htmlpath = path + '.html'
            htmlpath2 = os.path.join(path, 'index.html')
            if os.path.isfile(htmlpath):
                return htmlpath
            if os.path.isfile(htmlpath2):
                return htmlpath2
            return path
    else:
        path = os.path.abspath(path)
        htmlpath = os.path.join(path, 'index.html')
        if os.path.isdir(path) and os.path.isfile(htmlpath):
            return htmlpath
        return path


class WebappLogFile:
    def __init__(self, filename):
        assert filename.endswith('.jsonl')
        self.filename = filename
        self.t0 = None
        self.f = None

    async def log(self, name, value):
        if self.f is None:
            logger.info(f'opening log file {self.filename}')
            self.f = await aiofiles.open(self.filename, 'w', encoding='utf-8')
        if self.t0 is None:
            self.t0 = time.monotonic()

        if isinstance(value, bytes):
            value = value.decode(encoding='utf-8')

        await self.f.write(f'[{time.time():.6f}, {time.monotonic()-self.t0:.6f}, "{name}", {value}]\n')

    async def close(self):
        logger.info(f'closing log file {self.filename}')
        await self.f.close()


class AbstractWebappViewer:
    def __init__(self, webapp):
        self.webapp = webapp

    def setupLoop(self):
        raise NotImplementedError

    def createTasks(self, loop):
        raise NotImplementedError

    @property
    def connectionCount(self):
        raise NotImplementedError


class Webapp:
    """
    Runs webapps.

    There are two different viewers for different use cases that are used based on the :attr:`show` property.
    The default PySide-based viewer ('window' or 'webapp') uses the QtWebEngine to show the webapp in a window and a
    QtWebChannel for real-time communication between Python code and the webapp. The aiohttp-based viewer ('chromium',
    'iframe', or 'none') opens a local web server and uses websockets for real-time communication.

    See :ref:`tutorial_py_webapps`, :ref:`ref_webapps`, and the files ``webapp_example_script.py`` and
    ``webapp_example_notebook.ipynb`` in the ``examples`` folder for information on how to use existing webapps and
    :ref:`dev_webapps` for information on how to create custom webapps.

    For playback of stored data from .mat or .json files, there is a command-line utility called ``qmt-webapp``. Run
    ``qmt-webapp -h`` to see how to use it.
    """

    baseDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'webapps'))

    prefixes = (
        ('/demo/', os.path.join(baseDir, 'demo')),
        ('/view/', os.path.join(baseDir, 'view')),
    )

    def __init__(self, path=None, config=None, data=None, show='window', quiet=False):
        """
        :param path: path to the webapp. Can either be the path to a folder containing an index.html file, a path to a
            html file or a special path (e.g. /view/imubox, /demo/euler-angles).
        :param config: configuration for the webapp (qmt.Struct, dict or filename)
        :param data: data for offline playback (qmt.Struct, dict or filename)
        :param show: how to display the webapp, one of ('window', 'widget', 'chromium', 'iframe', 'none')
        :param quiet: if set to True, disable most log output
        """
        self.viewer = None
        self._tasks = []
        self._loopWasRunning = False
        self._callbacks = defaultdict(list)
        self.sendQueue = None
        self._url = None
        self.logfile = None

        self.path = path
        self.config = config
        self.data = data
        self.show = show

        self.noLib = False

        # for PySide viewer only
        self.jsLogLevel = 'warning'  # info, warning, error, disabled
        self.devServerUrl = None

        # for aiohttp viewer only
        self.host = '127.0.0.1'
        self.port = 8000
        self.autoIncrementPort = True
        self.stopOnDisconnect = True
        self.stopOnWindowClose = False
        self.iframeWidth = '100%'
        self.iframeHeight = '500'
        self.chromiumExecutable = None  # 'chromium-browser' on Linux, auto-detect on Windows

        self.source = None
        self.block = None

        self.logLevel = None
        self.setupLogging(logging.WARNING if quiet else logging.INFO)

    def run(self):
        """
        Runs the webapp.

        If no asyncio event loop is running, this function will block until the webapp is closed. If an event loop is
        running (e.g. in Jupyter notebooks), the webapp will start in the background and this function immediately
        returns.

        :return: None
        """

        if self.viewer is None:
            self.viewer = self._createViewer()

        loop = asyncio.get_event_loop()

        self._loopWasRunning = loop.is_running()
        if not self._loopWasRunning:  # let's not mess with the loop if it is already running, e.g. in Jupyter notebooks
            self.viewer.setupLoop()
            loop = asyncio.get_event_loop()

            if sys.platform != 'win32':
                for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
                    loop.add_signal_handler(s, lambda sig=s: loop.create_task(self._shutdown(loop, sig)))
            loop.set_exception_handler(self._handleException)
            loop.set_debug(True)

        self.sendQueue = asyncio.Queue()

        t = loop.create_task(self.arun())

        if self._loopWasRunning:
            return

        try:
            loop.run_until_complete(t)
        except asyncio.CancelledError:
            logger.info('cancelled')
        # most likely an exception in the custom code, let's just raise it
        # except Exception as exception:
        #     exc_info = (type(exception), exception, exception.__traceback__)
        #     logger.error(f'caught exception: {exception!r}', exc_info=exc_info)

    async def arun(self):
        """
        Async method to run the webapp.

        ``await webapp.arun()`` will run the webapp in the current event loop and return after it is closed.

        :return: None
        """
        if self.viewer is None:
            self.viewer = self._createViewer()

        loop = asyncio.get_event_loop()
        self._tasks = [
            loop.create_task(self._runOnlineLoop()),
            loop.create_task(self._emitRunning()),
            # loop.create_task(self._exceptionTest()),  # tests to check if exception handling works
            # loop.create_task(self._exceptionTest2()),
        ] + self.viewer.createTasks(loop)
        return await asyncio.gather(*self._tasks)

    # async def _exceptionTest(self):
    #     asyncio.get_event_loop().create_task(self._exceptionTest2())

    # async def _exceptionTest2(self):
    #     await asyncio.sleep(5)
    #     logger.error('RAISE')
    #     raise RuntimeError()

    def runInProcess(self):
        """
        Runs the webapp in a separate process.

        The attributes of the current class are transferred to a new process (with the multiprocessing module) and a
        new Webapp instance is created and started in this process. The returned connection object can be used to
        communicate with the webapp.

        :return: :class:`qmt.WebappProcessConnection` object
        """
        from qmt.utils.webapp_process import WebappProcessConnection
        return WebappProcessConnection(self)

    def setupLogging(self, logLevel=logging.INFO):
        logging.basicConfig(format='%(asctime)s - %(name)s:%(filename)s:%(lineno)d - %(levelname)s - %(message)s')
        logger.setLevel(logLevel)
        self.logLevel = logLevel

    def setupOnlineLoop(self, source, block=None):
        """
        Setup the online data processing loop. When ``webapp.run()`` is called after setting this, the source will be
        polled for samples. Each sample is sent to the processing block (if it is not None) and the output samples
        are sent to the websocket.

        :param source: :class:`qmt.AbstractDataSource` instance
        :param block: optional :class:`qmt.Block` instance
        :return: None
        """
        if isinstance(source, str) or isinstance(source, Struct):
            source = PlaybackDataSource(source)
        assert isinstance(source, AbstractDataSource)
        assert block is None or isinstance(block, Block)
        self.source = source
        self.block = block

    def logToFile(self, filename):
        """
        Setup logging of all data sent and received over the websocket to a `jsonl <https://jsonlines.org/>`_ file.

        :param filename: filename ending with '.jsonl'
        :return: None
        """
        self.logfile = WebappLogFile(filename)

    async def emit(self, event, *args, **kwargs):
        for isAsync, callback in self._callbacks[event]:
            if isAsync:
                await callback(self, *args, **kwargs)
            else:
                callback(self, *args, **kwargs)

    def on(self, event, callback):
        """
        Register a callback that is notified on events.

        :param event: string describing events, one of ('message', 'params', 'command', 'connected', 'disconnected')
        :param callback: coroutine or function that will be called with the webapp as first argument and event-specific
            data as second argument
        :return: None
        """
        assert isinstance(event, str)
        self._callbacks[event].append((inspect.iscoroutinefunction(callback), callback))

    async def onMessage(self, message):
        """
        Handles all messages received over the websocket.

        You can override this method or register a callback with ``webapp.on('message', f)``.

        :param message: message as decoded json object (i.e. should be either dict or list)
        :return: None
        """
        await self.emit('message', message)
        if isinstance(message, dict):
            await self.onParams(message)
        else:
            assert isinstance(message, list)
            await self.onCommand(message)

    async def onParams(self, params):
        """
        Handles parameters received over the websocket.

        You can override this method or register a callback with ``webapp.on('params', f)``.

        :param params: parameter dict
        :return: None
        """
        if self.source is not None:
            self.source.setParams(params)
        if self.block is not None:
            if hasattr(self.block, 'setParams'):
                self.block.setParams(params)

        await self.emit('params', params)

    async def onCommand(self, command):
        """
        Handles commands received over the websocket.

        You can override this method or register a callback with ``webapp.on('command', f)``.

        :param command: command (will always be a list)
        :return: None
        """
        handled = False
        if self.source is not None:
            handled = self.source.command(command)
        if self.block is not None and not handled:
            if hasattr(self.block, 'command'):
                self.block.command(command)

        await self.emit('command', command)

    async def onConnected(self, numberOfConnections):
        """
        Called when a client connects to the websocket.

        You can override this method or register a callback with ``webapp.on('connected', f)``.

        :param numberOfConnections: current number of connections
        :return: None
        """
        await self.emit('connected', numberOfConnections)

    async def onDisconnected(self, numberOfConnections):
        """
        Called when a client disconnects to the websocket.

        You can override this method or register a callback with ``webapp.on('disconnected', f)``.

        :param numberOfConnections: current number of connections
        :return: None
        """
        await self.emit('disconnected', numberOfConnections)

    @property
    def connected(self):
        """
        Read-only property, True if there is at least one client connected to the websocket
        """
        return False if self.viewer is None else self.viewer.connectionCount > 0

    def runCommand(self, command):
        """
        Runs a command as if it were received over the websocket.

        :param command: command, must be a list
        :return: None
        """
        assert isinstance(command, list)
        asyncio.get_event_loop().create_task(self.onCommand(command))

    def setParams(self, params):
        """
        Sets parameters as if they were received over the websocket.

        :param params: parameter dict
        :return: None
        """
        assert isinstance(params, dict)
        asyncio.get_event_loop().create_task(self.onParams(params))

    def sendCommand(self, command):
        """
        Sends a command to the webapp via the websocket.

        :param command: command, must be a list
        :return: None
        """
        assert isinstance(command, list)
        self.sendMessage(command)

    def sendSample(self, sample):
        """
        Sends a sample to the webapp via the websocket.

        :param sample: sample dict
        :return: None
        """
        assert isinstance(sample, dict)
        self.sendMessage(sample)

    def sendMessage(self, message):
        """
        Sends an arbitrary message over the websocket.

        :param message: arbitrary object that can be encoded as JSON
        :return: None
        """
        data = toJson(message)
        self.sendQueue.put_nowait(data)

    @property
    def path(self):
        """
        Path of the webapp. Can either be the path to a folder containing an index.html file, a path to a
        html file or a special path (e.g. /view/imubox, /demo/euler-angles). When reading this
        property, a standard absolute file path will always be returned.
        """
        return self._path

    @path.setter
    def path(self, path):
        self._path = _handleSpecialPaths(path, self.prefixes)

    @property
    def dirname(self):
        """
        Read-only property, returns the path to the directory containing the webapp.
        """
        return os.path.dirname(self._path) if self._path.endswith('.html') else self._path

    @property
    def config(self):
        """
        Holds the current configuration as a Struct.

        Can be set to a qmt.Struct, dict or filename. Setting the config while the webapp is running causes a
        'setConfig' command to be sent.
        """
        return self._config

    @config.setter
    def config(self, config):
        self._config = _toStruct(config)
        if self.connected:
            self.sendMessage(['setConfig', self._config.data])

    @property
    def data(self):
        """
        Holds the current data as a Struct.

        Can be set to a qmt.Struct, dict or filename. Setting the data while the webapp is running causes a
        'setData' command to be sent.
        """
        return self._data

    @data.setter
    def data(self, data):
        self._data = _toStruct(data)
        if self.connected:
            self.sendMessage(['reloadData'])

    @property
    def show(self):
        """
        Determines how the webapp should be displayed.

        Possible values are 'window', 'widget', 'chromium', 'iframe', and 'none'.

        window
            Uses the PySide-based viewer and opens the webapp in a window. Does not open a local web server, but serves
            files internally using a custom scheme and uses a QtWebChannel for real-time communication.

        widget
            Uses the PySide-based viewer but does not show the window. Use this to integrate the webapp in a custom
            application.

        chromium
            Uses the aiohttp-based viewer and starts a chromium browser window in app mode (without any toolbars).
            The aiohttp-based viewer opens a local web server and uses a websocket for real-time communication.

        iframe
            Uses the aiohttp-based viewer and creates an iframe in a Jupyter notebook.

        none
            Only starts the aiohttp-based web server.
        """
        return self._show

    @show.setter
    def show(self, show):
        assert show in ('window', 'widget', 'chromium', 'iframe', 'none')
        self._show = show

    @property
    def noLib(self):
        """If set to ``True``, do not serve the special paths ``/lib-qmt``, ``/view``, and ``/demo``."""
        return self._noLib

    @noLib.setter
    def noLib(self, noLib):
        self._noLib = noLib

    @property
    def jsLogLevel(self):
        """
        Logging level for JavaScript messages (PySide viewer only).

        Possible values are 'info', 'warning', 'error', and 'disabled'.

        JavaScript log messages are captured and redicted to a ``js`` logger. By default, only warnings and errors are
        shown. To also print output from ``console.log``, set this property to 'info'.

        An alternative (and more powerful) way to get JavaScript logging output is to use the developer tools. Set the
        environment variable ``QTWEBENGINE_REMOTE_DEBUGGING=5000`` and then open http://localhost:5000/ in a
        Chromium-based browser. For more information, see https://doc.qt.io/qt-5/qtwebengine-debugging.html.
        """
        return self._jsLogLevel

    @jsLogLevel.setter
    def jsLogLevel(self, jsLogLevel):
        assert jsLogLevel in ('info', 'warning', 'error', 'disabled')
        self._jsLogLevel = jsLogLevel

    @property
    def devServerUrl(self):
        """
        Custom URL to load to support using a development server (PySide viewer only).

        This URL is loaded instead of the default ``qmt://app/``. The websocket/webchannel connection will automatically
        work with the ``Backend`` class, but by default, the webapps will try to load data and config from the dev
        server and not from the Python code. To load data and config from Python, use::

            webapp.devServerUrl = 'http://localhost:3000/?config=qmt://app/config.json&data=qmt://app/data.json'
        """
        return self._devServerUrl

    @devServerUrl.setter
    def devServerUrl(self, devServerUrl):
        self._devServerUrl = devServerUrl

    @property
    def host(self):
        """Hostname for the webserver, default: '127.0.0.1' (aiohttp only)"""
        return self._host

    @host.setter
    def host(self, host):
        self._host = str(host)

    @property
    def port(self):
        """Port for the webserver, default: 8000 (aiohttp only)"""
        return self._port

    @port.setter
    def port(self, port):
        self._port = int(port)

    @property
    def autoIncrementPort(self):
        """
        If set to True (default), automatically increment the port number if another server is already running the
        specified port (aiohttp only).
        """
        return self._autoIncrementPort

    @autoIncrementPort.setter
    def autoIncrementPort(self, autoIncrementPort):
        self._autoIncrementPort = bool(autoIncrementPort)

    @property
    def stopOnDisconnect(self):
        """
        If set to True (default), the server is automatically stopped once the websocket connection is closed and not
        immediately reconnected (aiohttp only).
        """
        return self._stopOnDisconnect

    @stopOnDisconnect.setter
    def stopOnDisconnect(self, stopOnDisconnect):
        self._stopOnDisconnect = bool(stopOnDisconnect)

    @property
    def stopOnWindowClose(self):
        """
        If set to True, the server is automatically stopped when the browser window process terminates (aiohttp only).

        Note that this is not used by default because (a) chromium immediately exists when another instance is already
        running and (b) it does not work for iframes embedded in Jupyter notebooks.
        """
        return self._stopOnWindowClose

    @stopOnWindowClose.setter
    def stopOnWindowClose(self, stopOnWindowClose):
        self._stopOnWindowClose = bool(stopOnWindowClose)

    @property
    def iframeWidth(self):
        """Width of the iframe embedded in Jupyter notebooks, default: '100%' (aiohttp only)."""
        return self._iframeWidth

    @iframeWidth.setter
    def iframeWidth(self, iframeWidth):
        self._iframeWidth = iframeWidth

    @property
    def iframeHeight(self):
        """Height of the iframe embedded in Jupyter notebooks, default: '500' (aiohttp only)."""
        return self._iframeHeight

    @iframeHeight.setter
    def iframeHeight(self, iframeHeight):
        self._iframeHeight = iframeHeight

    @property
    def chromiumExecutable(self):
        """
        Custom path to Chromium or Chrome executable that is run when :attr:`show` is set to 'chromium' (aiohttp only).

        By default, several standard paths are searched on Windows. On other platforms, 'chromium-browser' is used.
        """
        return self._chromiumExecutable

    @chromiumExecutable.setter
    def chromiumExecutable(self, chromiumExecutable):
        self._chromiumExecutable = chromiumExecutable

    def shutdown(self):
        asyncio.get_event_loop().create_task(self._shutdown(asyncio.get_event_loop()))

    @staticmethod
    def initialize():
        """
        Initialize the PySide viewer.

        This function is automatically called when a webapp is started. However, the initialization has to happen before
        a QApplication instance is created. In a custom PySide application or when other code creates a QApplication
        (e.g., matplotlib with the Qt5Agg backend), it is necessary to call ``qmt.Webapp.initialize()`` before this
        happens.
        """
        from qmt.utils.webapp_pyside import PysideWebappViewer
        PysideWebappViewer.initialize()

    def _createViewer(self):
        if self.show in ('window', 'widget') and os.getenv('QMT_CHROMIUM_FALLBACK', '0') == '1':
            logging.warning('QMT_CHROMIUM_FALLBACK environment variable set - using the "chromium" fallback')
            self.show = 'chromium'

        if self.show in ('window', 'widget'):
            try:
                if 'PyQt5' in sys.modules:
                    logger.warning('The PyQt5 module is already loaded, which might not work in combination with the '
                                   'PySide-based viewer. This might be caused by importing matplotlib with the Qt5Agg '
                                   'backend. To mitigate this, either call ``qmt.Webapp.initialize()`` early enough, '
                                   'set the environment variable QT_API to PySide2/6, or import PySide2/6.QtCore early '
                                   'in your application. If this does not work, use the Chromium-based viewer by '
                                   'setting  Webapp.show to "chromium" or setting the  QMT_CHROMIUM_FALLBACK '
                                   'environment variable to 1.')
                from qmt.utils.webapp_pyside import PysideWebappViewer
                return PysideWebappViewer(self)
            except ImportError as e:
                if e.name not in ('PySide2', 'PySide6'):
                    raise
                logging.warning(f'PySide not available: "{e}" - using the "chromium" fallback')
                self.show = 'chromium'

        from qmt.utils.webapp_aiohttp import AiohttpWebappViewer
        return AiohttpWebappViewer(self)

    async def _runOnlineLoop(self):
        if self.source is None:
            return

        dpTime = 0.0
        otherTime = 0.0
        dpLoad = 0.0
        count = 0
        await self.source.setup()
        t0 = time.monotonic()
        async for sample in self.source:
            otherTime += time.monotonic() - t0
            t0 = time.monotonic()

            if isinstance(sample, list):  # sample is actually a command
                if sample[0] == '@block':  # only send to processing block
                    if hasattr(self.block, 'command'):
                        self.block.command(sample[1:])
                elif sample[0] == '@webapp':  # only send to webapp
                    self.sendCommand(sample[1:])
                else:  # send to processing block first and then, unless True is returned, to the webapp
                    if hasattr(self.block, 'command'):
                        handled = self.block.command(sample)
                    else:
                        handled = False
                    if handled is not True:
                        self.sendCommand(sample)

                dpTime += time.monotonic() - t0
                t0 = time.monotonic()
                continue

            if self.block is not None:
                if self.logfile is not None:
                    await self.logfile.log('raw_sample', toJson(sample))
                sample = self.block.step(sample)

            if isinstance(sample, dict) or (sample and isinstance(sample[0], str)):
                sample = [sample]  # allow resampling during processing

            dpTime += time.monotonic() - t0
            t0 = time.monotonic()
            count += len(sample)

            totalTime = dpTime + otherTime
            if totalTime > 10.0:
                dpLoad = dpTime/totalTime
                logger.info(f'sent {count} samples in {totalTime:.2f} s, {count/totalTime:.2f} Hz, '
                            f'dpLoad={100*dpLoad:.0f}%, websocket connections: {self.viewer.connectionCount}')
                dpTime, otherTime, count = 0.0, 0.0, 0

            for s in sample:
                if isinstance(s, list):  # command from processing block
                    if s[0] == '@webapp':  # does not have any effect, but remove the first entry
                        self.sendCommand(s[1:])
                    elif s[0] == '@datasource':
                        self.source.command(s)  # send to data source if the first entry is @datasource
                    else:
                        self.sendCommand(s)  # send to webapp by default
                else:
                    s['dpLoad'] = dpLoad
                    self.sendSample(s)
        logger.info('shutting down event loop because data source is done')
        self.shutdown()

    async def _emitRunning(self):
        await self.emit('running')

    async def _shutdown(self, loop, signal=None):
        # cf. https://www.roguelynn.com/words/asyncio-graceful-shutdowns/
        if signal is not None:
            logger.info(f'received signal {signal.name}...')
        if self._loopWasRunning:
            tasks = self._tasks
        else:
            try:
                tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            except AttributeError:  # fallback for Python <3.7
                tasks = [t for t in asyncio.Task.all_tasks(loop) if t is not asyncio.Task.current_task(loop)]
        logger.info(f'cancelling {len(tasks)} outstanding tasks')
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug(f'shutdown complete, results: {results}')
        # if not self._loopWasRunning:
        #     loop.stop()

    def _handleException(self, loop, context):
        # cf. https://www.roguelynn.com/words/asyncio-exception-handling/
        # https://github.com/python/cpython/blob/10ac0cded26d91c3468e5e5a87cecad7fc0bcebd/Lib/asyncio/base_events.py#L1688
        # context["message"] will always be there; but context["exception"] may not
        msg = context.get('exception', context['message'])

        exception = context.get('exception')
        if exception is not None:
            exc_info = (type(exception), exception, exception.__traceback__)
        else:
            exc_info = False

        logger.error(f'caught exception in handler: {msg!r}', exc_info=exc_info)
        loop.create_task(self._shutdown(loop))
