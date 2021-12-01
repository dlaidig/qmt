# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import asyncio
import logging
import multiprocessing

from qmt.utils.webapp import Webapp
from qmt.utils.datasource import ProcessDataSourceConnection


logger = logging.getLogger(__name__)


async def _readPipe(pipe, webapp):
    frameAvailable = None
    try:
        event = asyncio.Event()  # https://stackoverflow.com/a/62098165
        asyncio.get_event_loop().add_reader(pipe.fileno(), event.set)
        frameAvailable = event
    except NotImplementedError:
        # add_reader is not supported on Windows with the ProactorEventLoop:
        # https://docs.python.org/3/library/asyncio-platforms.html#asyncio-platform-support
        logger.info('Webapp.runInProcess: falling back to polling because add_reader is not supported')

    while True:
        while pipe.poll():
            msg = pipe.recv()
            if isinstance(msg, dict):
                webapp.sendSample(msg)
            elif isinstance(msg, list):
                webapp.sendCommand(msg)
            elif msg == 'shutdown':
                webapp.shutdown()
            else:
                raise RuntimeError(f'unknown message received: {msg}')

        if frameAvailable is None:
            await asyncio.sleep(0.01)
        else:
            frameAvailable.clear()
            if not pipe.poll():
                await frameAvailable.wait()


def _run(pipe, attrs):
    webapp = Webapp()
    for k, v in attrs.items():
        setattr(webapp, k, v)

    def onParams(_, params):
        pipe.send(['params', params])

    def onCommand(_, command):
        pipe.send(['command', command])

    async def onRunning(_):
        asyncio.get_event_loop().create_task(_readPipe(pipe, webapp))

    webapp.on('params', onParams)
    webapp.on('command', onCommand)
    webapp.on('running', onRunning)
    webapp.run()


class WebappProcessConnection(ProcessDataSourceConnection):
    """
    Helper for :meth:`Webapp.runInProcess`. An instance of this class is returned and can be used to communicate with
    the webapp. See the base class :class:`qmt.ProcessDataSourceConnection` for documentation on most methods.
    """
    def __init__(self, webapp):
        pipe, targetPipe = multiprocessing.Pipe()
        super().__init__(pipe)

        self.webapp = webapp

        keys = ('path', 'config', 'data', 'show', 'noLib', 'jsLogLevel', 'devServerUrl', 'host', 'port',
                'autoIncrementPort', 'stopOnDisconnect', 'stopOnWindowClose', 'chromiumExecutable', 'source', 'block')
        attrs = {k: getattr(webapp, k) for k in keys}

        self.process = multiprocessing.Process(target=_run, args=(targetPipe, attrs))
        self.process.start()

    def shutdown(self):
        """Shut down the webapp."""
        try:
            self._pipe.send('shutdown')
        except ConnectionResetError:
            self._closed()
