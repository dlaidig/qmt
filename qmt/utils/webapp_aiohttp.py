# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import asyncio
import json
import logging
import os
import shutil
import sys

import aiohttp
from aiohttp import web

from qmt.utils.misc import toJson
from qmt.utils.webapp import AbstractWebappViewer

logger = logging.getLogger(__name__)


def _findChromium():
    candidates = [
        'chromium-browser',
        'chromium',
        'google-chrome',
    ]
    for path in candidates:
        if shutil.which(path) is not None:
            return path
    logger.warning('chromium executable not found, you might want to set webapp.chromiumExecutable manually')
    return 'chromium-browser'


def _findChromiumOnWindows():
    candidates = [
        'C:\\Program Files\\Chromium\\chrome.exe',
        'C:\\Program Files (x86)\\Chromium\\chrome.exe',
        os.path.expandvars('%LOCALAPPDATA%\\Chromium\\Application\\chrome.exe'),
        'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
        'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    logger.warning('chromium executable not found, you might want to set webapp.chromiumExecutable manually')
    return 'chromium-browser'


class AiohttpWebappViewer(AbstractWebappViewer):
    def __init__(self, webapp):
        super().__init__(webapp)
        self._connections = []
        logger.setLevel(webapp.logLevel)

    def setupLoop(self):
        pass

    def createTasks(self, loop):
        runningEvent = asyncio.Event()
        return [
            loop.create_task(self._runServer(runningEvent)),
            loop.create_task(self._runWebsocketSendLoop()),
            loop.create_task(self._runWindow(runningEvent)),
        ]

    @property
    def connectionCount(self):
        return len(self._connections)

    def _buildBrowserWindowCommand(self, url):
        executable = self.webapp.chromiumExecutable
        if executable is None:
            if sys.platform == 'win32':
                executable = _findChromiumOnWindows()
            else:
                executable = _findChromium()
        return [executable, '--disk-cache-dir=/dev/null', '--disk-cache-size=1', f'--app={url}']

    @staticmethod
    async def _disableCache(request, response):
        response.headers['Cache-Control'] = 'no-store'

    async def _runServer(self, runningEvent):
        app = web.Application()
        app.on_response_prepare.append(self._disableCache)
        app.router.add_route('*', '/', self._serveIndex)
        app.router.add_route('*', '/config.json', self._serveConfig)
        app.router.add_route('*', '/data.json', self._serveData)
        app.router.add_route('*', '/favicon.ico', self._serveFavicon)
        app.router.add_route('GET', '/ws', self._handleWebsocket)

        if not self.webapp.noLib:
            path = os.path.join(self.webapp.baseDir, 'lib-qmt')
            app.router.add_static('/lib-qmt', path=path, show_index=True, follow_symlinks=True, name='lib-qmt')

            # always serve included webapps under /demo/ and /view/
            path = os.path.join(self.webapp.baseDir, 'demo')
            app.router.add_static('/demo', path=path, show_index=False, follow_symlinks=True, name='demo')
            path = os.path.join(self.webapp.baseDir, 'view')
            app.router.add_static('/view', path=path, show_index=False, follow_symlinks=True, name='view')

        app.router.add_static('/', path=self.webapp.dirname, show_index=True, follow_symlinks=True, name='static')

        runner = web.AppRunner(app)
        await runner.setup()

        i = 0
        while True:
            try:
                site = web.TCPSite(runner, host=self.webapp.host, port=self.webapp.port+i)
                await site.start()
                break
            except OSError as e:
                if e.errno == 98 and self.webapp.autoIncrementPort and i < 100:
                    logger.info(f'port {self.webapp.port + i} already in use, incrementing...')
                    i += 1
                else:
                    raise
        logger.info(f'running web server on {site.name}')
        self._url = site.name
        runningEvent.set()

        try:
            await asyncio.Event().wait()  # wait until cancelled
        except asyncio.CancelledError:
            logger.info(f'shutting down server on {site.name}')
            await runner.cleanup()
            if self.webapp.logfile is not None:
                await self.webapp.logfile.close()
                self.webapp.logfile = None

    async def _runWindow(self, runningEvent):
        if self.webapp.show not in ('chromium', 'iframe'):
            return
        await runningEvent.wait()

        if self.webapp.show == 'iframe':
            from IPython.display import IFrame, display
            display(IFrame(self._url, width=self.webapp.iframeWidth, height=self.webapp.iframeHeight))
            return

        cmd = self._buildBrowserWindowCommand(self._url)
        logger.debug(f'opening browser window: {cmd!r}')
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE)

        stdout, stderr = await proc.communicate()

        logger.debug('browser window closed')
        logger.debug(f'command: {cmd!r} exited with {proc.returncode}')
        if stdout:
            logger.debug(f'stdout: {stdout.decode(errors="replace")!r}')
        if stderr:
            logger.debug(f'stderr: {stderr.decode(errors="replace")!r}')

        if self.webapp.stopOnWindowClose:
            # see https://bugs.chromium.org/p/chromium/issues/detail?id=68608
            logger.debug('shutting down event loop because browser window closed and stopOnWindowClose is set to True')
            asyncio.get_event_loop().create_task(self._shutdown(asyncio.get_event_loop()))

    async def _serveIndex(self, _):
        if self.webapp.dirname == self.webapp.path:
            filename = os.path.join(self.webapp.path, 'index.html')
        else:
            filename = self.webapp.path
        return web.FileResponse(filename)

    async def _serveConfig(self, _):
        if self.webapp.config is not None:
            body = toJson(self.webapp.config)
            return web.Response(body=body, content_type='application/json')
        path = os.path.join(self.webapp.dirname, 'config.json')
        if not os.path.isfile(path):
            return web.HTTPNotFound()
        return web.FileResponse(path)

    async def _serveData(self, _):
        if self.webapp.data is not None:
            body = toJson(self.webapp.data)
            return web.Response(body=body, content_type='application/json')
        path = os.path.join(self.webapp.dirname, 'data.json')
        if not os.path.isfile(path):
            return web.HTTPNotFound()
        return web.FileResponse(path)

    async def _serveFavicon(self, _):
        path = os.path.join(self.webapp.baseDir, 'lib-qmt', 'favicon.ico')
        if not os.path.isfile(path):
            return web.HTTPNotFound()
        return web.FileResponse(path)

    async def _handleWebsocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self._connections.append(ws)
        logger.info(f'websocket client connected, number of connections: {len(self._connections)}')
        await self.webapp.onConnected(len(self._connections))

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                if self.webapp.logfile is not None:
                    await self.webapp.logfile.log('received', msg.data)
                data = json.loads(msg.data)
                await self.webapp.onMessage(data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.info(f'websocket connection closed with exception {ws.exception()}')

        self._connections.remove(ws)
        logger.info(f'websocket connection closed, number of connections: {len(self._connections)}')
        await self.webapp.onDisconnected(len(self._connections))

        if self.webapp.stopOnDisconnect and len(self._connections) == 0:
            asyncio.get_event_loop().create_task(self._shutdownIfWebsocketDisconnected())

        return ws

    async def _runWebsocketSendLoop(self):
        while True:
            data = await self.webapp.sendQueue.get()
            if self.webapp.logfile is not None:
                await self.webapp.logfile.log('sent', data)
            for ws in self._connections:
                if isinstance(data, bytes):
                    data = data.decode()
                try:
                    await ws.send_str(data)
                except ConnectionResetError:
                    logger.warning('got ConnectionResetError when sending to websocket')
                    self._connections.remove(ws)
                    ws.close()

    async def _shutdownIfWebsocketDisconnected(self):
        logger.debug('last websocket disconnected, waiting for reconnect...')  # e.g. when pressing F5
        await asyncio.sleep(1)
        if len(self._connections) == 0:
            logger.debug('shutting down event loop because websocket disconnected and stopOnDisconnect is set to True')
            self.webapp.shutdown()
