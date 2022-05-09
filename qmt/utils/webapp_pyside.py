# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import asyncio
import json
import logging
import os
import sys

try:
    if 'PySide2' in sys.modules or os.environ.get('QT_API') == 'PySide2':
        raise ImportError()  # do not try to import PySide6
    from PySide6 import QtWebEngineWidgets
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets
    from PySide6.QtWebEngineQuick import QtWebEngineQuick as QtWebEngine
    from PySide6 import QtWebEngineCore
    from PySide6 import QtWebChannel
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
    from PySide6.QtGui import QAction
except ImportError:
    from PySide2 import QtWebEngineWidgets
    from PySide2 import QtCore
    from PySide2 import QtGui
    from PySide2 import QtWidgets
    from PySide2.QtWebEngine import QtWebEngine
    from PySide2 import QtWebEngineCore
    from PySide2 import QtWebChannel
    from PySide2.QtWebEngineWidgets import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
    from PySide2.QtWidgets import QAction

import qasync
from qmt.utils.misc import toJson
from qmt.utils.webapp import AbstractWebappViewer


logger = logging.getLogger(__name__)


# https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types/Common_types
_mimeTypes = {
    '.babylon': b'application/babylon',
    '.css': b'text/css',
    '.html': b'text/html',
    '.ico': b'image/vnd.microsoft.icon',
    '.jpg': b'image/jpeg',
    '.jpeg': b'image/jpeg',
    '.js': b'text/javascript',
    '.json': b'application/json',
    '.mjs': b'text/javascript',
    '.mp3': b'video/mp3',
    '.mp4': b'video/mp4',
    '.png': b'image/png',
    '.pdf': b'application/pdf',
    '.svg': b'image/svg+xml',
    '.txt': b'text/plain',
    '.webm': b'video/webm',
    '.woff': b'font/woff',
    '.woff2': b'font/woff2',
}


class Connection(QtCore.QObject):
    """An instance of this class is used with the QtWebChannel and replaces the websocket functionality."""
    @QtCore.Slot(str)
    def sendToPython(self, msg):  # called from Javascript code
        self.messageFromWeb.emit(msg)

    @QtCore.Slot(str)
    def sendToWeb(self, msg):  # called from Python code
        self.messageFromPython.emit(msg)

    @QtCore.Slot(str)
    def copyToClipboard(self, text):  # navigator.clipboard does not work with the qrc:// protocol
        QtWidgets.QApplication.clipboard().setText(text)

    messageFromWeb = QtCore.Signal(str)  # triggers Python code
    messageFromPython = QtCore.Signal(str)  # triggers Javascript code


class Interceptor(QtWebEngineCore.QWebEngineUrlRequestInterceptor):
    """
    Redirects http://app/ URLs to qmt://app/. This is needed for the workaround to get fetch working (see the
    javascript function fetchSchemeWorkaround).
    """
    def interceptRequest(self, info):
        url = info.requestUrl()
        if url.scheme() != 'http' or url.host() != 'app':
            return

        url.setScheme('qmt')
        info.redirect(url)


class SchemeHandler(QtWebEngineCore.QWebEngineUrlSchemeHandler):
    """This custom scheme handler handles requests to qmt://app/ urls and serves the webapp files."""
    def __init__(self, webapp, parent=None):
        super().__init__(parent)
        self.webapp = webapp

    # @qasync.asyncSlot(QtWebEngineCore.QWebEngineUrlRequestJob)
    def requestStarted(self, request):
        url = request.requestUrl()
        path = url.path()
        assert url.scheme() == 'qmt'
        assert url.host() == 'app'

        if path in ('/', '/index.html'):
            self._serve(request, self.webapp.path, self.webapp.dirname)
        elif path.startswith(('/lib-qmt/', '/demo/', '/view/')) and not self.webapp.noLib:
            self._serve(request, os.path.join(self.webapp.baseDir, path[1:]), self.webapp.baseDir)
        elif path == '/config.json':
            self._serveConfig(request)
        elif path == '/data.json':
            self._serveData(request)
        else:
            self._serve(request, os.path.join(self.webapp.dirname, path[1:]), self.webapp.dirname)

    def _serve(self, request, path, basedir):
        if basedir is not None:  # check if normalized path is inside basedir
            if os.path.commonpath((basedir, os.path.abspath(path))) != basedir:
                logger.error(f'path "{path}" => "{os.path.abspath(path)}" is outside of basedir "{basedir}"')
                request.fail(request.RequestDenied)
                return

        if not os.path.isfile(path):
            logger.error(f'path "{path}" => "{os.path.abspath(path)}" not found')
            request.fail(request.UrlNotFound)
            return

        # Note: Using aiofiles causes issues on shutdown ("QThread: Destroyed while thread is still running") and
        # does not seem to significantly impact performance.
        # async with aiofiles.open(path, mode='rb') as f:
        #     contents = await f.read()
        with open(path, mode='rb') as f:
            contents = f.read()

        self._reply(request, path, contents)

    def _reply(self, request, path, body):
        buf = QtCore.QBuffer(self)
        request.destroyed.connect(buf.deleteLater)
        buf.open(QtCore.QIODevice.WriteOnly)
        buf.write(body)
        buf.seek(0)
        buf.close()

        ext = os.path.splitext(path)[1]
        mime = _mimeTypes.get(ext, b'application/octet-stream')
        if mime == b'application/octet-stream':
            logger.warning(f'unknown mime type for "{path}"')

        request.reply(mime, buf)

    def _serveConfig(self, request):
        if self.webapp.config is not None:
            body = toJson(self.webapp.config)
            self._reply(request, 'config.json', body)
            return
        path = os.path.join(self.webapp.dirname, 'config.json')
        if not os.path.isfile(path):
            request.fail(request.UrlNotFound)
            return
        self._serve(request, path, self.webapp.dirname)

    def _serveData(self, request):
        if self.webapp.data is not None:
            body = toJson(self.webapp.data)
            self._reply(request, 'data.json', body)
            return
        path = os.path.join(self.webapp.dirname, 'data.json')
        if not os.path.isfile(path):
            request.fail(request.UrlNotFound)
            return
        self._serve(request, path, self.webapp.dirname)


class JsLogger(logging.Logger):
    """
    Custom logging.Logger class that allows for custom filenames and line numbers to be set. This makes it possible to
    show nice JavaScript logs with correct information.

    By default, an error is raised in makeRecord if the ``extras`` passed to the log functions contain 'filename' or
    'fileno'. Note that we first '_filename' and '_fileno' and then later replace the attributes of the record. In
    case the standard logging.Logger is used, setting those values will not cause errors.
    """
    def makeRecord(self, *args, **kwargs):
        record = super().makeRecord(*args, **kwargs)
        if '_filename' in record.__dict__:
            record.__dict__['filename'] = record.__dict__['_filename']
        if '_lineno' in record.__dict__:
            record.__dict__['lineno'] = record.__dict__['_lineno']
        return record


class Page(QWebEnginePage):
    """
    Custom QWebEnginePage that intercepts JavaScript log messages and sends them to the Python logging module.
    """
    LEVELS = {
        QWebEnginePage.InfoMessageLevel: logging.INFO,
        QWebEnginePage.WarningMessageLevel: logging.WARNING,
        QWebEnginePage.ErrorMessageLevel: logging.ERROR,
    }
    LEVEL_CONFIG = {
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'disabled': logging.CRITICAL,
    }

    def __init__(self, profile, parent, jsLogLevel, baseDirs):
        super().__init__(profile, parent)
        self.baseDirs = baseDirs

        # create custom JsLogger class to allow for setting custom file names and line numbers
        oldclass = logging.getLoggerClass()
        logging.setLoggerClass(JsLogger)
        self.logger = logging.getLogger('js')
        logging.setLoggerClass(oldclass)

        self.logger.setLevel(self.LEVEL_CONFIG[jsLogLevel])

    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        if '/@fs/' in sourceID:  # shorten path for static files served by vite
            _, sourceID = sourceID.split('/@fs', maxsplit=1)
            for basedir in self.baseDirs:
                if os.path.commonpath((basedir, os.path.abspath(sourceID))) == basedir:
                    sourceID = os.path.relpath(sourceID, basedir)
                    break
        extra = dict(_filename=sourceID, _lineno=lineNumber)
        self.logger.log(self.LEVELS[level], message, extra=extra)


class WebappWindow(QtWebEngineWidgets.QWebEngineView):
    def __init__(self, webapp, closeEvent):
        super().__init__()
        logger.setLevel(webapp.logLevel)
        self._closeEvent = closeEvent

        icon = QtGui.QIcon(os.path.join(webapp.baseDir, 'lib-qmt/favicon.png'))
        self.setWindowIcon(icon)
        QtWidgets.QApplication.instance().setWindowIcon(icon)

        self.fullScreenAction = QAction('Full Screen', self)
        self.fullScreenAction.setCheckable(True)
        self.fullScreenAction.setShortcut(QtGui.QKeySequence('F11'))
        self.fullScreenAction.toggled.connect(self.onFullScreenToggled)
        self.addAction(self.fullScreenAction)

        self.resetZoomAction = QAction('Reset Zoom', self)
        self.resetZoomAction.setShortcut(QtGui.QKeySequence('Ctrl+0'))
        self.resetZoomAction.triggered.connect(lambda: self.page.setZoomFactor(1.0))
        self.addAction(self.resetZoomAction)

        self.profile = QWebEngineProfile(self)
        self.interceptor = Interceptor(self)
        self.profile.setUrlRequestInterceptor(self.interceptor)
        self.page = Page(self.profile, self, webapp.jsLogLevel, [webapp.baseDir, webapp.dirname])
        self.channel = QtWebChannel.QWebChannel()
        self.page.setWebChannel(self.channel)
        self.connection = Connection()
        self.channel.registerObject('connection', self.connection)

        self.handler = SchemeHandler(webapp, self)
        self.profile.installUrlSchemeHandler(QtCore.QByteArray(b'qmt'), self.handler)

        # this works with http:// but not with qrc://, use web channel instead
        self.page.settings().setAttribute(QWebEngineSettings.JavascriptCanAccessClipboard, True)

        self.titleChanged.connect(lambda title: self.setWindowTitle(title))

        self.setPage(self.page)

        self.page.action(self.page.Reload).setShortcut('F5')
        self.addAction(self.page.action(self.page.Reload))  # to enable the keyboard shortcut
        self.page.action(self.page.SavePage).setVisible(False)
        self.page.action(self.page.ViewSource).setVisible(False)

        # Note: webcam access does not seem to work with the qmt:// scheme
        # self.page.featurePermissionRequested.connect(self.onFeaturePermissionRequested)

        self.load(QtCore.QUrl(webapp.devServerUrl if webapp.devServerUrl else 'qmt://app/'))

        # generate hash from webapp path to save and load different window states for each webapp
        h = QtCore.QCryptographicHash(QtCore.QCryptographicHash.Sha256)
        h.addData(webapp.path)
        self.configHash = bytes(h.result()).hex()
        settings = QtCore.QSettings('qmt', 'qmt-webapp')
        self.restoreGeometry(settings.value(f'geometry/webapp-{self.configHash}'))

    def closeEvent(self, event):
        super().closeEvent(event)
        settings = QtCore.QSettings('qmt', 'qmt-webapp')
        settings.setValue(f'geometry/webapp-{self.configHash}', self.saveGeometry())
        self._closeEvent.set()

    def contextMenuEvent(self, event):
        try:  # Qt5
            menu = self.page.createStandardContextMenu()
        except AttributeError:  # Qt 6
            menu = self.createStandardContextMenu()
        menu.addAction(self.resetZoomAction)
        menu.addAction(self.fullScreenAction)
        menu.popup(event.globalPos())

    def onFullScreenToggled(self, enabled):
        if enabled:
            self.showFullScreen()
        else:
            self.showNormal()

    # def onFeaturePermissionRequested(self, origin, feature):
    #     # Webcam access does not work with the qmt:// scheme (but works with http://).
    #     # When this is working, this function should ask the user for permission and offer to persistently store the
    #     # permission for this webapp (using self.configHash to identify the webapp).
    #     print('request', origin, feature)
    #     self.page.setFeaturePermission(origin, feature, QtWebEngineWidgets.QWebEnginePage.PermissionGrantedByUser)


class PysideWebappViewer(AbstractWebappViewer):
    _INITIALIZED = False

    def __init__(self, webapp):
        super().__init__(webapp)
        self.mainWindow = None

    @staticmethod
    def initialize():
        if PysideWebappViewer._INITIALIZED:
            return
        if QtCore.QCoreApplication.instance() is not None:
            logger.warning('A QApplication instance already exists when trying to initialize the QtWebEngine for use '
                           'with the PySide2 webapp viewer. This may, for example, be caused by creating matplotlib '
                           'plots with the Qt5Agg backend. Please call "qmt.Webapp.initialize()" early in your '
                           'application.')
        QtWebEngine.initialize()
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        scheme = QtWebEngineCore.QWebEngineUrlScheme(b'qmt')
        scheme.setFlags(QtWebEngineCore.QWebEngineUrlScheme.SecureScheme)
        scheme.setFlags(QtWebEngineCore.QWebEngineUrlScheme.CorsEnabled)
        QtWebEngineCore.QWebEngineUrlScheme.registerScheme(scheme)
        PysideWebappViewer._INITIALIZED = True

    def setupLoop(self):
        self.initialize()
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication()
        loop = asyncio.get_event_loop()
        if not isinstance(loop, qasync.QEventLoop):
            if loop.is_running():
                logger.warning(f'non-qasync event loop is already running: {loop}')
            loop = qasync.QEventLoop(app)
            asyncio.set_event_loop(loop)
        app.setQuitOnLastWindowClosed(False)
        # Instead of using setQuitOnLastWindowClosed, matplotlib connects the lastWindowClosed signal to the quit slot:
        # https://github.com/matplotlib/matplotlib/blob/6b84f24217a9/lib/matplotlib/backends/backend_qt.py#L124
        # If we don't disconnect this, the QApplication will quit and the event loop will be destroyed too early.
        try:
            app.lastWindowClosed.disconnect()
        except RuntimeError:
            pass

    def createTasks(self, loop):
        return [
            loop.create_task(self._run()),
            loop.create_task(self._runWebchannelSendLoop()),
        ]

    @property
    def connectionCount(self):
        return 0 if self.mainWindow is None else 1

    def prepareShutdown(self):
        self.mainWindow.page.deleteLater()

    async def _run(self):
        closeEvent = asyncio.Event()

        self.mainWindow = WebappWindow(self.webapp, closeEvent)
        self.mainWindow.connection.messageFromWeb.connect(self._onMessageFromWeb)
        if self.webapp.show != 'widget':
            self.mainWindow.show()
        await self.webapp.onConnected(1)
        await closeEvent.wait()
        self.mainWindow.page.deleteLater()  # fixes "Release of profile requested but WebEnginePage still not deleted"
        self.mainWindow = None
        self.webapp.shutdown()
        # re-connect this signal to make matplotlib happy if plots are shown after running a webapp
        QtWidgets.QApplication.instance().lastWindowClosed.connect(QtWidgets.QApplication.instance().quit)

    @qasync.asyncSlot(str)
    async def _onMessageFromWeb(self, message):
        if self.webapp.logfile is not None:
            await self.webapp.logfile.log('received', message)
        data = json.loads(message)
        await self.webapp.onMessage(data)

    async def _runWebchannelSendLoop(self):
        while True:
            data = await self.webapp.sendQueue.get()
            if self.webapp.logfile is not None:
                await self.webapp.logfile.log('sent', data)
            if isinstance(data, bytes):
                data = data.decode()
            if self.mainWindow is not None:
                self.mainWindow.connection.sendToWeb(data)
