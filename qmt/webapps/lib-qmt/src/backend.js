// SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
//
// SPDX-License-Identifier: MIT

import {_, Emitter, fetchSchemeWorkaround} from './index.js'

export class Backend extends Emitter {
    constructor(options) {
        super()

        const defaults = {
            ws: 'auto',
            defaultConfig: {},
            autoPlay: true,
            webSocketDataSourceOptions: {},
            loadJsonFiles: true,
            jsonDataSourceOptions: {},
        }
        options = {...defaults, ...options}
        this.options = options

        let url = options.ws
        const urlParams = new URLSearchParams(window.location.search)
        if (url === 'auto') {
            url =  urlParams.get('ws') ?? 'local'
        }
        if (url === 'local') {
            if (typeof qt !== 'undefined' && typeof qt.webChannelTransport !== 'undefined') {
                url = 'webchannel'
            } else {
                url = ((window.location.protocol === 'https:') ? 'wss://' : 'ws://') + window.location.host + '/ws'
            }
        }
        this.ws = new WebSocketDataSource(url, {...options.webSocketDataSourceOptions, open: false, reconnect: false})
        this.playback = new JsonDataSource(undefined, options.jsonDataSourceOptions)
        this.lastSample = undefined
        this._hooks = []

        this.ws.on('sample', sample => {
            if (this.mode !== 'online') {
                if (!this.playback.paused) {
                    this.playback.pause()
                }
                this.mode = 'online'
                this.emit('modeChanged', this.mode)
            }
            for (let hook of this._hooks) {
                sample = hook(sample)
            }
            this.lastSample = sample
            this.emit('sample', this.lastSample)
        })

        this.playback.on('sample', sample => {
            if (this.mode !== 'playback') {
                this.mode = 'playback'
                this.emit('modeChanged', this.mode)
            }
            for (let hook of this._hooks) {
                sample = hook(sample)
            }
            this.lastSample = sample
            this.emit('sample', this.lastSample)
        })

        this.ws.on('command', command => {
            if (command[0] === 'setData') {
                this.playback.data = command[1]
                this.emit('data', command[1])
                if (this.options.autoPlay && command[1]) {
                    this.playback.play()
                }
            } else if (command[0] === 'reloadData') {
                const dataUrl = urlParams.get('data') ?? './data.json'
                fetch(fetchSchemeWorkaround(dataUrl))
                .then(response => {
                    if (!response.ok) {
                        throw `HTTP response ${response.status} ${response.statusText}`
                    }
                    return response.json()
                })
                .then(data => {
                    console.log(`loaded ${dataUrl}`, data)
                    this.ws.emit('command', ['setData', data])
                })
                .catch(error => {
                    console.log(`failed to load ${dataUrl}`, error)
                })
            } else if (command[0] === 'setConfig') {
                this.emit('config', command[1])
            } else {
                this.emit('command', command)
            }
        })

        this._mode = 'init'
        this._configLoaded = false
        this._dataLoaded = false
        this.config = undefined
        this._data = undefined

        if (options.loadJsonFiles) {
            const configUrl = urlParams.get('config') ?? './config.json'
            if (configUrl[0] === '{') {
                console.log('loading config from URL string')
                this.config = JSON.parse(configUrl)
                this._configLoaded = true
            } else {
                fetch(fetchSchemeWorkaround(configUrl))
                    .then(response => {
                        if (!response.ok) {
                            throw `HTTP response ${response.status} ${response.statusText}`
                        }
                        return response.json()
                    })
                    .then(data => {
                        console.log(`loaded ${configUrl}`, data)
                        this.config = data
                    })
                    .catch(error => {
                        console.log(`failed to load ${configUrl}`, error)
                    })
                    .finally(() => {
                        this._configLoaded = true
                        this._init()
                    })
            }

            const dataUrl = urlParams.get('data') ?? './data.json'
            fetch(fetchSchemeWorkaround(dataUrl))
                .then(response => {
                    if (!response.ok) {
                        throw `HTTP response ${response.status} ${response.statusText}`
                    }
                    return response.json()
                })
                .then(data => {
                    console.log(`loaded ${dataUrl}`, data)
                    this._data = data
                })
                .catch(error => {
                    console.log(`failed to load ${dataUrl}`, error)
                })
                .finally(() => {
                    this._dataLoaded = true
                    this._init()
                })
        } else {
            this._configLoaded = true
            this._dataLoaded = true
            this._init()
        }
    }

    _init() {
        if (!this._configLoaded || !this._dataLoaded) {
            return
        }

        if (this._data && this._data.config) {  // config stored in data.config overrides config.json
            this.config = this._data.config
        }
        if (this.config === undefined) {
            this.config = this.options.defaultConfig
        } else { // add undefined config entries
            this.config = {...this.options.defaultConfig, ...this.config}
        }
        if (this.config.autoPlay !== undefined) { // autoPlay from config.json overrides autoPlay from constructor
            this.options.autoPlay = this.config.autoPlay
        }

        this.playback.data = this._data
        this.emit('config', this.config)
        this.emit('data', this._data)

        if (this.options.autoPlay && this._data) {
            this.playback.play()
        }

        if (this.config.ws !== false) {
            this.ws.openSocket()
        }
    }

    // Adds a callback that modifies or replaces each sample before it is being sent out. The callback must return
    // the modified or new sample.
    addProcessingHook(callback) {
        this._hooks.push(callback)
    }

    // Sends parameters whenever a property changes. Source must either be a Vue object (i.e., `this` inside a
    // component) or an Emitter subclass that emits a 'change' event. By default, the property name is used as the
    // parameter name. Use the optional name argument to set a different name.
    addParam(source, property, name=null) {
        name = name ?? property
        if(source instanceof Object && '$watch' in source) {
            source.$watch(property, val => this.sendParam(name, val), {immediate: true})
        } else if (source instanceof Emitter) {
            source.on('change', () => this.sendParam(name, source[property]))
            this.sendParam(name, source[property])
        } else {
            console.error('Cannot watch source. Must be Vue object or Emitter.')
        }
    }

    // Sends a single parameter. If multiple parameters are sent directly after each other, they are combined in
    sendParam(name, value) {
        this.ws.sendParam(name, value)
    }

    // Sends multiple parameters. Params must be an object.
    sendParams(params) {
        this.ws.sendParams(params)
    }

    // Sends a command. Command must be an array.
    sendCommand(command) {
        this.ws.sendCommand(command)
    }
}

export class WebSocketDataSource extends Emitter {
    constructor(url, options) {
        super()
        this.url = url

        const defaults = {
            open: true,
            reconnect: true,
        }
        options = {...defaults, ...options}
        this.options = options

        this.parameters = {}
        this.open = false
        this.sendQueue = []
        this.paramSendQueue = {}
        this.paramSendTimeout = null

        if (options.open) {
            this.openSocket()
        }
    }

    openSocket() {
        console.log('opening WebSocket:', this.url)
        if (this.url === 'webchannel') {
            this.socket = new WebchannelConnection()
        } else {
            this.socket = new WebSocket(this.url)
        }
        if (!this.socket) {
            alert('Opening WebSocket failed!')
        }

        this.socket.onopen = this.onOpen.bind(this)
        this.socket.onmessage = this.onMessage.bind(this)
        this.socket.onerror = this.onError.bind(this)
        this.socket.onclose = this.onClose.bind(this)
    }

    onOpen(open) {
        console.log("WebSocket has been opened", open, this)
        this.open = true
        if (!_.isEmpty(this.parameters)) { // send all parameters to ensure a consistent state
            this.paramSendQueue = {}
            this.sendMessage(this.parameters)
        }
        for (let message of this.sendQueue) {
            this.socket.send(message)
        }
        this.sendQueue = []
        this.emit('open')
    }

    onMessage(message) {
        // console.log(message.data)
        const msg = JSON.parse(message.data)
        if (Array.isArray(msg)) {
            this.emit('command', msg)
            return
        }

        this.lastSample = msg

        this.emit('sample', this.lastSample)
    }

    onError(error) {
        console.log('WebSocket error:', error, this)
        this.emit('error', error)
    }

    onClose(close) {
        console.log('WebSocket has been closed', close, this)
        this.open = false
        this.emit('close', close)
        if (this.options.reconnect)
            this.openSocket()
    }

    sendParam(name, value) {
        this.parameters[name] = value
        this.paramSendQueue[name] = value
        if (this.open) {
            clearTimeout(this.paramSendTimeout)
            this.paramSendTimeout = setTimeout(this._flushParamQueue.bind(this), 0)
        }
    }

    sendParams(params) {
        this.parameters = {...this.parameters, ...params}
        if (this.open) {
            this.sendMessage(params)
        }
    }

    sendCommand(command) {
        this._flushParamQueue() // preserve order when parameters and commands are sent together
        this.sendMessage(command)
    }

    sendMessage(message) {
        const msg = JSON.stringify(message)
        if (!this.open) {
            this.sendQueue.push(msg)
        } else {
            this.socket.send(msg)
        }
    }

    _flushParamQueue() {
        if (!_.isEmpty(this.paramSendQueue)) {
            this.sendMessage(this.paramSendQueue)
        }
        this.paramSendQueue = {}
    }
}

export class JsonDataSource extends Emitter {
    constructor(url, options) {
        super()

        const defaults = {
            fps: 30,
            speed: 1,
            reverse: false,
            loop: false,
        }
        options = {...defaults, ...options}

        this._data = undefined
        this._paused = true
        this._timer = undefined
        this._currentTime = 0
        this._currentIndex = -1
        this._lastTick = 0
        this._sampleCount = 0
        this._keys = []
        this.lastSample = undefined

        this.fps = options.fps
        this.speed = options.speed
        this.reverse = options.reverse
        this.loop = options.loop

        this.url = url
    }

    get fps() {
        return this._fps
    }

    set fps(fps) {
        if (!this._paused) {
            this.pause()
            this.play()
        }
        this._fps = fps
    }

    get url() {
        return this._url
    }

    set url(url) {
        this._url = url
        if (url) {
            $.getJSON(url, this._dataLoaded.bind(this))
        } else {
            this.data = undefined
        }
    }

    get data() {
        return this._data
    }

    set data(data) {
        this._data = data
        this._sampleCount = (this.data && this.data.t) ? this.data.t.length : 0  // use this.data to allow overriding of getter!
        this._startTime = (this.data && this.data.t) ? this.data.t[0] : 0
        this._currentIndex = -1
        this._currentTime = this._startTime
        this._keys = JsonDataSource._findKeys(this.data, this._sampleCount, '')
        this.emit('ready')
    }

    isLoaded() {
        return Boolean(this.data)
    }

    get sampleCount() {
        return this._sampleCount
    }

    get currentIndex() {
        return this._currentIndex
    }

    set currentIndex(i) {
        const index = Math.max(0, Math.min(i, this.sampleCount))
        if (index === this._currentIndex) {
            return
        }
        this._currentIndex = index
        this.sendCurrentSample()
    }

    get currentTime() {
        return this._currentTime
    }

    set currentTime(t) {
        const dataT = this.data.t
        const N = this.sampleCount
        let i = this.currentIndex

        while (i+1 < N && dataT[i+1] < t + 1e-6) {
            i++
        }
        while (i > 0 && dataT[i] > t + 1e-6) {
            i--
        }

        this.currentIndex = i
        this._currentTime = Math.max(Math.min(t, dataT[dataT.length-1]), this._startTime)
    }

    play() {
        clearInterval(this._timer)
        this._timer = setInterval(this._tick.bind(this), 1000.0/this.fps)
        this._lastTick = Date.now()
        this._paused = false
        this.emit('play')
    }

    pause() {
        clearInterval(this._timer)
        this._paused = true
        this.emit('pause')
    }

    get paused() {
        return this._paused
    }

    stop() {
        clearInterval(this._timer)
        this._paused = true
        this._currentIndex = -1
        this._currentTime = this._startTime
        this.emit('stop')
    }

    _dataLoaded(data) {
        console.log('json loaded:', data, this)
        this.data = data
    }

    _tick() {
        const now = Date.now()
        const deltaT = (this.reverse ? -1 : 1) * this.speed * (now - this._lastTick)/1000.0
        // console.log('deltaT', deltaT)
        this._lastTick = now

        this.currentTime = this.currentTime + deltaT

        if (this.currentIndex >= this.sampleCount-1) {
            console.log('reached end')
            this.stop()
            if (this.loop) {
                this.play()
            }
        }
    }

    sendCurrentSample() {
        const i = this.currentIndex
        const data = this.data

        let sample = {
            ind: i,
            length: this.sampleCount,
        }

        for (let key of this._keys) {
            _.set(sample, key, _.get(data, key)[i])
        }

        this.lastSample = sample
        this._currentTime = sample.t
        this.emit('sample', sample)
    }

    static _findKeys(data, N, prefix) {
        if (N === 0) {
            return []
        }
        const keys = []

        _.forOwn(data, function(v, k) {
            if(_.isArray(v) && v.length === N) {
                keys.push(prefix + k)
            } else if(_.isPlainObject(v)) {
                keys.push(...JsonDataSource._findKeys(v, N, prefix + k + '.'))
            }
        })

        return keys
    }
}

class WebchannelConnection {
    constructor() {
        this._loadWebchannel()
        this.conn = null
    }

    _loadWebchannel() {
        // console.log('running in QtWebEngine -- loading qwebchannel.js')
        const head = document.getElementsByTagName('head')[0]
        const script = document.createElement('script')
        script.src = 'qrc:///qtwebchannel/qwebchannel.js'
        script.type = 'text/javascript'
        script.onload = () => this._openWebchannel()
        head.appendChild(script)
    }

    _openWebchannel() {
        // console.log('webchannel.js loaded -- opening transport')
        new QWebChannel(qt.webChannelTransport, channel => this._setupWebchannel(channel))
    }

    _setupWebchannel(channel) {
        // console.log('transport opened -- setting up connection')
        this.conn = channel.objects.connection
        this.conn.messageFromPython.connect(msg => this.onmessage({data: msg}))
        this.onopen(this)

        // provide custom function to copy text to clipboard since the navigator.clipboard does not work with qrc://
        if (typeof navigator.clipboard === 'undefined') {
            navigator.clipboard = { writeText: text => this.conn.copyToClipboard(text) }
        }
    }

    send(message) {
        this.conn.sendToPython(message)
    }

    onopen(open) {}
    onmessage(message) {}
    onerror(error) {}
    onclose(close) {}
}
