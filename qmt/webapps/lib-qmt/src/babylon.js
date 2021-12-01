// SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
//
// SPDX-License-Identifier: MIT

import {BABYLON, Emitter, fetchSchemeWorkaround} from './index.js'

import { COLORS, Quaternion, vecNorm, clip } from './utils.js'

export class Scene extends Emitter {
    constructor(canvasOrScene, fovMode='both') {
        super()
        this.engineIsShared = (canvasOrScene instanceof BABYLON.Engine)
        if (this.engineIsShared) {
            this.engine = canvasOrScene
            this.canvas = this.engine.getRenderingCanvas()
        } else {
            if (typeof canvasOrScene === 'string') {
                this.canvas = document.getElementById(canvasOrScene)
            } else if (typeof canvasOrScene.$el !== 'undefined') { // UiRenderCanvas
                this.canvas = canvasOrScene.$el
            } else {
                this.canvas = canvasOrScene
            }
            // preserveDrawingBuffer and stencil is required for creating screenshots/videos or for use with Puppeteer
            this.engine = new BABYLON.Engine(this.canvas, true, { preserveDrawingBuffer: true, stencil: true })
        }
        this.scene = new BABYLON.Scene(this.engine)
        this.scene.useRightHandedSystem = true

        // set this as parent for the cameras to get global coordinates with z pointing upwards
        this.cameraTransform = new BABYLON.TransformNode('earth')
        this.cameraTransform.rotationQuaternion = new Quaternion(1, 1, 0, 0).babylon()

        this.fovMode = fovMode

        if (!this.engineIsShared) {
            this.engine.runRenderLoop(() => this.scene.render())

            new ResizeObserver(() => setTimeout(() => this.engine.resize(), 0)).observe(this.canvas)

            this.engine.onResizeObservable.add(() => this.updateCameraFov())
            setTimeout(() => this.updateCameraFov(), 0)
        }
    }

    updateCameraFov() {
        if (!this.camera)
            return

        // https://www.html5gamedevs.com/topic/38890-engineresize-according-to-screen-width/
        if (this.fovMode == 'horizontal' || this.engine.getRenderHeight() > this.engine.getRenderWidth()) {
            this.camera.fovMode = BABYLON.Camera.FOVMODE_HORIZONTAL_FIXED
        } else {
            this.camera.fovMode = BABYLON.Camera.FOVMODE_VERTICAL_FIXED
        }
    }
}

export const IMU_CS_OPTIONS = [
    'FLU', 'LBU', 'BRU', 'RFU', // z pointing upwards
    'LFD', 'BLD', 'RBD', 'FRD', // z pointing downwards
    'LUF', 'BUL', 'RUB', 'FUR', // y pointing upwards
    'FDL', 'LDB', 'BDR', 'RDF', // y pointing downwards
    'UFL', 'ULB', 'UBR', 'URF', // x pointing upwards
    'DLF', 'DBL', 'DRB', 'DFR', // x pointing downwards
]

// vecs = {'F': [1, 0, 0], 'B': [-1, 0, 0], 'R': [0, 0, 1], 'L': [0, 0, -1], 'U': [0, 1, 0], 'D': [0, -1, 0]}
// for cs in options:
//     q = qmt.qinv(qmt.quatFrom2Axes(x=vecs[cs[0]], y=vecs[cs[1]]))
//     print(f'{cs}: {[int(round(e/np.max(np.abs(q)))) for e in q]},')


const IMU_QUAT_BOX2IMU = {
    FLU: [1, 1, 0, 0],
    LBU: [1, 1, -1, -1],
    BRU: [0, 0, -1, -1],
    RFU: [1, 1, 1, 1],
    LFD: [1, -1, -1, 1],
    BLD: [0, 0, -1, 1],
    RBD: [1, -1, 1, -1],
    FRD: [1, -1, 0, 0],
    LUF: [1, 0, -1, 0],
    BUL: [0, 0, -1, 0],
    RUB: [1, 0, 1, 0],
    FUR: [1, 0, 0, 0],
    FDL: [0, -1, 0, 0],
    LDB: [0, -1, 0, 1],
    BDR: [0, 0, 0, -1],
    RDF: [0, -1, 0, -1],
    UFL: [0, -1, -1, 0],
    ULB: [1, 1, 1, -1],
    UBR: [1, 0, 0, -1],
    URF: [1, -1, -1, -1],
    DLF: [1, 1, -1, 1],
    DBL: [0, -1, 1, 0],
    DRB: [1, -1, 1, 1],
    DFR: [1, 0, 0, 1],
}


const IMUBoxTextureLoader = {
    originalFiles: new Map(), // cache that stores promises to original svg files as text (key: texture path)
    dataStrings: new Map(), // cache that stores promises to modified svg files as data url (key: option json)

    async get(options) {
        const key = JSON.stringify(options)
        if (!this.dataStrings.has(key)) {
            this.dataStrings.set(key, this._generateTexture(options))
        }
        return [key, await this.dataStrings.get(key)]
    },

    async _generateTexture(options) {
        // console.log('get', JSON.stringify(options))
        if (!this.originalFiles.has(options.texture)) {
            this.originalFiles.set(options.texture, this._loadFile(options.texture))
        }
        const text = await this.originalFiles.get(options.texture)

        const parser = new DOMParser ()
        const svg = parser.parseFromString (text, 'image/svg+xml')

        // remove template layer and make sure that background layer is visible
        svg.querySelector('#layer_template').innerHTML = ''
        svg.querySelector('#layer_background').style['display'] = ''

        // make sure the coordinate system string is one of the 24 valid combinations
        console.assert(IMU_CS_OPTIONS.includes(options.cs))

        // determine which of the 4 coordinate system sketch layers to use
        var cs;
        if (options.cs.includes('F') && options.cs.includes('L'))
            cs = 1
        else if (options.cs.includes('B') && options.cs.includes('L'))
            cs = 2
        else if (options.cs.includes('B') && options.cs.includes('R'))
            cs = 3
        else
            cs = 4

        // hide other 3 coordinate system layers
        for (var i = 1; i < 5; i++) {
            svg.querySelector(`#layer_cs${i}`).style['display'] = i === cs ? '' : 'none'
        }

        // replace text
        svg.querySelector(`#cs${cs}letter tspan`).innerHTML = options.letter
        svg.querySelector('#left tspan').innerHTML = options.left.replace('{letter}', options.letter).trim()
        svg.querySelector('#right tspan').innerHTML = options.right.replace('{letter}', options.letter).trim()

        // set x/y/z letter elements
        svg.querySelector(`#cs${cs}ax${options.cs[0].replace('D', 'U')} tspan`).innerHTML = 'x'
        svg.querySelector(`#cs${cs}ax${options.cs[1].replace('D', 'U')} tspan`).innerHTML = 'y'
        svg.querySelector(`#cs${cs}ax${options.cs[2].replace('D', 'U')} tspan`).innerHTML = 'z'

        if (options.cs.includes('D'))
            svg.querySelector(`#cs${cs}dot`).style['display'] = 'none'
        else
            svg.querySelector(`#cs${cs}cross`).style['display'] = 'none'

        const svgString = new XMLSerializer().serializeToString(svg.documentElement)
        const dataString = 'data:image/svg+xml;base64,'+btoa(unescape(encodeURIComponent(svgString)))
        return dataString
    },

    async _loadFile(path) {
        // console.log('load', path)
        const response = await fetch(fetchSchemeWorkaround(path))
        return await response.text()
    }
}

// import.meta.env.BASE_URL is '/' in built library and when using "npm run dev" and '' when running "npm run build" with base=''
export class IMUBox {
    constructor(scene, options) {
        this.scene = scene
        const defaults = {
            // 3 letter string describing local coordinate system wrt. box (x, y, and z).
            // Possible letters: F (forward), B (backward), L (left), R (right), U (up), D (down)
            // FLU means that the x axis is pointing forward (F) along the longitudinal axis, the y axis is pointing
            // to the left (L) and the z axis is pointing upwards (i.e. the CS sketch will have a dot, not a cross).
            // Smartphones often use RFU (https://www.w3.org/TR/gyroscope/#model), and FLU is also a common convention.
            cs: 'RFU',
            dimX: 5.5,
            dimY: 3.5,
            dimZ: 1.3,
            texture: import.meta.env.BASE_URL + 'lib-qmt/assets/texture_imu.svg', // set to false to use colors on top and front side
            letter: '',
            left: 'IMU {letter}',
            right: 'IMU {letter}',
            color: COLORS.C1,
            led: true,
            axes: false,
            scale: 1.0,
        }
        options = {...defaults, ...options}

        const x = options.dimX, y = options.dimY, z = options.dimZ
        const boxOpts = {
            width: x*options.scale,
            height: z*options.scale,
            depth: y*options.scale,
        }
        const mat = new BABYLON.StandardMaterial('mat', scene)

        if (options.texture) {
            const textureOptions = {
                texture: options.texture,
                cs: options.cs,
                letter: options.letter,
                left: options.left,
                right: options.right
            }
            IMUBoxTextureLoader.get(textureOptions).then(([key, dataString]) => {
                mat.diffuseTexture = BABYLON.Texture.LoadFromDataString(key, dataString, scene)
            })

            const width = Math.max(4 + y + 2 * z, 3 + y + x)
            const height = 3 + 2 * x
            boxOpts.faceUV = [
                new BABYLON.Vector4((2 + y) / width, (2 + y + z) / height, (2 + y + x) / width, (2 + y) / height), // right (-y)
                new BABYLON.Vector4((2 + y + x) / width, (3 + y + z) / height, (2 + y) / width, (3 + y + 2 * z) / height), // left (+y)
                new BABYLON.Vector4((2 + y) / width, (1 + y) / height, (2 + y + z) / width, 1 / height), // front (+x)
                new BABYLON.Vector4((3 + y + z) / width, (1 + y) / height, (3 + y + 2 * z) / width, 1 / height), // back (-x)
                new BABYLON.Vector4(1 / width, (2 + 2 * x) / height, (1 + y) / width, (2 + x) / height), // top (+z)
                new BABYLON.Vector4((1 + y) / width, 1 / height, 1 / width, (1 + x) / height), // bottom (-z)
            ]
        } else {
            const faceColors = new Array(6)
            faceColors[4] = options.color // top
            faceColors[2] = options.color // front
            boxOpts.faceColors = faceColors
        }

        this.box = BABYLON.MeshBuilder.CreateBox('box', boxOpts, scene)
        this.box.material = mat

        this.q_Box2Imu = new Quaternion(IMU_QUAT_BOX2IMU[options.cs])


        if (options.led) {
            this.led = BABYLON.MeshBuilder.CreateSphere('led', {}, scene)
            this.led.parent = this.box
            const ledScale = x*y*z*0.01*options.scale
            this.led.scaling = new BABYLON.Vector3(ledScale, ledScale, ledScale)
            this.led.position.x = -0.8*x/2*options.scale
            this.led.position.y = z/2*options.scale
            this.led.position.z = 0.75*y/2*options.scale
            this.led.material = new BABYLON.StandardMaterial('ledmaterial', scene)
            this.led.material.diffuseColor = COLORS.grey
            this.ledOn = false
            this.defaultEmissiveColor = this.led.material.emissiveColor
            setInterval(this._toggleLed.bind(this), 500)
        }

        if (options.axes) {
            const axisLen = Math.max(x, y, z)*1.5*options.scale
            const q = this.q_Box2Imu.conj()
            this.x_axis = new Arrow(this.scene, {parent: this.box, color: COLORS.C3, vector: q.rotate([axisLen, 0, 0]), origin: q.rotate([-axisLen/2, 0, 0]), diameter: 0.1*options.scale})
            this.y_axis = new Arrow(this.scene, {parent: this.box, color: COLORS.C2, vector: q.rotate([0, axisLen, 0]), origin: q.rotate([0, -axisLen/2, 0]), diameter: 0.1*options.scale})
            this.z_axis = new Arrow(this.scene, {parent: this.box, color: COLORS.C0, vector: q.rotate([0, 0, axisLen]), origin: q.rotate([0, 0, -axisLen/2]), diameter: 0.1*options.scale})
        }

        this.quat = Quaternion.identity()
    }

    get quat() {
        return this._quat
    }

    set quat(quat) {
        if (Array.isArray(quat)) {
            quat = new Quaternion(quat)
        }
        this._quat = quat
        this.box.rotationQuaternion = this._quat.multiply(this.q_Box2Imu).babylon()
    }

    set visibility(visible) {
        this.box.visibility = visible
        if (this.led)
            this.led.visibility = visible
        if (this.x_axis) {
            this.x_axis.visibility = visible
            this.y_axis.visibility = visible
            this.z_axis.visibility = visible
        }
    }

    set opacity(val) {
        val = clip(val, 0, 1)
        this.box.material.alpha = val
        if (this.led)
            this.led.material.alpha = val
        if (this.x_axis) {
            this.x_axis.opacity = val
            this.y_axis.opacity = val
            this.z_axis.opacity = val
        }
    }

    dispose() {
        this.box.dispose(true, true)
        if (this.led)
            this.led.dispose(true, true)
        if (this.x_axis) {
            this.x_axis.dispose()
            this.y_axis.dispose()
            this.z_axis.dispose()
        }
    }

    _toggleLed() {
        if (this.ledOn)
            this.led.material.emissiveColor = this.defaultEmissiveColor
        else
            this.led.material.emissiveColor = COLORS.white
        this.ledOn = !this.ledOn
    }

    setAxis(axis) {
        if (this.axis === undefined) {
            this.axis = BABYLON.MeshBuilder.CreateCylinder('axis', {height: 7, diameter: 0.1}, this.scene)
            this.axis.parent = this.box
        }

        // default orientation is in z direction
        const angle = Math.acos(axis[2]) // dot([0 0 1], j)
        const rotAxis = [-axis[1], axis[0], 0] // cross([0 0 1], j)
        const qAxis = Quaternion.fromAngleAxis(angle, rotAxis)
        this.axis.rotationQuaternion = this.q_Box2Imu.multiply(qAxis).multiply(this.q_Box2Imu.conj()).babylon()
    }
}

export class Arrow {
    constructor(scene, options) {
        const defaults = {
            parent: undefined,
            diameter: 0.1,
            scale: 1.0,
            origin: [0, 0, 0],
            color: undefined,
            vector: [0,1,0],
            arrowSize: 5,
        }
        options = {...defaults, ...options}

        this.scene = scene
        this.mesh = BABYLON.Mesh.CreateCylinder('cylinder', 1, options.diameter, options.diameter, 12, 1, scene)
        this.mesh.parent = options.parent

        this.mesh.material = new BABYLON.StandardMaterial('matarrow', scene)
        if (options.color) {
            this.mesh.material.diffuseColor = options.color
        }

        this.arrowheadLength = options.arrowSize*options.diameter
        if (options.arrowSize) {
            this.arrowhead = BABYLON.Mesh.CreateCylinder('arrowhead', this.arrowheadLength, 1e-10, this.arrowheadLength/2, 12, 1, scene)
            this.arrowhead.parent = options.parent
            this.arrowhead.material = new BABYLON.StandardMaterial('matarrowhead', scene)
        }

        if (options.color) {
            this.mesh.material.diffuseColor = options.color
            if (this.arrowhead)
                this.arrowhead.material.diffuseColor = options.color
        }

        this.origin = new BABYLON.Vector3(...options.origin)
        this.quat = new Quaternion(1,0,0,0)
        this.offset = [0,0,0]
        this.scale = options.scale

        this.setVector(options.vector)
    }

    setVector(vec) {
        this.quat = Quaternion.rotationFrom2Vectors([0,1,0], vec)
        const length = vecNorm(vec)*this.scale - this.arrowheadLength
        const offset = this.quat.rotate([0, length/2, 0])
        this.mesh.position = this.origin.add(new BABYLON.Vector3(offset[0], offset[1], offset[2]))
        this.mesh.scaling.y = length
        this.mesh.rotationQuaternion = this.quat.babylon()
        if (this.arrowhead) {
            const pos = this.quat.rotate([0, length + this.arrowheadLength/2, 0])
            this.arrowhead.position = new BABYLON.Vector3(pos[0]+this.origin.x, pos[1]+this.origin.y, pos[2]+this.origin.z)
            this.arrowhead.rotationQuaternion = this.quat.babylon()
        }
    }

    set visibility(visible) {
        this.mesh.visibility = visible
        if (this.arrowhead)
            this.arrowhead.visibility = visible
    }

    set opacity(val) {
        val = clip(val, 0, 1)
        this.mesh.material.alpha = val
        if (this.arrowhead)
            this.arrowhead.material.alpha = val
    }

    dispose() {
        this.mesh.dispose(true, true)
        if (this.arrowhead) {
            this.arrowhead.dispose(true, true)
        }
    }
}
