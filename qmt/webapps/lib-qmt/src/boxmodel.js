// SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
// SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>
//
// SPDX-License-Identifier: MIT

import { BABYLON, _, Emitter } from './vendor.js'
import { Quaternion, COLORS, clip, vecNorm, deg2rad } from './utils.js'
import { IMUBox } from './babylon.js'


import TEMPLATES from '../assets/boxmodel.json'
export { TEMPLATES }

export class BoxModel extends Emitter {
    constructor(scene, config) {
        super()
        this.scene = scene
        this.config = this._processConfig(config)
        // make it possible to select boxes by clicking on them
        this.actionManager = new BABYLON.ActionManager(this.scene)
        this.clickAction = new BABYLON.ExecuteCodeAction(BABYLON.ActionManager.OnLeftPickTrigger, evt => {
            this.emit('pick', evt.source.name)
        })
        this._pickable = false

        this.segments = {} // segment boxes
        this.joints = {} // joint spheres
        this.segmentNames = []

        this.segments.root = new BoxModelSegment(scene, {...TEMPLATES.common.defaults, sphere: 0, dimensions: [0.0, 0.0, 0.0]}, 'root')

        this._createChain('root')
    }

    onSample(sample) {
        for (let name of this.segmentNames) {
            if (this.segments[name].signal !== null && this.segments[name].signal !== '') {
                const quat = new Quaternion(_.get(sample, this.segments[name].signal))
                this.segments[name].quat = quat
            } else {
                this.segments[name].refresh()
            }
        }
        this.emit('sample', sample)
    }

    refresh() {
        for (let name of this.segmentNames) {
            this.segments[name].refresh()
        }
    }

    set position(position) {
        this.segments.root.sphere.position = new BABYLON.Vector3(...position)
    }

    get position() {
        const pos = this.segments.root.sphere.position
        return [pos.x, pos.y, pos.z]
    }

    setProp(segments, prop, val) {
        if (segments === 'all')
            segments = this.segmentNames
        else if (!Array.isArray(segments)) {
            segments = [segments]
        }
        // in case val == null
        if (typeof val === 'object' && !Array.isArray(val) && val !== null) {
            for (let name of segments) {
                if (val.hasOwnProperty(name))
                    this.segments[name][prop] = val[name]
                else
                    this.segments[name][prop] = val
            }

        } else {
            for (let name of segments) {
                this.segments[name][prop] = val
            }

        }
    }

    setVisilibity(segments, val) {
        this.setProp(segments, 'visibility', val)
    }

    setOpacity(segments, val) {
        this.setProp(segments, 'opacity', val)
    }

    setEdges(segments, val) {
        this.setProp(segments, 'edges', val)
    }

    setColor(segments, val) {
        this.setProp(segments, 'color', val)
    }

    setSignal(segments, val) {
        this.setProp(segments, 'signal', val)
    }

    setImuVisibility(segments, val) {
        this.setProp(segments, 'imuVisibility', val)
    }

    setSelectedSegments(segments) {
        if (segments === 'all')
            segments = this.segmentNames
        if (segments === 'none')
            segments = []
        for (let name of this.segmentNames) {
            this.segments[name].selected = segments.includes(name)
        }
    }

    get pickable() {
        return this._pickable
    }

    set pickable(val) {
        if (val && !this._pickable)
            this.actionManager.registerAction(this.clickAction)
        if (!val && this._pickable)
            this.actionManager.unregisterAction(this.clickAction)
        this._pickable = val
    }

    _processConfig(config) {
        // recursively merge base templates
        const out = this._mergeBaseConfig(config)

        // apply values found in defaults to every single segment
        if (typeof out.defaults !== 'undefined') {
            for (let segment of Object.keys(out.segments)) {
                out.segments[segment] = {...out.defaults, ...out.segments[segment]}
            }
        }

        return out
    }

    _mergeBaseConfig(config) {
        const out = _.cloneDeep(config)

        const base = typeof out.base !== 'undefined' ? this._mergeBaseConfig(TEMPLATES[out.base]) : TEMPLATES['common']
        out.defaults = {...base.defaults, ...out.defaults}
        for (let segment of Object.keys(base.segments)) {
            out.segments[segment] = {...base.segments[segment], ...out.segments[segment]}
        }
        return out
    }

    _createChain(base) {
        for (let segment of Object.keys(this.config.segments).sort()) {
            if (this.config.segments[segment].parent === base) {
                if (this.segmentNames.includes(segment)) {
                    console.log('loop in chain definition found', segment)
                    return
                }
                this._createSegment(segment, this.config.segments[segment])
                this.segmentNames.push(segment)
                this._createChain(segment) // recursively create child elements
            }
        }
        if (base === 'root' && this.segmentNames.length !== Object.keys(this.config.segments).length) {
            console.log('chain definition seems to be broken, some elements are not reachable from root element',
                Object.keys(this.config.segments).filter(e => !this.segmentNames.includes(e)))
        }
    }

    _createSegment(name, options) {
        console.log('create', name)
        const parent = this.segments[options.parent]
        this.segments[name] = new BoxModelSegment(this.scene, options, name, parent)

        this.segments[name].box.actionManager = this.actionManager
    }

    dispose() {
        for (let segmentName of this.segmentNames) {
            this.segments[segmentName].dispose()
        }
    }
}

class BoxModelSegment {
    constructor(scene, options, name, parent) {
        this.scene = scene
        if (typeof options.color === 'string') {
            if (options.color[0] === '#') {
                options.color = BABYLON.Color4.FromHexString(options.color.length === 7 ? options.color + 'FF' : options.color)
            } else {
                options.color = COLORS[options.color]
            }
        }
        if (Array.isArray(options.q_segment2sensor)) {
            options.q_segment2sensor = new Quaternion(options.q_segment2sensor)
        }

        this.options = options
        this.name = name
        this.parent = parent

        this.changes = {} // stores manually changed options to be able to create config with only changed values
        this.origSeg2Sensor = this.q_segment2sensor

        this._opacity = 1.0

        this.sphere = null
        this.box = null
        this._signal = null

        if (typeof options.signal === 'string') {
            this.signal = options.signal
        }
        this.recreateMeshes()
        this.quat = new Quaternion(options.quat)
    }

    get quat() {
        return this._quat
    }

    set quat(quat) {
        this._quat = quat
        if (this.changes.q_segment2sensor) {
            // when tuning the segment-to-sensor orientation, we assume that the current quat is the orientation
            // obtained with the old segment-to-sensor orientation and therefore adjust it
            quat = quat.multiply(this.origSeg2Sensor.conj()).multiply(this.q_segment2sensor)
        }
        if (typeof this.changes.heading_offset !== 'undefined') {
            const delta = this.changes.heading_offset - (this.options.heading_offset ?? 0)
            let deltaQuat = Quaternion.fromAngleAxis(deg2rad(delta), [0, 0, 1])

            quat = deltaQuat.multiply(quat)
        }
        this._actualQuat = quat
        if (this.parent && !this.options.relative) { // calculate orientation relative to parent
            quat = this.parent.actualQuat.conj().multiply(quat)
        }
        this.sphere.rotationQuaternion = quat.babylon()

        if (this.imu) {
            this._updateImuBox()
        }
    }

    get actualQuat() {
        return this._actualQuat
    }

    refresh() {
        this.quat = this._quat
    }

    set visibility(val) {
        this.sphere.visiblity = val
        this.box.visiblity = val
        if (this.imu) {
            this.imu.visibility = val
        }
    }

    get opacity() {
        return this._opacity
    }

    set opacity(val) {
        val = clip(val, 0, 1)
        this._opacity = val
        this.sphere.material.alpha = val
        this.box.material.alpha = val
        if (this.imu) {
            this.imu.opacity = val
        }
        this.box.isPickable = val > 0
    }

    set edges(val) {
        if (val) {
            this.box.enableEdgesRendering()
        } else {
            this.box.disableEdgesRendering()
        }
    }

    set color(color) {
        const face = this.options.face
        for (let v = 4*face; v < 4*face+4; v++) {
            this.boxColors[v * 4] = color.r
            this.boxColors[v * 4 + 1] = color.g
            this.boxColors[v * 4 + 2] = color.b
            this.boxColors[v * 4 + 3] = color.a ?? 1
        }
        this.box.setVerticesData(BABYLON.VertexBuffer.ColorKind, this.boxColors)
    }

    set selected(val) {
        if (val) {
            this.box.edgesColor = new BABYLON.Color4(0.95, 0.59, 0.18,  this.options.opacity)
            this.box.edgesWidth = 20.0
        } else {
            this.box.edgesColor = new BABYLON.Color4(0, 0, 0,  this.options.opacity)
            this.box.edgesWidth = 14.0
        }
    }

    get imubox_cs() {
        return this.options.imubox_cs
    }

    set imubox_cs(cs) {
        if (this.options.imubox_cs === cs)
            return

        this.options.imubox_cs = cs
        this.changes.imubox_cs = cs
        this._recreateImuBox()
    }

    get q_segment2sensor() {
        return this.options.q_segment2sensor ?? Quaternion.identity()
    }

    set q_segment2sensor(quat) {
        this.options.q_segment2sensor = quat
        this.changes.q_segment2sensor = quat
        if (this.imu)
            this._updateImuBox()
        else
            this._recreateImuBox()
        this.refresh()
    }

    set heading_offset(val) {
        this.changes.heading_offset = val
        if (Math.abs(this.changes.heading_offset - this.options.heading_offset) < 1e-8) {
            delete this.changes.heading_offset
        }
        if (this.imu)
            this._updateImuBox()
        else
            this._recreateImuBox()
        this.refresh()
    }

    get heading_offset() {
        return this.changes.heading_offset ?? (this.options.heading_offset ?? 0)
    }

    get signal() {
        return this._signal
    }

    set signal(signal) {
        this._signal = signal
    }

    dispose() {
        if (this.sphere) {
            this.sphere.dispose(true, true)
            this.sphere = null
        }
        if (this.box) {
            this.box.dispose(true, true)
            this.box = null
        }
        if (this.imu) {
            this.imu.dispose()
            this.imu = null
        }
    }

    recreateMeshes() {
        if (this.sphere) {
            this.sphere.dispose(true, true)
            this.sphere = null
        }
        if (this.box) {
            this.box.dispose(true, true)
            this.box = null
        }

        // create joint sphere
        const diameter = this.options.sphere * this.options.scale
        const sphereOpts = {diameterX: diameter, diameterY: diameter, diameterZ: diameter}
        this.sphere = BABYLON.MeshBuilder.CreateSphere('sphere', sphereOpts, this.scene)
        this.sphere.material = new BABYLON.StandardMaterial('mat', this.scene)
        this.sphere.material.diffuseColor = new BABYLON.Color3(1, 1, 1)
        if (this.options.parent === 'root' || this.options.sphere === 0) {
            this.sphere.visibility = 0
            this.sphere.material.diffuseColor = new BABYLON.Color3(1, 0, 0)
        }

        // create box
        const faceColors = new Array(6)
        faceColors[this.options.face] = this.options.color
        const boxOpts = {
            faceColors: faceColors,
            width: this.options.dimensions[0] * this.options.scale,
            height: this.options.dimensions[1] * this.options.scale,
            depth: this.options.dimensions[2] * this.options.scale,
        }
        this.box = BABYLON.MeshBuilder.CreateBox(this.name, boxOpts, this.scene)
        this.box.material = new BABYLON.StandardMaterial('mat', this.scene)
        this.box.edgesWidth = 14.0
        this.box.edgesColor = new BABYLON.Color4(0, 0, 0,  this.options.opacity)
        this.boxColors = this.box.getVerticesData(BABYLON.VertexBuffer.ColorKind)
        if (this.options.dimensions[0] === 0 && this.options.dimensions[1] === 0 && this.options.dimensions[2] === 0) {
            this.box.visibility = 0
        }

        // the joint sphere is parented to the sphere of the parent segment with an position offset
        // specified by position_rel and position_abs
        if (this.parent) {
            const pos = this._calcPosition(this.parent.options.dimensions, this.parent.options.scale, this.options.position_rel, this.options.position_abs)
            this.sphere.position = new BABYLON.Vector3(...pos)
            this.sphere.parent = this.parent.sphere
        }

        // the box is parented to the sphere with a position offset specified by origin_rel and origin_abs
        const pos = this._calcPosition(this.options.dimensions, this.options.scale, this.options.origin_rel, this.options.origin_abs)
        this.box.position = new BABYLON.Vector3(...pos).scale(-1)
        this.box.parent = this.sphere

        this.edges = this.options.edges
        this.opacity = this.options.opacity

        this._recreateImuBox()
    }

    _recreateImuBox() {
        if (this.imu) {
            this.imu.dispose()
            this.imu = null
        }

        // create IMU box
        if (this.options.imubox_show && this.options.q_segment2sensor) {
            const cs = this.options.imubox_cs
            const imuOpts = {cs, scale: this.options.scale * this.options.imubox_scale, led: false, axes: true, ...this.options.imubox_options}
            const imuboxClass = this.options.imubox_class ?? IMUBox
            this.imu = new imuboxClass(this.scene, imuOpts)
            this.imu.box.parent = this.box

            this._updateImuBox()
            this.imu.opacity = this._opacity
        }
    }

    _updateImuBox() {
        const cs = this.options.imubox_cs

        // get unit vector pointing upwards in IMU coordinate system
        const vec = [0, 0, 0]
        if (cs.indexOf('U') >= 0) {
            vec[cs.indexOf('U')] = 1
        } else {
            console.assert(cs.indexOf('D') >= 0)
            vec[cs.indexOf('D')] = -1
        }

        // transform vec into sensor coordinate system and calculate distance from center to box face
        const vecSegment = this.q_segment2sensor.conj().rotate(vec)
        const baseDist = vecNorm([
            vecSegment[0]*this.options.dimensions[0],
            vecSegment[1]*this.options.dimensions[1],
            vecSegment[2]*this.options.dimensions[2]
        ])

        // calculate distance from segment box center to IMU box center
        const dist = this.options.imubox_distance_rel*baseDist/2 + this.options.imubox_distance_abs
        this.imu.position = vecSegment.map(e => e*this.options.scale*dist)

        this.imu.quat = this.q_segment2sensor.conj()
    }

    _calcPosition(dim, scale, rel, abs) {
        const position = [0, 0, 0]
        for (let i = 0; i < 3; i++) {
            position[i] = scale * (dim[i] * rel[i] + abs[i])
        }
        return position
    }

    set imuVisibility(val) {
        this.options.imubox_show = val
        this._recreateImuBox()

    }

    get imuVisibility() {
        return this.options.imubox_show
    }
}
