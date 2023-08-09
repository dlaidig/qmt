// SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
//
// SPDX-License-Identifier: MIT

import * as BABYLON from '@babylonjs/core/Legacy/legacy'

export const COLORS = {
    C0: new BABYLON.Color4(31/255, 119/255, 180/255, 1), // matplotlib C0
    C1: new BABYLON.Color4(255/255, 127/255, 14/255, 1), // matplotlib C1
    C2: new BABYLON.Color4(44/255, 160/255, 44/255, 1), // matplotlib C2
    C3: new BABYLON.Color4(214/255, 39/255, 40/255, 1), // matplotlib C3
    C4: new BABYLON.Color4(148/255, 103/255, 189/255, 1), // matplotlib C4
    C5: new BABYLON.Color4(140/255, 86/255, 75/255, 1), // matplotlib C5
    C6: new BABYLON.Color4(227/255, 119/255, 194/255, 1), // matplotlib C6
    C7: new BABYLON.Color4(127/255, 127/255, 127/255, 1), // matplotlib C7
    C8: new BABYLON.Color4(188/255, 189/255, 34/255, 1), // matplotlib C8
    C9: new BABYLON.Color4(23/255, 190/255, 207/255, 1), // matplotlib C9
    red: new BABYLON.Color4(1, 0, 0, 1),
    green: new BABYLON.Color4(0, 1, 0, 1),
    blue: new BABYLON.Color4(0, 0, 1, 1),
    white: new BABYLON.Color4(1, 1, 1, 1),
    black: new BABYLON.Color4(0/255, 0/255, 0/255, 1),
    darkgrey: new BABYLON.Color4(46/255, 52/255, 54/255, 1), // oxygen dark grey #2E3436
    grey: new BABYLON.Color4(136/255, 138/255, 133/255, 1), // oxygen grey #888A85
    lightgrey: new BABYLON.Color4(211/255, 215/255, 207/255, 1), // oxygen light grey #D3D7CF
}

const _axisIdentifiers = {
    1: 1, 'x': 1, 'X': 1, 'i': 1,
    2: 2, 'y': 2, 'Y': 2, 'j': 2,
    3: 3, 'z': 3, 'Z': 3, 'k': 3,
};

export class Quaternion {
    constructor(...args) {
        if (args.length === 1) {
            if (args[0] instanceof Quaternion)
                this.array = args[0].array.slice()
            else
                this.array = args[0]
        } else {
            this.array = args
        }

        this.normalizeInPlace()
    }

    static identity() {
        return new Quaternion(1, 0, 0, 0)
    }

    static fromAngleAxis(angle, axis) {
        if (axis === 'x') {
            axis = [1, 0, 0]
        } else if (axis === 'y') {
            axis = [0, 1, 0]
        } else if (axis === 'z') {
            axis = [0, 0, 1]
        }
        console.assert(axis.length === 3, axis)
        axis = normalizeVec(axis)

        return new Quaternion(Math.cos(angle/2), axis[0]*Math.sin(angle/2), axis[1]*Math.sin(angle/2), axis[2]*Math.sin(angle/2))
    }

    static fromEulerAngles(angles, convention, intrinsic) {
        console.assert(convention.length === 3, convention)
        if (intrinsic) {
            let quat = Quaternion.fromAngleAxis(angles[0], convention[0])
            quat = quat.multiply(Quaternion.fromAngleAxis(angles[1], convention[1]))
            quat = quat.multiply(Quaternion.fromAngleAxis(angles[2], convention[2]))
            return quat
        } else {
            let quat = Quaternion.fromAngleAxis(angles[0], convention[0])
            quat = Quaternion.fromAngleAxis(angles[1], convention[1]).multiply(quat)
            quat = Quaternion.fromAngleAxis(angles[2], convention[2]).multiply(quat)
            return quat
        }
    }

    static rotationFrom2Vectors(v1, v2) {
        const n1 = vecNorm(v1)
        const n2 = vecNorm(v2)
        const angle = Math.acos((v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2])/n1/n2)
        const axis = angle === 0.0 ? [1, 0, 0] : cross(v1, v2)
        return Quaternion.fromAngleAxis(angle, axis)
    }

    static random() {
        return new Quaternion(randn(), randn(), randn(), randn())
    }

    get w() {return this.array[0]}
    get x() {return this.array[1]}
    get y() {return this.array[2]}
    get z() {return this.array[3]}
    set w(val) {this.array[0] = val}
    set x(val) {this.array[1] = val}
    set y(val) {this.array[2] = val}
    set z(val) {this.array[3] = val}

    angle() {
        return 2*Math.acos(this.w)
    }

    conj() {
        return new Quaternion(this.array[0], -this.array[1], -this.array[2], -this.array[3])
    }

    norm() {
        return vecNorm(this.array)
    }

    normalizeInPlace() {
        const norm = this.norm()
        this.array = this.array.map(x => x/norm)
    }

    normalized() {
        const norm = this.norm();
        return new Quaternion(this.array.map(x => x/norm))
    }

    multiply(other) {
        return new Quaternion(
            this.w * other.w - this.x * other.x - this.y * other.y - this.z * other.z,
            this.w * other.x + this.x * other.w + this.y * other.z - this.z * other.y,
            this.w * other.y - this.x * other.z + this.y * other.w + this.z * other.x,
            this.w * other.z + this.x * other.y - this.y * other.x + this.z * other.w
        )
    }


    eulerAngles(convention, intrinsic) {
        console.assert(convention.length === 3, convention)
        console.assert(intrinsic === true || intrinsic === false, intrinsic)

        if (intrinsic)
            convention = convention.split('').reverse().join('')

        const a = _axisIdentifiers[convention[0]]
        const b = _axisIdentifiers[convention[1]]
        const c = _axisIdentifiers[convention[2]]
        let d = 'invalid'
        if (a === c) {
            if (a !== 1 && b !== 1)
                d = 1
            else if (a !== 2 && b !== 2)
                d = 2
            else
                d = 3
        }

        console.assert(b !== a && b !== c, [a, b, c])

        // sign factor depending on the axis order
        let s
        if (b === (a % 3) + 1)  // cyclic order
            s = 1
        else  // anti-cyclic order
            s = -1

        let angle1, angle2, angle3, gimbalLock

        if (a === c) {  // proper Euler angles
            angle1 = Math.atan2(this.array[a] * this.array[b] - s * this.array[d] * this.array[0], this.array[b] * this.array[0] + s * this.array[a] * this.array[d])
            angle2 = Math.acos(clip(this.array[0] ** 2 + this.array[a] ** 2 - this.array[b] ** 2 - this.array[d] ** 2, -1, 1))
            angle3 = Math.atan2(this.array[a] * this.array[b] + s * this.array[d] * this.array[0], this.array[b] * this.array[0] - s * this.array[a] * this.array[d])
            gimbalLock = Math.abs(angle2) < 1e-7 || Math.abs(angle2-Math.PI) < 1e-7
        } else {  // Tait-Bryan
            angle1 = Math.atan2(2 * (this.array[a] * this.array[0] + s * this.array[b] * this.array[c]),
                this.array[0] ** 2 - this.array[a] ** 2 - this.array[b] ** 2 + this.array[c] ** 2)
            angle2 = Math.asin(clip(2 * (this.array[b] * this.array[0] - s * this.array[a] * this.array[c]), -1, 1))
            angle3 = Math.atan2(2 * (s * this.array[a] * this.array[b] + this.array[c] * this.array[0]),
                this.array[0] ** 2 + this.array[a] ** 2 - this.array[b] ** 2 - this.array[c] ** 2)
            gimbalLock = Math.abs(angle2-Math.PI/2) < 1e-7 || Math.abs(angle2+Math.PI/2) < 1e-7
        }

        if (gimbalLock) {
            // get quaternion corresponding to second angle (which is well-defined)
            const axis2 = [0, 0, 0]
            axis2[b-1] = 1
            const quat2 = Quaternion.fromAngleAxis(angle2, axis2)

            // get quaternion corresponding to first angle (assuming the third angle is zero)
            const quat1 = intrinsic ? this.multiply(quat2.conj()) : quat2.conj().multiply(this)

            // get angle along first rotation axis
            const axis1 = [0, 0, 0]
            axis1[intrinsic ? c - 1 : a - 1] = 1
            angle1 = quat1.project(axis1)[0]

            return [angle1, angle2, 0]
        }


        if (intrinsic)
            return [angle3, angle2, angle1]
        else
            return [angle1, angle2, angle3]
    }

    rotate(v) {
        console.assert(v.length === 3, v)
        return [
            (1 - 2*this.array[2]*this.array[2] - 2*this.array[3]*this.array[3])*v[0] + 2*v[1]*(this.array[2]*this.array[1] - this.array[0]*this.array[3]) + 2*v[2]*(this.array[0]*this.array[2] + this.array[3]*this.array[1]),
            2*v[0]*(this.array[0]*this.array[3] + this.array[2]*this.array[1]) + v[1]*(1 - 2*this.array[1]*this.array[1] - 2*this.array[3]*this.array[3]) + 2*v[2]*(this.array[2]*this.array[3] - this.array[1]*this.array[0]),
            2*v[0]*(this.array[3]*this.array[1] - this.array[0]*this.array[2]) + 2*v[1]*(this.array[0]*this.array[1] + this.array[3]*this.array[2]) + v[2]*(1 - 2*this.array[1]*this.array[1] - 2*this.array[2]*this.array[2])
        ]
    }

    project(axis) {
        console.assert(axis.length === 3, axis)
        axis = normalizeVec(axis)
        const phi = wrapToPi(2 * Math.atan2(axis[0] * this.x + axis[1] * this.y + axis[2] * this.z, this.w))
        const qProj = Quaternion.fromAngleAxis(phi, axis)
        const qRes = qProj.conj().multiply(this)
        return [phi, qRes.angle(), qProj, qRes]
    }

    babylon() {
        return new BABYLON.Quaternion(this.array[1], this.array[2], this.array[3], this.array[0]).normalize()
    }
}

export function clip(value, lower, upper) {
    return Math.max(lower, Math.min(value, upper))
}

export function vecNorm(vec) {
    const sqSum = vec.reduce((pv, cv) => pv+cv*cv, 0)
    return Math.sqrt(sqSum)
}

export function normalizeVec(vec) {
    const norm = vecNorm(vec)
    return vec.map(x => x/norm)
}

export function cross(a, b) {
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]
}

export function rad2deg(val) {
    if (Array.isArray(val))
        return val.map(rad2deg)
    return val*180.0/Math.PI
}

export function deg2rad(val) {
    if (Array.isArray(val))
        return val.map(deg2rad)
    return val*Math.PI/180.0
}

export function wrapToPi(val) {
    if (Array.isArray(val))
        return val.map(wrapToPi)
    // return (val + Math.PI) % (2 * Math.PI) - Math.PI;
    return (((val + Math.PI) % (2 * Math.PI)) + (2 * Math.PI)) % (2 * Math.PI) - Math.PI
}

export function round(val, digits=0) {
    if (Array.isArray(val))
        return val.map(v => round(v, digits))
    return Math.round(val * Math.pow(10, digits) + Number.EPSILON) / Math.pow(10, digits)
}

export function randn() {
    // normally distributed random number
    // Box–Muller transform, https://stackoverflow.com/q/25582882
    return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random())
}

export function fetchSchemeWorkaround(url) {
    // Workaround for missing fetch support with custom schemes when using the QtWebEngine viewer.
    //
    // This viewer uses the special scheme qmt://, which is necessary to add a custom handler for network request
    // to serve the files without running a local web server. However, this custom scheme cannot be used with
    // fetch and will cause the following error:
    //     js: Fetch API cannot load qmt://app/data.json. URL scheme "qmt" is not supported.
    //
    // XMLHttpRequest does not seem to have this problem.
    //
    // It seems like currently (2021-10-29, Qt 5.15.2) there is no proper way to solve this:
    // https://stackoverflow.com/questions/64892161/qtwebengine-fetch-api-fails-with-custom-scheme
    // https://bugreports.qt.io/browse/QTBUG-88830
    //
    // Now the workaround: A custom QWebEngineUrlRequestInterceptor will redirect http://app/ URLs to qmt://app/.
    // If the javascript code fetches, e.g., http://app/data.json instead of ./data.json or qmt://app/data.json,
    // fetch works without problems and the custom scheme handler is called on the Python side.
    //
    // Unfortunately, that means that this function has to be called on all relative urls when using fetch:
    //     fetch(fetchSchemeWorkaround('./data.json')).then ....

    const abs = new URL(url, document.baseURI).href
    if (abs.startsWith('qmt://')) {
        return 'http://' + abs.slice(6)
    } else if (abs.startsWith('qmt:/')) {
        return 'http://app/' + abs.slice(5)
    }
    return url
}
