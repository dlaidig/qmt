// SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
//
// SPDX-License-Identifier: MIT

// This stub makes it possible to build a version of lib-qmt without the BabylonJS dependency. To enable this, pass
// babylon: false in the vite-plugin-qmt options.

export class Color4 {
    constructor(r, g, b, a) {
        // this class is functional enough for non-babylon usage in lib-qmt
        // (i.e., allows for the generation of hex color codes)
        this.r = r
        this.g = g
        this.b = b
        this.a = a
    }

    toHexString(returnAsColor3=false) {
        // https://stackoverflow.com/a/39077686
        const vals = returnAsColor3 ? [this.r, this.g, this.b] : [this.r, this.g, this.b, this.a]
        return '#' + vals.map(x => {
            const hex = Math.round(255*x).toString(16).toUpperCase()
            return hex.length === 1 ? '0' + hex : hex
        }).join('')
    }
}

export class Engine {}
export class Scene {}
export class Camera {}
export class Vector3 {}
export class Vector4 {}
export class StandardMaterial {}
export class MeshBuilder {}
export class Mesh {}
export class Texture {}
export class Quaternion {}
