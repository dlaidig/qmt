<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <UiRenderCanvas ref="renderCanvas" @click="rotate"/>
</template>

<script>
import { _, BABYLON, GUI } from '../vendor.js'
import UiRenderCanvas from './UiRenderCanvas.vue'
import { Scene, IMUBox } from '../babylon.js'
import { Quaternion, rad2deg } from '../utils.js'

class IMUScene extends Scene {
    constructor(canvas, options) {
        const defaults = {
            signal: 'quat',
            axisSignal: '',
        }
        options = {...defaults, ...options}
        super(canvas)
        this.options = options
        this.createScene()
    }

    createScene() {
        this.camera = new BABYLON.ArcRotateCamera('camera', 1, 0.8, 10, new BABYLON.Vector3(0, 0, 0), this.scene)
        this.camera.parent = this.cameraTransform
        this.light = new BABYLON.HemisphericLight('light', new BABYLON.Vector3(0,0,1), this.scene)

        const imuboxClass = this.options.imuboxClass ?? IMUBox
        this.imu = new imuboxClass(this.scene, this.options)

        this.ui = GUI.AdvancedDynamicTexture.CreateFullscreenUI('ui', true, this.scene)
        this.textbox = new GUI.TextBlock()
        this.textbox.fontSize = 14
        this.textbox.text = ''
        this.textbox.color = 'white'
        this.textbox.paddingTop = 3
        this.textbox.paddingLeft = 3
        this.textbox.textVerticalAlignment = GUI.Control.VERTICAL_ALIGNMENT_TOP
        this.textbox.textHorizontalAlignment = GUI.Control.HORIZONTAL_ALIGNMENT_LEFT
        this.ui.addControl(this.textbox)
    }

    onSample(sample) {
        const q = new Quaternion(_.get(sample, this.options.signal, [1, 0, 0, 0]))
        this.imu.quat = q
        if (this.options.axisSignal) {
            this.imu.setAxis(_.get(sample, this.options.axisSignal))
        }

        const angles = q.eulerAngles('zxy', true).map(rad2deg)
        const text = 'yaw:\n' + Math.round(angles[0]) + '°\npitch:\n' + Math.round(angles[1]) + '°\nroll:\n' + Math.round(angles[2]) + '°'
        this.textbox.text = text
    }

    rotate() {
        this.camera.alpha = this.camera.alpha + Math.PI/2
    }
}


export default {
    name: 'UiIMUScene',
    components: { UiRenderCanvas },
    props: {
        source: { type: Object, default: {} },
        options: { type: Object, default: {} },
        rotateOnClick: { type: Boolean, default: false },
    },
    data() {
        return {
            scene: null,
        }
    },
    mounted() {
        this.scene = new IMUScene(this.$refs.renderCanvas, this.options)
        this.source.on('sample', sample => this.scene.onSample(sample))
    },
    updated() { console.log('updated UiIMUScene') },
    watch: {
        options(options) {
            this.scene.options.signal = options.signal ?? 'quat'
            this.scene.options.axisSignal = options.axisSignal ?? ''
        }
    },
    methods: {
        rotate() {
            if (!this.rotateOnClick)
                return
            this.scene.rotate()
        }
    }
}
</script>

<style scoped>

</style>
