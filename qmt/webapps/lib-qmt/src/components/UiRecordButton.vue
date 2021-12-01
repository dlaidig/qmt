<!--
SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <UiButton @click="onClick" variant="outline-primary">
        <i :class="recordButtonIcon"></i> <slot>Record</slot>
    </UiButton>
</template>

<script>

export default {
    name: 'UiRecordButton',
    props: {
        backend: Object,
        camera: Object,
        scene: Object,
        width: { type: Number, default: 1920 },
        height: { type: Number, default: 1080 },
        antialias: {type: Boolean, default: true },
    },
    data() {
        return {
            recording: false,
            t: 0,
            i: 0,
            sampleCount: 0,
            wasPaused: false,
        }
    },
    computed: {
        recordButtonIcon() {
            return this.recording ? 'bi bi-record2-fill blinkIcon' : 'bi bi-record2-fill'
        },
    },
    watch: {
        recording() {
            if (this.recording) {
                this.enableRecord()
            } else {
                this.disableRecord()
            }
        }
    },
    methods: {
        onClick() {
            this.recording = !this.recording
        },
        enableRecord() {
            console.log('start recording')
            this.i = this.backend.playback.currentIndex
            this.sampleCount = this.backend.playback.sampleCount
            this.wasPaused = this.backend.playback.paused
            this.backend.playback.pause()
            setTimeout(this.tick.bind(this), 0)
        },
        disableRecord(){
            if (!this.wasPaused) {
                this.backend.playback.play()
            }
        },
        tick() {
            this.emitSample()
            this.createFrame()
            this.i += 1
            if (this.i >= this.sampleCount) {
                this.recording = false
                console.log('recording done')
            }
            if (this.recording) {
                setTimeout(this.tick.bind(this), 0)
            }
        },
        emitSample() {
            this.backend.playback.currentIndex = this.i
            this.backend.playback.sendCurrentSample()
        },
        createFrame() {
            const ind = this.i
            const size = {width: this.width, height: this.height}
            BABYLON.Tools.CreateScreenshotUsingRenderTarget(this.scene.engine, this.camera, size, data => {
                console.log('send frame', ind)
                this.backend.sendCommand(['frame', ind, data])
            }, undefined, undefined, this.antialias)
        },
    },
}
</script>

<style scoped>
.blinkIcon {
    animation: blinker 1s linear infinite;
    color: red;
}
@keyframes blinker {
    50% {
        opacity: 0;
    }
}
</style>
