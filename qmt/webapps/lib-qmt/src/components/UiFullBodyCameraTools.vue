<!--
SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <UiPanel title="Camera Controls">
        <div style="flex-grow: 1" class="ms-2" >
                <div class="form-check form-check-inline">
                    <input class="form-check-input" type="radio" name="inlineRadioOptions" id="Free" value="Free"
                           v-model="angle" checked>
                    <label class="form-check-label" for="Free">Free</label>
                </div>
                <div class="form-check form-check-inline">
                    <input class="form-check-input" type="radio" name="inlineRadioOptions" id="Front" value="Front"
                           v-model="angle">
                    <label class="form-check-label" for="Front">Front</label>
                </div>
                <div class="form-check form-check-inline">
                    <input class="form-check-input" type="radio" name="inlineRadioOptions" id="Left" value="Left"
                           v-model="angle">
                    <label class="form-check-label" for="Left">Left</label>
                </div>
                <div class="form-check form-check-inline">
                    <input class="form-check-input" type="radio" name="inlineRadioOptions" id="Right" value="Right"
                           v-model="angle">
                    <label class="form-check-label" for="Right">Right</label>
                </div>
                <div class="form-check form-check-inline">
                    <input class="form-check-input" type="radio" name="inlineRadioOptions" id="Back" value="Back"
                           v-model="angle">
                    <label class="form-check-label" for="Back">Back</label>
                </div>
    </div>
        <br>
        <div class="mb-3" style="float: right">
        <UiCheckbox toggle inline v-model="advancedMode">Advanced Mode</UiCheckbox>
        </div>
        <div v-if="advancedMode">
        <div><strong>Adjust Camera Angle:</strong></div>
        <div>
            <UiSlider  v-model="alphaOffset"  label="α" labelWidth="3em" :valueLabel="v => Math.round(v) + '°'" valueLabelWidth="3em"
                         :options="{min: -180, max: 180, step: 1, ticks: [-180, -90,  0, 90, 180], ticks_labels: [-180, -90,  0, 90, 180], ticks_snap_bounds: 4, precision: 1}" />
            <UiSlider  v-model="betaOffset"  label="β" labelWidth="3em" :valueLabel="v => Math.round(v) + '°'" valueLabelWidth="3em"
                         :options="{min: 0, max: 180, step: 1, ticks: [0, 90, 180], ticks_labels: [0, 90, 180], ticks_snap_bounds: 4, precision: 1}" />
            <UiSlider  v-model="FOVOffset"  label="fov" labelWidth="3em" :valueLabel="v => v" valueLabelWidth="3em"
                         :options="{min: 0.1, max: 1.5, step: 0.1, ticks: [0.1, 1.5],}" />
        </div>
        </div>
    </UiPanel>
</template>
<script>

import {rad2deg, deg2rad } from "../utils.js"

export default {
    name: 'UiFullBodyCameraTools',
    props: {
        camera: Object,
        scene: Object,
        backend: Object,
        initHeading: {type: Number, default: 164},
        initCameraOffsetBeta: {type: Number, default: 0},
    },
    data() {
        return {
            angle: 'Free',
            alphaOffset: 0,
            betaOffset: 0,
            FOVOffset: 0,
            cameraAlpha: 0,
            cameraBeta: 0,
            cameraFOV: 0,
            selectedSegment: null,
            advancedMode: false,
        }
    },
    computed: {
        angleButtonIcon() {
            return 'outline-primary'
        },
    },
    watch:{
        // Camera settings
        alphaOffset() {
            this.onSliderChange()
        },
        betaOffset() {
            this.onSliderChange()
        },
        FOVOffset() {
            this.onSliderChange()
        },
        angle() {
            this.updateCamera()
        },
    },
    mounted() {
        this.scene.on('pick', name => {
            if (this.selectedSegment === name) {
                this.selectedSegment = null;
            } else {
                this.selectedSegment = name;
            }
        })
        this.initCamera()
    },
    methods: {
        getSelectedSegment() {
            if (this.scene.boxmodel.segmentNames.includes(this.selectedSegment)) {
                return this.scene.boxmodel.segments[this.selectedSegment]
            }
            return null
        },

        // Camera settings
        initCamera() {
            this.cameraAlpha = this.camera.alpha
            this.cameraBeta  = this.camera.beta
            this.cameraFOV = this.camera.fov
            this.updateCamera()
            this.alphaOffset = rad2deg(this.cameraAlpha)
            this.betaOffset = rad2deg(this.cameraBeta)
            this.FOVOffset = this.cameraFOV
        },
        updateCamera() {
            if (this.angle === 'Front') { // Front
                this.camera.alpha = this.cameraAlpha + deg2rad(this.initHeading);
                this.scene.camera.beta = this.cameraBeta + deg2rad(this.initCameraOffsetBeta);
            } else if  (this.angle === 'Top') { //top
                this.camera.alpha = this.cameraAlpha + deg2rad(this.initHeading);
                this.camera.beta = this.cameraBeta-Math.PI/2 + deg2rad(this.initCameraOffsetBeta);
            }
            else if  (this.angle === 'Right') { //Right
                this.camera.alpha = this.cameraAlpha + Math.PI/2 + deg2rad(this.initHeading);
                this.camera.beta = this.cameraBeta + deg2rad(this.initCameraOffsetBeta);
            }
            else if  (this.angle === 'Left') { // Left
                this.camera.alpha = this.cameraAlpha - Math.PI/2 + deg2rad(this.initHeading);
                this.camera.beta = this.cameraBeta + deg2rad(this.initCameraOffsetBeta);
            }
            else if (this.angle === 'Back') { // Back
                this.camera.alpha = this.cameraAlpha + Math.PI + deg2rad(this.initHeading);
                this.camera.beta = this.cameraBeta + deg2rad(this.initCameraOffsetBeta);
            } else {
                this.camera.alpha = this.cameraAlpha + deg2rad(this.initHeading);
                this.camera.beta = this.cameraBeta + deg2rad(this.initCameraOffsetBeta);
            }
            this.camera.fov = this.cameraFOV
        },
        onSliderChange() {
            this.cameraAlpha = deg2rad(this.alphaOffset);
            this.cameraBeta = deg2rad(this.betaOffset);
            this.cameraFOV =  this.FOVOffset
            this.updateCamera();
        },
    },
}
</script>

<style scoped>

</style>
