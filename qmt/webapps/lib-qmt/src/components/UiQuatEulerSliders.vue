<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<!-- "overflow: hidden" prevents horizontal scrollbar due to slider tick labels while not cutting off the slider tooltips -->

<template>
    <div style="overflow: hidden">
        <UiDropdown :options="modes" v-model="modeState" class="mb-3" />
        <div v-if="modeState === 'Quaternion'">
            <UiSlider
                v-model="w" ref="slider_w"
                label="w" labelWidth="2em"
                :valueLabel="value.w.toFixed(2)" valueLabelWidth="3em"
                :options="quatSliderParams"
            />
            <UiSlider
                v-model="x" ref="slider_x"
                label="x" labelWidth="2em"
                :valueLabel="value.x.toFixed(2)" valueLabelWidth="3em"
                :options="quatSliderParams"
            />
            <UiSlider
                v-model="y"  ref="slider_y"
                label="y" labelWidth="2em"
                :valueLabel="value.y.toFixed(2)" valueLabelWidth="3em"
                :options="quatSliderParams"
            />
            <UiSlider
                v-model="z"  ref="slider_z"
                label="z" labelWidth="2em"
                :valueLabel="value.z.toFixed(2)" valueLabelWidth="3em"
                :options="quatSliderParams"
            />
        </div>
        <div v-else>
            <UiSlider
                v-model="alpha" ref="slider_alpha"
                label="α" labelWidth="2em"
                :valueLabel="v => Math.round(v)+'°'" valueLabelWidth="3em"
                :options="angleSliderParams"
            />
            <UiSlider
                v-model="beta" ref="slider_beta"
                label="β" labelWidth="2em"
                :valueLabel="v => Math.round(v)+'°'" valueLabelWidth="3em"
                :options="angleSliderParams"
            />
            <UiSlider
                v-model="gamma" ref="slider_gamma"
                label="γ" labelWidth="2em"
                :valueLabel="v => Math.round(v)+'°'" valueLabelWidth="3em"
                :options="angleSliderParams"
            />
        </div>
    </div>
</template>

<script>
import UiDropdown from './UiDropdown.vue'
import { Quaternion, rad2deg, deg2rad } from "../utils";

export default {
    name: 'UiQuatEulerSliders',
    components: { UiDropdown },
    props: {
        modelValue: { type: Quaternion, default: Quaternion.identity() },
        mode: { type: String, default: '' },
    },
    data() {
        const modes = ['Quaternion']
        for (let inex of ['intrinsic', 'extrinsic']) {
            for (let order of ['z-y-x', 'z-x-y', 'y-x-z', 'y-z-x', 'x-z-y', 'x-y-z', 'z-x-z', 'z-y-z', 'y-x-y', 'y-z-y', 'x-y-x', 'x-z-x']) {
                modes.push('Euler ' + order + ' ' + inex)
            }
        }

        return {
            value: this.modelValue,
            w: 1.0, x: 0.0, y: 0.0, z: 0.0,
            alpha: 0.0, beta: 0.0, gamma: 0.0,
            modes,
            modeState: this.mode || modes[0],
            angleSliderParams: {
                min: -180,
                max: 180,
                step: 1,
                ticks: [-180, -90,  0, 90, 180],
                ticks_labels: [-180, -90,  0, 90, 180],
                ticks_snap_bounds: 4,
                precision: 1,
            },
            quatSliderParams: {
                min: -1,
                max: 1,
                step: 0.01,
                ticks: [-1, 0, 1],
                ticks_labels: [-1, 0, 1],
                ticks_snap_bounds: 0.04,
                precision: 3,
            }
        }
    },
    watch: {
        w() { this.update() },
        x() { this.update() },
        y() { this.update() },
        z() { this.update() },
        alpha() { this.update() },
        beta() { this.update() },
        gamma() { this.update() },
        modelValue(newVal, oldVal) {
            if (Math.abs(newVal.conj().multiply(oldVal).angle()) < 1e-8) {
                return
            }
            if (Math.abs(newVal.conj().multiply(this.getQuat()).angle()) < 1e-8) {
                return
            }
            this.setQuat(new Quaternion(this.modelValue.array))
        },
        mode(newMode) {
            this.modeState = newMode
        },
        modeState(newMode, oldMode) {
            const quat = this.getQuat(oldMode)
            this.setQuat(quat)
        },
    },
    created() {
        this.setQuat(new Quaternion(this.modelValue.array))
    },
    methods: {
        getQuat(mode) {
            if ((mode ?? this.modeState) === 'Quaternion') {
                if (this.w === 0.0 && this.x === 0.0 && this.y === 0.0 && this.z === 0.0) {
                    return Quaternion.identity()
                }
                return new Quaternion(this.w, this.x, this.y, this.z)
            } else { // Euler angles
                const modeStr = (mode ?? this.modeState).split(' ')
                console.assert(modeStr[0] === 'Euler', modeStr)
                console.assert(modeStr[1].length === 5, modeStr)
                console.assert(modeStr[2] === 'intrinsic' || modeStr[2] === 'extrinsic', modeStr)

                const angles = [deg2rad(this.alpha), deg2rad(this.beta), deg2rad(this.gamma)]
                return Quaternion.fromEulerAngles(angles, modeStr[1][0] + modeStr[1][2] + modeStr[1][4], modeStr[2] === 'intrinsic')
            }
        },
        setQuat(quat) {
            if (this.modeState === 'Quaternion') {
                this.w = quat.w
                this.x = quat.x
                this.y = quat.y
                this.z = quat.z
            } else {
                const modeStr = this.modeState.split(' ')
                const angles = quat.eulerAngles(modeStr[1][0] + modeStr[1][2] + modeStr[1][4], modeStr[2] === 'intrinsic')
                this.alpha = rad2deg(angles[0])
                this.beta = rad2deg(angles[1])
                this.gamma = rad2deg(angles[2])
                this.value = this.getQuat()
            }
        },
        update() {
            const newVal = this.getQuat()
            if (Math.abs(newVal.conj().multiply(this.value).angle()) < 1e-8) {
                return
            }
            this.value = this.getQuat()
            this.$emit('update:modelValue', this.value)
        },
        relayout() {
            if (this.$refs.slider_w) {
                this.$refs.slider_w.relayout()
                this.$refs.slider_x.relayout()
                this.$refs.slider_y.relayout()
                this.$refs.slider_z.relayout()
            } else {
                this.$refs.slider_alpha.relayout()
                this.$refs.slider_beta.relayout()
                this.$refs.slider_gamma.relayout()
            }
        }
    }
}
</script>

<style scoped>

</style>
