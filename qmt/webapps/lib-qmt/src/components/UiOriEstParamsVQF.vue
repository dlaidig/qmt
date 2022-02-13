<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <UiSlider v-model="tauAccSliderVal" :="tauSliderParams" :valueLabel="formatTau(tauAcc)">
        <template #label><span class="ms-4"></span>τ<sub>acc</sub></template>
    </UiSlider>
    <UiSlider v-model="tauMagSliderVal" :="tauSliderParams" :uncheckedValue="-1000.0" :valueLabel="formatTau(tauMag)">
        <template #label>τ<sub>mag</sub></template>
    </UiSlider>
    <UiCheckbox v-model="restBiasEstEnabled">Enable rest gyro bias estimation</UiCheckbox>
    <UiCheckbox v-model="motionBiasEstEnabled">Enable motion gyro bias estimation</UiCheckbox>
    <UiCheckbox v-model="magDistRejectionEnabled">Enable magnetic disturbance rejection</UiCheckbox>
</template>

<script>
import { round } from '../utils.js'
import { _ } from '../vendor.js'

export default {
    name: 'UiOriEstParamsVQF',
    props: {
        modelValue: Object,
    },
    data() {
            return {
                tauAccSliderVal: Math.log10(3),
                tauMagSliderVal: Math.log10(9),
                restBiasEstEnabled: true,
                motionBiasEstEnabled: true,
                magDistRejectionEnabled: true,
                tauSliderParams: {
                    min: -2,
                    max: 2,
                    step: 0.1,
                    ticks: [-2, -1, 0, 1, 2],
                    labelWidth: '4.7em',
                    valueLabelWidth: '2.5em',
                    options: {
                        ticks_labels: [0.01, 0.1, 1, 10, 100],
                        formatter: val => round(Math.pow(10, val), 2),
                    },
                },
            }
        },
        computed: {
            value() {
                return {
                    tauAcc: this.tauAcc,
                    tauMag: this.tauMag,
                    restBiasEstEnabled: this.restBiasEstEnabled,
                    motionBiasEstEnabled: this.motionBiasEstEnabled,
                    magDistRejectionEnabled: this.magDistRejectionEnabled,
                }
            },
            tauAcc() {
                return this.tauAccSliderVal === -1000.0 ? -1 : round(Math.pow(10, this.tauAccSliderVal), 2)
            },
            tauMag() {
                return this.tauMagSliderVal === -1000.0 ? -1 : round(Math.pow(10, this.tauMagSliderVal), 2)
            },
        },
        watch: {
            value(newValue) {
                this.$emit('update:modelValue', newValue)
            },
            modelValue: {
                handler(newValue, oldValue) {
                    if ('tauAcc' in newValue && newValue.tauAcc !== this.tauAcc) {
                        this.tauAccSliderVal = newValue.tauAcc === -1.0 ? -1000.0 : Math.log10(newValue.tauAcc)
                    }
                    if ('tauMag' in newValue && newValue.tauMag !== this.tauMag) {
                        this.tauMagSliderVal = newValue.tauMag === -1.0 ? -1000.0 : Math.log10(newValue.tauMag)
                    }
                    if ('restBiasEstEnabled' in newValue && newValue.restBiasEstEnabled !== this.restBiasEstEnabled) {
                        this.restBiasEstEnabled = newValue.restBiasEstEnabled
                    }
                    if ('motionBiasEstEnabled' in newValue && newValue.motionBiasEstEnabled !== this.motionBiasEstEnabled) {
                        this.motionBiasEstEnabled = newValue.motionBiasEstEnabled
                    }
                    if ('magDistRejectionEnabled' in newValue && newValue.magDistRejectionEnabled !== this.magDistRejectionEnabled) {
                        this.magDistRejectionEnabled = newValue.magDistRejectionEnabled
                    }
                    if (!oldValue || !_.isEqual(Object.keys(oldValue), Object.keys(this.value))) {
                        this.$emit('update:modelValue', this.value) // trigger initial update
                    }
                },
                deep: true,
                immediate: true,
            },
        },
        methods: {
            formatTau(tau) {
                return tau === -1.0 ? '-1' : (tau >= 100 ? tau.toFixed(0) : (tau >= 10 ? tau.toFixed(1) : tau.toFixed(2)))
            }
        },
}
</script>

<style scoped>

</style>
