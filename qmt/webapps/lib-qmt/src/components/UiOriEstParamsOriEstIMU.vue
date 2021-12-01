<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <UiSlider v-model="tauAccSliderVal" :="tauSliderParams" :valueLabel="formatTau(tauAcc)">
        <template #label>τ<sub>acc</sub></template>
    </UiSlider>
    <UiSlider v-model="tauMagSliderVal" :="tauSliderParams" :valueLabel="formatTau(tauMag)">
        <template #label>τ<sub>mag</sub></template>
    </UiSlider>
    <UiSlider v-model="zeta" :="zetaSliderParams" :valueLabel="zeta">
        <template #label>ζ<sub>bias</sub></template>
    </UiSlider>
    <UiSlider v-model="accRating" :="ratingSliderParams" :valueLabel="accRating">
        <template #label>r<sub>acc</sub></template>
    </UiSlider>
</template>

<script>
import { round } from '../utils.js'
import { _ } from '../vendor.js'

export default {
    name: 'UiOriEstParamsOriEstIMU',
    props: {
        modelValue: Object,
    },
    data() {
        return {
            tauAccSliderVal: Math.log10(1),
            tauMagSliderVal: Math.log10(3),
            zeta: 0.0,
            accRating: 1.0,
            tauSliderParams: {
                uncheckedValue: -1000.0,
                min: -2,
                max: 2,
                step: 0.1,
                ticks: [-2, -1, 0, 1, 2],
                labelWidth: '4.7em',
                valueLabelWidth: '2em',
                options: {
                    ticks_labels: [0.01, 0.1, 1, 10, 100],
                    formatter: val => round(Math.pow(10, val), 2),
                },
            },
            zetaSliderParams: {
                uncheckedValue: 0.0,
                min: 0,
                max: 10,
                step: 0.1,
                ticks: [0, 1, 5, 10],
                labelWidth: '4.7em',
                valueLabelWidth: '2em',
                options: {
                    ticks_labels: [0, 1, 5, 10],
                    ticks_positions: [0, 10, 50, 100],
                },
            },
            ratingSliderParams: {
                uncheckedValue: 0.0,
                min: 0,
                max: 5,
                step: 0.1,
                ticks: [0, 1, 5],
                labelWidth: '4.7em',
                valueLabelWidth: '2em',
                options: {
                    ticks_labels: [0, 1, 5],
                    ticks_positions: [0, 20, 100],
                },
            },
        }
    },
    computed: {
        value() {
            return {
                tauAcc: this.tauAcc,
                tauMag: this.tauMag,
                zeta: this.zeta,
                accRating: this.accRating,
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
                if ('zeta' in newValue && newValue.zeta !== this.zeta) {
                    this.zeta = newValue.zeta
                }
                if ('accRating' in newValue && newValue.accRating !== this.accRating) {
                    this.accRating = newValue.accRating
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
        },
    },
}
</script>

<style scoped>

</style>
