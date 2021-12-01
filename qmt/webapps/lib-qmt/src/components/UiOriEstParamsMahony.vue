<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <UiSlider v-model="kpSliderVal"  :="{...sliderParams, ...kpSliderParams}" :valueLabel="formatKp(Kp)">
        <template #label>K<sub>p</sub></template>
    </UiSlider>
    <UiSlider v-model="kiSliderVal"  :="{...sliderParams, ...kiSliderParams}" :valueLabel="formatKi(Ki)">
        <template #label>K<sub>i</sub></template>
    </UiSlider>
</template>

<script>
import { round } from '../utils.js'
import { _ } from '../vendor.js'

export default {
    name: 'UiOriEstParamsMahony',
    props: {
        modelValue: Object,
    },
    data() {
        return {
            kpSliderVal: Math.log10(0.8),
            kiSliderVal: -1000.0,
            sliderParams: {
                uncheckedValue: -1000.0,
                step: 0.05,
                labelWidth: '4em',
                valueLabelWidth: '5em',
            },
            kpSliderParams: {
                min: -2,
                max: 2,
                ticks: [-2, -1, 0, 1, 2],
                options: {
                    ticks_labels: [0.01, 0.1, 1, 10, 100],
                    formatter: val => round(Math.pow(10, val), 3),
                },
            },
            kiSliderParams: {
                uncheckedValue: -1000.0,
                min: -6,
                max: -3,
                ticks: [-6, -5, -4, -3],
                options: {
                    ticks_labels: [0.00001, 0.00001, 0.0001, 0.001],
                    formatter: val => round(Math.pow(10, val), 6),
                },
            },
        }
    },
    computed: {
        value() {
            return {
                Kp: this.Kp,
                Ki: this.Ki,
            }
        },
        Kp() {
            return this.kpSliderVal === -1000.0 ? 0.0 : round(Math.pow(10, this.kpSliderVal), 3)
        },
        Ki() {
            return this.kiSliderVal === -1000.0 ? 0.0 : round(Math.pow(10, this.kiSliderVal), 8)
        },
    },
    watch: {
        value(newValue) {
            this.$emit('update:modelValue', newValue)
        },
        modelValue: {
            handler(newValue, oldValue) {
                if ('Kp' in newValue && newValue.Kp !== this.Kp) {
                    this.KpSliderVal = newValue.Kp === 0.0 ? -1000.0 : Math.log10(newValue.Kp)
                }
                if ('Ki' in newValue && newValue.Ki !== this.Ki) {
                    this.KiSliderVal = newValue.Ki === 0.0 ? -1000.0 : Math.log10(newValue.Ki)
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
        formatKp(val) {
            return val.toFixed(3)
        },
        formatKi(val) {
            return val.toFixed(6)
        },
    },
}
</script>

<style scoped>

</style>
