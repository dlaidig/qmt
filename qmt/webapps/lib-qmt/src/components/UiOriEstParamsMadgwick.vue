<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <UiSlider v-model="betaSliderVal" :="betaSliderParams" :valueLabel="formatBeta(beta)">
        <template #label>β</template>
    </UiSlider>
</template>

<script>
import { round } from '../utils.js'
import { _ } from '../vendor.js'

export default {
    name: 'UiOriEstParamsMadgwick',
    props: {
        modelValue: Object,
    },
    data() {
        return {
            betaSliderVal: Math.log10(0.1),
            betaSliderParams: {
                min: -3,
                max: 0,
                step: 0.05,
                ticks: [-3, -2, -1, 0],
                labelWidth: '2em',
                valueLabelWidth: '4em',
                options: {
                    ticks_labels: [0.001, 0.01, 0.1, 1],
                    formatter: val => Math.pow(10, val).toFixed(3),
                },
            },
        }
    },
    computed: {
        value() {
            return {
                beta: this.beta,
            }
        },
        beta() {
            return round(Math.pow(10, this.betaSliderVal), 4)
        },
    },
    watch: {
        value(newValue) {
            this.$emit('update:modelValue', newValue)
        },
        modelValue: {
            handler(newValue, oldValue) {
                if ('beta' in newValue && newValue.beta !== this.beta) {
                    this.betaSliderVal = Math.log10(newValue.beta)
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
        formatBeta(beta) {
            return beta.toFixed(3)
        },
    },
}
</script>

<style scoped>

</style>
