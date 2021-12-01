<!--
SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <div style="overflow: hidden">
        <UiSlider
            v-model="x" ref="slider_x"
            label="x" labelWidth="2em"
            :valueLabel="x.toFixed(2)" valueLabelWidth="3em"
            :options="vectorSliderParams"
        />
        <UiSlider
            v-model="y" ref="slider_y"
            label="y" labelWidth="2em"
            :valueLabel="y.toFixed(2)" valueLabelWidth="3em"
            :options="vectorSliderParams"
        />
        <UiSlider
            v-model="z" ref="slider_z"
            label="z" labelWidth="2em"
            :valueLabel="z.toFixed(2)" valueLabelWidth="3em"
            :options="vectorSliderParams"
        />
    </div>
</template>

<script>
import { vecNorm } from '../utils.js'

export default {
    name: 'UiVectorSliders',
    props: {
        modelValue: {type: Array, default: [0, 0, 0]},
        min:{type: Number, default: -1},
        max:{type: Number, default: 1},
        step:{type: Number, default: 0.01},
        ticks:{type: Array, default: [-1, 0, 1]},
        ticks_labels:{type: Array, default: [-1, 0, 1]},
        ticks_snap_bounds:{type: Number, default: 0.04},
        precision:{type: Number, default: 3}
    },
    data() {
        return {
            value: this.modelValue,
            x: 0.0, y: 0.0, z: 0.0,
            vectorSliderParams: {
                min: this.min,
                max: this.max,
                step: this.step,
                ticks: this.ticks,
                ticks_labels: this.ticks_labels,
                ticks_snap_bounds: this.ticks_snap_bounds,
                precision: this.precision,
            }
        }
    },
    methods: {
        update() {
            const newVal = this.getVal()
            if (Math.abs(vecNorm(newVal)-vecNorm(this.value)) < 1e-8) {
                return
            }
            this.value = this.getVal()
            this.$emit('update:modelValue', this.value)
        },
        setVal(val) {
            this.x = val[0]
            this.y = val[1]
            this.z = val[2]
        },
        getVal() {
            return [this.x, this.y, this.z]
        },
        relayout() {
            this.$refs.slider_x.relayout()
            this.$refs.slider_y.relayout()
            this.$refs.slider_z.relayout()
        },
    },
    watch: {
        x() { this.update() },
        y() { this.update() },
        z() { this.update() },
        modelValue() {
            // if (Math.abs(vecNorm(newVal)-vecNorm(oldVal)) < 1e-8){
            //     return
            // }
            // if (Math.abs(vecNorm(newVal)- vecNorm(this.getVal())) < 1e-8) {
            //     return
            // }
            this.setVal(this.modelValue)
        },
    },
    created() {
        this.setVal(this.modelValue)
    },
}
</script>

<style scoped>

</style>
