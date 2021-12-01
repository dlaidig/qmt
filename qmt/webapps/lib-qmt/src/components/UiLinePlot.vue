<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT

set style to control the plot size, e.g.:
:style="{width: '100%', height: '200px'}"
-->

<template>
    <canvas ref="canvas" :style="style"></canvas>
    <p class="text-center">
        <template v-for="(label, index) in labels">
            <span v-if="label">
                <i class="bi bi-circle-fill" :style="{color: colorCodes[index]}"></i> {{ label }} {{ ' ' }}
                <slot name="legendExtra" :index="index" :signal="signals[index]"></slot>
            </span>
        </template>
    </p>
</template>

<script>
import { BABYLON, smoothie, _ } from '../vendor.js'
import { COLORS } from '../utils.js'

export default {
    name: 'UiLinePlot',
    props: {
        style: { type: Object, default: {width: '100%', height: '200px'} },
        source: { type: Object, default: {} },
        signals: { type: Array, default: [] },
        labels: { type: Array, default: [] },
        colors: { type: Array, default: [] }, // name from COLORS, hex string like '#1f77b4' or BABYLON.Color4
        ylim: { type: Array, default: [undefined, undefined]},
        scaleFactor: { type: Number, default: 1.0 },
        verticalSections: { type: Number, default: 6 },
        clip: { type: Boolean, default: true },
        useTimeSignal: { type: Boolean, default: true }, // if true, use sample.t as time signal in seconds
    },
    data() {
        return {
            startTimeSys: new Date().getTime(),
            dataTimeShift: 0.0,
            chart: null,
            series: [],
            missingSignals: new Set(),
            listener: null,
        }
    },
    computed: {
        colorCodes() {
            const defaultColors = [COLORS.C0, COLORS.C1, COLORS.C2, COLORS.C3, COLORS.C4, COLORS.C5, COLORS.C6,
                COLORS.C7, COLORS.C8, COLORS.C9]
            const codes = []
            for (let i = 0; i < this.signals.length; i++) {
                const color = this.colors[i] || defaultColors[i % defaultColors.length]
                if (color in COLORS) {
                    codes.push(COLORS[color].toHexString(true))
                } else if (color instanceof BABYLON.Color4) {
                    codes.push(color.toHexString(true))
                } else {
                    codes.push(color)
                }
            }
            return codes
        }
    },
    mounted() {
        // http://smoothiecharts.org/builder/
        this.chart = new smoothie.SmoothieChart({
            responsive: true,
            interpolation: 'linear',
            maxValue: this.ylim[1],
            minValue: this.ylim[0],
            grid: {
                fillStyle: '#ffffff',
                sharpLines: true,
                verticalSections: this.verticalSections,
            },
            labels: {
                fillStyle: '#000000',
                precision: 1,
            },
            timestampFormatter: t => {
                if (this.useTimeSignal)
                    return (Math.round(t.getTime()/1000.0 - this.dataTimeShift).toString() + ' ')
                return (Math.round((t.getTime() - this.startTimeSys) / 1000).toString() + ' ')
            }
        })

        const colorCodes = this.colorCodes
        for (let i = 0; i < this.signals.length; i++) {
            const s = new smoothie.TimeSeries()
            this.chart.addTimeSeries(s, {lineWidth: 3, strokeStyle: colorCodes[i]})
            this.series.push(s)
        }

        this.chart.streamTo(this.$refs.canvas, 0)
        this.listener = sample => this.onSample(sample)
        this.source.on('sample', this.listener)
    },
    beforeUnmount() {
        this.source.off('sample', this.listener)
    },
    methods: {
        onSample(sample) {
            let t = new Date().getTime()
            if (this.useTimeSignal) {
                const shift = t/1000.0 - sample.t
                if (Math.abs(shift - this.dataTimeShift) > 1.0) {
                    this.dataTimeShift = shift // large gap or init
                } else {
                    this.dataTimeShift = this.dataTimeShift + 0.001 * (shift - this.dataTimeShift) // filter
                }

                t = 1000*(sample.t + this.dataTimeShift)
            }

            for (let i = 0; i < this.signals.length; i++) {
                const name = this.signals[i]
                let value = _.get(sample, name)
                if (value === undefined) {
                    if (!this.missingSignals.has(name)) {
                        console.warn('signal used in UiLinePlot does not exist in sample:', name)
                        this.missingSignals.add(name)
                    }
                }
                value = this.scaleFactor * value
                if (this.clip) {
                    value = this.clipVal(value)
                }
                this.series[i].append(t, value)
            }
        },
        clipVal(value) {
            if (this.ylim[0] === undefined || this.ylim[1] == undefined)
                return value
            const eps = Math.abs(this.ylim[1] - this.ylim[0]) / 200
            return Math.min(Math.max(value, this.ylim[0] + eps), this.ylim[1] - eps)
        },
    }
}
</script>

<style scoped>

</style>
