<!--
SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <UiPanel title="Heading Correction Plot"  class="mt-3">
        <canvas
            ref="el"
            style="width: 100%; height: 300px;"
        />
    </UiPanel>
</template>
<script>

import { deg2rad, rad2deg, wrapToPi } from "../utils.js"
import { _, Chart } from '../vendor.js'

export default {
    name: 'UiHeadingCorrectionPlot',
    props: {
        delta_min: {type: Number, default: -180},
        delta_max: {type: Number, default: 180},
        rating_min: {type: Number, default: 0},
        rating_max: {type: Number, default: 1},
        rating_step: {type: Number, default: 0.1},
        delta_step: {type: Number, default: 30},
        seconds: {type: Number, default: 60},
        delta_line_color: {type: String, default: 'rgba(255, 0, 50, .8)'},
        delta_filt_line_color: {type: String, default: 'rgba(0, 255, 50, .8)'},
        rating_line_color: {type: String, default: 'rgba(255, 174, 99, .8)'},
        rating_fill_color: {type: String, default: 'rgba(255,208,99, .2)'},
        source: Object,
        config: Object,
        segment: String,
    },
    data() {
        return {
        }
    },

    mounted() {
        this.chart = this.createChart()
        this.source.on('sample', this.onSample.bind(this))
    },
    methods: {
        createChart() {
            return new Chart(this.$refs.el, {
                type: 'line',
                data: {
                    labels: new Array(60).fill(0),
                    datasets: [
                        {
                            label: 'delta',
                            yAxisID: 'A',
                            data: new Array(60).fill(0),
                            backgroundColor: [
                                'rgba(255,255,255, .0)',
                            ],
                            borderColor: [
                                this.delta_line_color,
                            ],
                            borderWidth: 2,
                            pointRadius: 1,
                            pointHoverRadius: 1
                        },
                        {
                            label: 'delta_filt',
                            yAxisID: 'A',
                            data: new Array(60).fill(0),
                            backgroundColor: [
                                'rgba(255,255,255, .0)',
                            ],
                            borderColor: [
                                this.delta_filt_line_color,
                            ],
                            borderWidth: 2,
                            pointRadius: 1,
                            pointHoverRadius: 1
                        },
                        {
                            label: 'rating',
                            hidden: false,
                            yAxisID: 'B',
                            data: new Array(60).fill(0),
                            backgroundColor: [
                                this.rating_fill_color,
                            ],
                            borderColor: [
                                this.rating_line_color,
                            ],
                            borderWidth: 2,
                            pointRadius: 1,
                            pointHoverRadius: 1
                        }
                    ]
                },
                options: {
                    tooltips: {
                        enabled: true
                    },
                    responsive: true,
                    animation: {duration: 0},
                    scales: {
                        A: {
                            title: {
                                display: true,
                                text: 'delta [°]'
                            },
                            position: 'left',
                            ticks: {
                                stepSize: this.delta_step
                            },
                            max: this.delta_max,
                            min: this.delta_min,
                        },
                        B:{
                            title: {
                                display: true,
                                text: 'rating [-]'
                            },
                            grid: {
                                drawOnChartArea: false
                            },
                            position: 'right',
                            ticks: {
                                stepSize: this.rating_step
                            },
                            max: this.rating_max,
                            min: this.rating_min,
                        },

                        x: {
                            ticks: {
                                stepSize: 1,
                                display:false,
                            },
                            max: this.seconds,
                            min: 0,
                        }
                    }
                }})
        },

        pushValue(delta,delta_filt,rating){
            this.chart.data.labels.pop()
            this.chart.data.labels.push('0')
            this.chart.data.datasets[0].data.splice(0,1)
            this.chart.data.datasets[0].data.push(delta)
            this.chart.data.datasets[1].data.splice(0,1)
            this.chart.data.datasets[1].data.push(delta_filt)
            this.chart.data.datasets[2].data.splice(0,1)
            this.chart.data.datasets[2].data.push(rating)
            /*
                  this.chart.data.datasets.forEach((dataset,index) => {
                      dataset.data.splice(0, 1);
                      dataset.data.push(value);
                  }); */
            this.chart.update()
        },

        clearAll(){
            this.chart.data.datasets.forEach((dataset) => {
                dataset.data = new Array(60).fill(0)
            })
            this.chart.update()
        },

        onSample(sample) {
            const plotData = _.get(sample, this.segment, {})
            const delta = rad2deg(wrapToPi(_.get(plotData, 'delta', NaN)))
            const delta_filt = rad2deg(wrapToPi(_.get(plotData, 'delta_filt', NaN)))
            const rating = _.get(plotData, 'rating', NaN)
            this.pushValue(delta, delta_filt, rating)
        }
    },
}
</script>

<style scoped>

</style>
