<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <UiDropdown v-model="selectedSegment" :options="['(select segment)', ...segmentList]"></UiDropdown>
    <p class="mt-3">
        Parent:
        <a v-if="selectedParent && selectedParent !== 'root'" href="#" @click.prevent="selectedSegment=selectedParent">{{ selectedParent }}</a>
        <span v-else>&ndash;</span>
        <br>
        Children:
        <span v-for="(name, index) in selectedChildren">
                <a href="#" @click.prevent="selectedSegment=name">{{ name }}</a>
                <span v-if="index != selectedChildren.length - 1">, </span>
            </span>
        <span v-if="!selectedChildren.length">&ndash;</span>
    </p>
<!--  Note: This could be extended into a full editor for the box model...  -->
<!--    <UiPanel title="Model" collapsed class="mt-3">-->
<!--        <div>parent, size, color, position, anchor, ...</div>-->
<!--    </UiPanel>-->
    <UiPanel title="Orientation" collapsed class="mt-3">
        <div class="form-check form-check-inline">
            <label class="form-check-label"><input class="form-check-input" type="radio" name="orimode" value="segment" checked>Segment</label>
        </div>
        <div class="form-check form-check-inline">
            <label class="form-check-label"><input class="form-check-input" type="radio" name="orimode" value="sensor">Sensor</label>
        </div>
        <UiQuatEulerSliders v-model="quatValue" mode="Euler z-y-x intrinsic" class="mt-3" />
        <UiCheckbox v-model="relativeToParent">Relative to parent</UiCheckbox>
    </UiPanel>
    <UiPanel title="Manual Tuning" collapsed class="mt-3">
        <p>
            <strong>Heading offset</strong>
            <UiButton variant="outline-primary" class="btn-sm float-end" title="Reset to original value" @click="resetHeading"><i class="bi bi-arrow-counterclockwise"></i></UiButton>
        </p>
        <UiSlider v-model="headingOffset"  label="δ" labelWidth="2em" :valueLabel="v => v + '°'" valueLabelWidth="3em"
                  :options="{min: -180, max: 180, step: 1, ticks: [-180, -90,  0, 90, 180], ticks_labels: [-180, -90,  0, 90, 180], ticks_snap_bounds: 4, precision: 1}" />
        <UiCheckbox v-model="adjustChildrenHeading">Adjust heading of children</UiCheckbox>
        <p class="mt-3">
            <strong>Segment-to-sensor orientation</strong>
            <UiButton variant="outline-primary" class="btn-sm float-end" title="Reset to original value" @click="resetSegment2Sensor"><i class="bi bi-arrow-counterclockwise"></i></UiButton>
        </p>
        <UiQuatEulerSliders v-model="qSegment2Sensor" mode="Euler z-y-x intrinsic" />
        <p class="mt-3"><strong>IMU coordinate system</strong></p>
        <UiDropdown v-model="imuCsValue" :options="imuCsOptions" />
    </UiPanel>
    <UiCheckbox v-model="imuVisibility">Show IMU boxes</UiCheckbox>
    <div>
        <UiButton variant="outline-primary" @click="copyConfig(false)" title="Copy full config to clipboard"><i class="bi bi-clipboard"></i> Full config</UiButton>
        <UiButton variant="outline-primary" class="ms-2" @click="copyConfig(true)" title="Copy configuration changes to clipboard"><i class="bi bi-clipboard"></i> Config changes</UiButton>
    </div>
</template>

<script>

import { Quaternion } from '../utils.js'
import { IMU_CS_OPTIONS } from '../babylon.js'
import { TEMPLATES } from '../boxmodel.js'
import { _ } from '../vendor.js'

export default {
    name: 'UiKinematicChainDebugControls',
    props: {
        scene: Object,
    },

    data() {
        return {
            segmentList: [],
            imuCsOptions: IMU_CS_OPTIONS,
            selectedSegment: '(select segment)',
            selectedParent: null,
            selectedChildren: [],

            imuCsValue: 'FLU',
            quatValue: Quaternion.identity(),
            qSegment2Sensor: Quaternion.identity(),
            headingOffset: 0,
            adjustChildrenHeading: false,
            imuVisibility: true,
            relativeToParent: false,
        }
    },
    watch: {
        scene() {
            this.scene.on('config', config => {
                console.log('box model config', config)
                this.segmentList = _.keys(config)
            })
            this.scene.on('pick', name => {
                if (this.selectedSegment === name)
                    this.selectedSegment = '(select segment)'
                else
                    this.selectedSegment = name
            })
            this.scene.on('sample', sample => this.onSample(sample))
        },
        selectedSegment(name) {
            this.scene.boxmodel.setSelectedSegments([name])
            const segment = this.getSelectedSegment()
            if (segment) {
                this.selectedParent = segment.options.parent
                this.selectedChildren = this.scene.boxmodel.segmentNames.filter(n => this.scene.boxmodel.segments[n].options.parent === name)

                this.imuCsValue = segment.imubox_cs
                this.headingOffset = segment.heading_offset
                this.qSegment2Sensor = segment.options.q_segment2sensor ?? Quaternion.identity()
                this.onSample()
            } else {
                this.selectedParent = null
                this.selectedChildren = []
            }
        },
        imuCsValue(cs) {
            const segment = this.getSelectedSegment()
            if (!segment)
                return
            if (segment.imubox_cs === cs)
                return
            segment.imubox_cs = cs
        },
        quatValue(quat) {
            const segment = this.getSelectedSegment()
            if (!segment)
                return
            if (this.relativeToParent) {
                const parent = this.getSelectedParent()
                if (parent) {
                    quat = parent.actualQuat.multiply(quat)
                }
            }
            if (Math.abs(quat.conj().multiply(segment.quat).angle()) < 1e-8)
                return
            segment.quat = quat
            // also store in segment.changes so that manual changes are included in the copied config
            segment.changes.quat = quat
            this.scene.boxmodel.refresh()
        },
        qSegment2Sensor(quat) {
            const segment = this.getSelectedSegment()
            if (!segment)
                return
            if (Math.abs(quat.conj().multiply(segment.q_segment2sensor).angle()) < 1e-8)
                return
            segment.q_segment2sensor = quat
            this.scene.boxmodel.refresh()
        },
        headingOffset(val) {
            const segment = this.getSelectedSegment()
            if (!segment)
                return
            const old = segment.heading_offset
            segment.heading_offset = val
            if (this.adjustChildrenHeading) {
                this.recursivelyAdjustHeading(segment.name, val - old)
            }

            this.scene.boxmodel.refresh()
        },
        imuVisibility(val) {
            for (const segmentName of this.segmentList) {
                if (this.scene.boxmodel.segmentNames.includes(segmentName)) {
                    this.scene.boxmodel.segments[segmentName].imuVisibility = val
                }
            }
        },
        relativeToParent() {
            this.onSample()
        }
    },
    mounted() {

    },
    methods: {
        getSelectedSegment() {
            if (this.scene.boxmodel.segmentNames.includes(this.selectedSegment)) {
                return this.scene.boxmodel.segments[this.selectedSegment]
            }
            return null
        },
        getSelectedParent() {
            if (this.scene.boxmodel.segmentNames.includes(this.selectedParent)) {
                return this.scene.boxmodel.segments[this.selectedParent]
            }
        },
        onSample() {
            const segment = this.getSelectedSegment()
            const parent = this.getSelectedParent()
            if (segment) {
                if (this.relativeToParent && parent) {
                    this.quatValue = parent.quat.conj().multiply(segment.quat)
                } else {
                    this.quatValue = segment.quat
                }
            }
        },
        recursivelyAdjustHeading(parent, diff) {
            const segments = this.scene.boxmodel.segmentNames.filter(n => this.scene.boxmodel.segments[n].options.parent === parent)
            for (let name of segments) {
                this.scene.boxmodel.segments[name].heading_offset = this.scene.boxmodel.segments[name].heading_offset + diff
                this.recursivelyAdjustHeading(name, diff)
            }
        },
        resetSegment2Sensor() {
            const segment = this.getSelectedSegment()
            if (!segment)
                return
            this.qSegment2Sensor = segment.origSeg2Sensor
        },
        resetHeading() {
            const segment = this.getSelectedSegment()
            if (!segment)
                return
            this.headingOffset = segment.options.heading_offset ?? 0
        },
        copyConfig(changesOnly) {
            const config = {}
            config.segments = {}
            for (let name of this.scene.boxmodel.segmentNames) {
                const segment = this.scene.boxmodel.segments[name]
                if (changesOnly) {
                    if (!_.isEmpty(segment.changes)) {
                        config.segments[name] = segment.changes
                    }
                } else {
                    const options = {...segment.options, ...segment.changes}
                    for (let key in options) {
                        const val = options[key]
                        const defaultVal = TEMPLATES.common.defaults[key]
                        if (val === defaultVal
                            || (key === 'quat' && Array.isArray(val) && Array.isArray(defaultVal) && _.isEqual(val, defaultVal))
                            || (val instanceof Quaternion && Array.isArray(defaultVal) && Math.abs(val.conj().multiply(new Quaternion(defaultVal)).angle()) < 1e-8)) {
                            delete options[key]
                        }
                    }
                    config.segments[name] = options
                }
            }
            const json = JSON.stringify(config, (key, val) => {
                if (val instanceof Quaternion) {
                    return val.array
                } if (val instanceof BABYLON.Color4) {
                    return val.toHexString(true)
                } else {
                    return val
                }
            }, '  ')
            navigator.clipboard.writeText(json).then(function() {
                console.log('wrote config to clipboard:', json)
            }, function() {
                console.error('error writing config to clipboard', json)
            })
        },
    }
}
</script>

<style scoped>

</style>
