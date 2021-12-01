<!--
SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <UiPanel v-if="signals" title="Signal Selection">
        <div v-for="(item, index) in chains" >
            <div :class="index > 0 ? 'mt-3' : ''"><strong>{{ index === 0 ? 'Main Box Model' : 'Secondary Box Model' }}</strong></div>
            <div><UiDropdown v-model="selectedSignal[index]" :options="signalList" /></div>

            <UiCheckbox style="float: right" class="mt-4" toggle inline v-model="advancedSettings[index]">Advanced Settings</UiCheckbox>

            <div class="btn-group mt-3" role="group">
                <input type="radio" class="btn-check" :name="'signalSelectionRadio'+id+'_'+index" :id="'signalSelectionRadio'+id+'visible'+index" autocomplete="off" checked v-if="index === 0" value="visible" v-model="mode[index]">
                <label class="btn btn-outline-primary" :for="'signalSelectionRadio'+id+'visible'+index" v-if="index === 0">Visible</label>

                <input type="radio" class="btn-check" :name="'signalSelectionRadio'+id+'_'+index" :id="'signalSelectionRadio'+id+'overlay'+index" autocomplete="off"  v-if="index > 0" value="overlay" v-model="mode[index]">
                <label class="btn btn-outline-primary" :for="'signalSelectionRadio'+id+'overlay'+index" v-if="index > 0">Overlay</label>

                <input type="radio" class="btn-check" :name="'signalSelectionRadio'+id+'_'+index" :id="'signalSelectionRadio'+id+'sbs'+index" autocomplete="off"  v-if="index > 0" value="sbs" v-model="mode[index]">
                <label class="btn btn-outline-primary" :for="'signalSelectionRadio'+id+'sbs'+index" v-if="index > 0">Side-by-side</label>

                <input type="radio" class="btn-check" :name="'signalSelectionRadio'+id+'_'+index" :id="'signalSelectionRadio'+id+'hidden'+index" autocomplete="off" :checked="index > 0" value="hidden" v-model="mode[index]">
                <label class="btn btn-outline-primary" :for="'signalSelectionRadio'+id+'hidden'+index">Hidden</label>
            </div>

            <div v-show="advancedSettings[index]" class="mt-3">
            <UiCheckbox v-model="applyToSelected" class="mt-3">only change selected segment: {{ selectedSegment }}</UiCheckbox>
            <strong><span> Opacity: </span></strong> <UiCheckbox style="float: right" v-model="edge[index]">Show Edges</UiCheckbox>
            <UiSlider v-model="opacity[index]"  label="" labelWidth="2em" :valueLabel="v => v + '%'" valueLabelWidth="3em"
                          :options="{min: 0, max: 100, step: 1, ticks: [0, 50, 100], ticks_labels: [0, 50, 100], ticks_snap_bounds: 4, precision: 1}" />
            <strong><span> Position: </span></strong>
            <UiVectorSliders
                :min="-50"
                :max="+50"
                :step="1"
                :ticks="[-50, 0, 50]"
                :ticks_labels="[-50, 0, 50]"
                :ticks_snap_bounds="3"
                v-model="position[index]"/>
            </div>
        </div>
    </UiPanel>
</template>

<script>

import  { _ } from '../vendor.js'
import UiVectorSliders from './UiVectorSliders.vue'
import UiCheckbox from "./UiCheckbox.vue";

export default {
    name: 'UiFullBodySignalSelection',
    components: {
        UiCheckbox,
        UiVectorSliders,
    },

    props: {
        backend: Object,
        scene: Object,
        config: Object,
        chains: Array,
    },
    data() {
        return {
            mode: [],
            oldMode: [],
            selectedSignal: [],
            selectedSegment:  'all',
            applyToSelected: false,
            advancedSettings: [],
            opacity:[],
            edge:[],
            position:[],
            defaultPosition: [],
            id: _.uniqueId()
        }
    },

    mounted() {
        this.scene.on('pick', name => {
            if (this.selectedSegment === name)
                this.selectedSegment = 'all'
            else
                this.selectedSegment = name
        })
        this.initialize()
    },

    methods: {
        initialize() {
            for (let i = 0; i < this.chains.length; i++){
                this.advancedSettings[i] = false
                this.position[i] = this.chains[i].position
                this.defaultPosition[i] = this.chains[i].position
                if (i === 0) {
                    this.edge[i] = true
                    this.opacity[i] = 100
                    this.mode[i] = 'visible'
                    this.oldMode[i] = 'visible'
                } else {
                    this.edge[i] = false
                    this.opacity[i] = 0
                    this.mode[i] = 'hidden'
                    this.oldMode[i] = 'hidden'
                }

                const ind = 2 - 2*i
                this.selectedSignal[i] = this.signalList[Math.min(ind, this.signalList.length-1)]
            }
        },

        updateSignal(chain, signal) {
            const selectedPath = this.signals[signal]
            // chain.setSignal(this.selectedSegment, selectedPath)
            // console.log(`Signal of segment: ${this.selectedSegment}:  is changed to: ${signal}`)
            chain.setSignal('all', selectedPath)
            if (this.backend.lastSample)
                this.backend.emit('sample', this.backend.lastSample)
        },
        updateOpacity(chain, opacity) {
            chain.setOpacity(this.applyToSelected ? this.selectedSegment : 'all', opacity/100)
        },
        updateEdge(chain, eg) {
            chain.setEdges(this.applyToSelected ? this.selectedSegment : 'all', eg)
        },
        updatePosition(chain, position) {
            chain.position = position
        }
    },
    watch: {
        selectedSignal: {
            handler(selectedSignals) {
                for (let i = 0; i < selectedSignals.length; i++){
                    if (selectedSignals[i] !==undefined) {
                        this.updateSignal(this.chains[i], selectedSignals[i])
                    }
                }
            },
            deep: true
        },
        opacity: {
            handler(ops) {
                console.log('OPS')
                for (let i = 0; i < ops.length; i++){
                    if (ops[i] !==undefined)
                        this.updateOpacity(this.chains[i], ops[i])
                }
            },
            deep: true
        },
        edge: {
            handler(eg) {
                for (let i = 0; i < eg.length; i++) {
                    if (eg[i] !==undefined)
                        this.updateEdge(this.chains[i], eg[i])
                }
            },
            deep: true
        },
        position: {
            handler(pos) {
                for (let i = 0; i < pos.length; i++){
                    if (pos[i] !==undefined)
                        this.updatePosition(this.chains[i], pos[i])
                }
            },
            deep: true
        },
        mode: {
            handler(mode) {
                this.applyToSelected = false
                for (let i = 0; i < mode.length; i++) {
                    if (mode[i] === this.oldMode[i])
                        continue
                    this.oldMode[i] = mode
                    if (mode[i] === 'visible') {
                        this.position[i] = this.defaultPosition[i]
                        this.opacity[i] = 100
                        this.edge[i] = true
                    } else if (mode[i] === 'hidden') {
                        this.position[i] = this.defaultPosition[i]
                        this.opacity[i] = 0
                        this.edge[i] = false
                    } else if (mode[i] === 'overlay') {
                        this.position[i] = this.defaultPosition[0]
                        this.opacity[i] = 50
                        this.edge[i] = false
                    } else if (mode[i] === 'sbs') {
                        this.position[i] = this.defaultPosition[i]
                        this.opacity[i] = 50
                        this.edge[i] = true
                    } else {
                        console.warn(`unknown mode ${mode[i]}`)
                    }
                }
            },
            deep: true
        },
    },
    computed: {
        signalList() {
            return Object.keys(this.config['signals'])
        },
        signals() {
            return this.config['signals']
        }
    },
}
</script>

<style scoped>

</style>

