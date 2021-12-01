<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <div  v-show="!autoHide || mode === 'playback'">
        <div style="display: flex; flex-direction: row; width: 100%; align-items: center;">
            <UiButton :variant="btnVariant" @click="playClicked"><i class="bi" :class="playButtonIcon"></i></UiButton>
            <UiSlider
                ref="slider"
                v-model="t"
                @update:modelValue="sliderChanged"
                :min="min" :max="max" :rangeHighlights="rangeHighlights" :options="{step: 0.01, ticks_snap_bounds: 0.5}"
                :valueLabel="v => v.toFixed(2)" valueLabelWidth="3em"
            />
            <UiDropdownMenu class="ms-2" :direction="direction">
                <div class="ms-2" style="width: 25em; display: grid; grid-template-columns: min-content 1fr; align-items: center; column-gap: 0.5em; row-gap: 0.5em">
                    <i class='bi bi-speedometer fs-4'></i>
                    <div>
                        <UiSlider
                            v-model="speed"
                            labelWidth="4em"
                            :valueLabel="v => Math.pow(2, v)+'x'"
                            valueLabelWidth="4em"
                            class="mt-2"
                            :min="-3" :max="3" :step="1" :ticks="[0]"
                            :options="{formatter: v => Math.pow(2, v)+'x', selection: 'none'}"
                        >
                            <template #label>
                                <UiButton variant="outline-primary" @click="playClicked"><i class="bi" :class="playButtonIcon"></i></UiButton>
                            </template>
                        </UiSlider>
                    </div>

                    <i class='bi bi-clock fs-4'></i>
                    <div>
                        <UiButton variant="outline-primary" @click="jump(-3)"><i class="bi bi-skip-backward"></i></UiButton>
                        <span style="display: inline-block; min-width: 3em; text-align: center">3 s</span>
                        <UiButton variant="outline-primary" @click="jump(3)"><i class="bi bi-skip-forward"></i></UiButton>
                    </div>

                    <div></div>
                    <div>
                        <UiButton variant="outline-primary" @click="jump(-10)"><i class="bi bi-skip-backward"></i></UiButton>
                        <span style="display: inline-block; min-width: 3em; text-align: center">10 s</span>
                        <UiButton variant="outline-primary" @click="jump(10)"><i class="bi bi-skip-forward"></i></UiButton>
                    </div>

                    <i class="bi bi-arrow-repeat fs-4"></i>
                    <UiCheckbox toggle inline v-model="loop"> loop</UiCheckbox>

                    <i class="bi bi-arrow-left-right fs-4"></i>
                    <UiCheckbox toggle inline v-model="reverse"> reverse</UiCheckbox>

                    <div v-show="markers" style="display: contents">
                        <i class='bi bi-pin fs-4'></i>
                        <div style="display: grid; grid-template-columns: min-content 1fr; align-items: center; column-gap: 0.25em">
                            <UiButton
                                :variant="prevMarkerText ? 'outline-primary' : 'outline-secondary'"
                                :disabled="!prevMarkerText"
                                @click="jumpToMarker(-1)"
                            >
                                <i class="bi bi-skip-backward-circle"></i>
                            </UiButton>
                            <div>{{ prevMarkerText || '(no previous marker)' }}</div>
                        </div>

                        <div></div>
                        <div style="display: grid; grid-template-columns: min-content 1fr; align-items: center; column-gap: 0.25em">
                            <UiButton
                                :variant="nextMarkerText ? 'outline-primary' : 'outline-secondary'"
                                :disabled="!nextMarkerText"
                                @click="jumpToMarker(0)"
                            >
                                <i class="bi bi-skip-forward-circle"></i>
                            </UiButton>
                            <div>{{ nextMarkerText || '(no next marker)' }}</div>
                        </div>
                    </div>
                </div>
            </UiDropdownMenu>
        </div>
    </div>
</template>

<script>
import UiSlider from './UiSlider.vue'
import UiButton from './UiButton.vue'

export default {
    name: 'UiPlaybackControls',
    components: {UiButton, UiSlider},
    props: {
        backend: Object,
        autoHide: { type: Boolean, default: true },
        markers: { type: Array },
        btnVariant: { type: String, default: 'outline-primary' },
        direction: { type: String, default: '' },  // '', dropup, dropstart, dropend
    },
    data() {
        return {
            mode: '',
            t: 0.0,
            min: 0.0,
            max: 1.0,
            playing: false,
            speed: 0,
            loop: false,
            reverse: false,
            nextMarkerInd: 0,
            prevMarkerText: '',
            nextMarkerText: '',
            lastMarkerJump: null,
        }
    },
    computed: {
        playButtonIcon() {
            return this.playing ? 'bi-pause-fill' : 'bi-play-fill'
        },
        rangeHighlights() {
            if (!this.markers)
                return []
            const highlights = []
            const minLength = (this.max-this.min)/100
            for (let marker of this.markers) {
                const label = this.markerLabel(marker)

                if (typeof marker === 'number')
                    highlights.push({start: marker, end: marker+minLength, class: 'C0', label: label})
                else
                    highlights.push({start: marker.pos, end: marker.end ?? marker.pos+minLength, class: marker.col ?? 'C0', label: label})

            }
            console.log('HIGHLIGHTS', highlights)
            return highlights
        }
    },
    mounted() {
        this.backend.on('modeChanged', mode => {
            this.mode = mode // online, playback
        })
        this.backend.on('data', data => {
            if (data === undefined)
                return
            this.min = data.t[0]
            this.max = data.t[data.t.length-1]
            console.log('set max', this.max)
        })
        this.backend.on('sample', sample => {
            if (this.mode === 'playback') {
                this.t = sample.t
            }
        })
        this.backend.playback.on('play', () => this.playing = true)
        this.backend.playback.on('pause', () => this.playing = false)
        this.backend.playback.on('stop', () => this.playing = false)

        this.backend.playback.pause()
    },
    watch: {
        speed(val) {
            this.backend.playback.speed = Math.pow(2, val)
        },
        loop(val) {
            this.backend.playback.loop = val
        },
        reverse(val) {
            this.backend.playback.reverse = val
        },
        t(t) {
            if (!this.markers)
                return
            let ind = this.nextMarkerInd

            // disable update based on playback time for one second after jumping to a marker
            if (this.lastMarkerJump !== null && Math.abs(t - this.lastMarkerJump) >= Math.pow(2, this.speed)) {
                console.log('reset', this.lastMarkerJump, t)
                this.lastMarkerJump = null

            }
            if (this.lastMarkerJump === null) {
                while (ind < this.markers.length && t >= (this.markers[ind].pos ?? this.markers[ind])) {
                    ind++
                }
                while (ind > 0 && t < (this.markers[ind - 1].pos ?? this.markers[ind - 1])) {
                    ind--
                }
                if (ind !== this.nextMarkerInd)
                    this.nextMarkerInd = ind
            }

            const prevMarkerText = ind > 0 ? this.markerLabel(this.markers[ind-1], t) : ''
            if (this.prevMarkerText !== prevMarkerText)
                this.prevMarkerText = prevMarkerText

            const nextMarkerText = ind < this.markers.length ? this.markerLabel(this.markers[ind], t) : ''
            if (this.nextMarkerText !== nextMarkerText)
                this.nextMarkerText = nextMarkerText
        },
    },
    methods: {
        playClicked() {
            if (this.backend.playback.paused) {
                this.backend.playback.play()
            } else {
                this.backend.playback.pause()
            }
        },
        sliderChanged(value) {
            this.backend.playback.currentTime = value
        },
        jump(diff) {
            this.backend.playback.currentTime += diff * Math.pow(2, this.speed)
        },
        jumpToMarker(offset) {
            const marker = this.markers[this.nextMarkerInd + offset]
            const t = marker.pos ?? marker
            if (offset < 0)
                this.nextMarkerInd = Math.max(0, this.nextMarkerInd - 1)
            else
                this.nextMarkerInd = Math.min(this.nextMarkerInd + 1, this.markers.length)
            this.backend.playback.currentTime = t
            this.lastMarkerJump = t
        },
        markerLabel(marker, t) {
            const pos = marker.pos ?? marker
            const label = `${marker.name ?? 'marker'}, t=${pos}${marker.end ? '..'+marker.end : ''} s`
            if (t === undefined)
                return label
            const diff = Math.round(pos - t)
            return `${label} (${diff >= 0 ? '+' : ''}${diff} s)`
        },
    },
}
</script>

<style scoped>

</style>
