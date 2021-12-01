<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <div style="display: flex; flex-direction: row; align-items: start; width: 100%;" :class="{unchecked: !checked}">
        <div style="flex-grow: 0;" :style="{ width: labelWidth }" v-if="label || labelWidth">
            <template v-if="uncheckedValue === undefined">
                {{ label }}<slot name="label"></slot>
            </template>
            <template v-else>
                <UiCheckbox v-model="checked">{{ label }}<slot name="label"></slot></UiCheckbox>
            </template>
        </div>
        <input type="text" ref="el">
        <div style="flex-grow: 0;" :style="{ marginLeft: marginEnd, width: valueLabelWidth }" v-if="valueLabel !== false || valueLabelWidth || marginEnd">{{ valueLabelString }}</div>
    </div>
</template>

<script>
import Slider from 'bootstrap-slider'
import UiCheckbox from "./UiCheckbox.vue"

export default {
    name: 'UiSlider',
    components: { UiCheckbox },
    props: {
        modelValue: { type: [Number, Array], default: 0.0 },
        uncheckedValue:{ type: [Number, Array], default: undefined },  // enables check box
        min: Number,
        max: Number,
        step: Number,
        ticks: Array,
        rangeHighlights: Array,
        options: { type: Object, default: () => {} },  // see https://github.com/seiyria/bootstrap-slider
        noUpdateDuringDrag: { type: Boolean, default: false},
        label: { type: String, default: ''},
        labelWidth: { type: String, default: '1em'},
        valueLabel: { type: [Boolean, String, Number, Function]},
        marginEnd: {type: String, default: '1em'},
        valueLabelWidth: { type: String, default: ''},
    },
    data() {
        const unchecked = this.modelValue === this.uncheckedValue && this.uncheckedValue < this.min
        return {
            sliderValue: unchecked ? this.min : this.modelValue,
            slider: null,
            checked: !unchecked,
        }
    },
    computed: {
        value() {
            if (!this.checked) {
                return this.uncheckedValue
            }
            return this.sliderValue
        },
        valueLabelString() {
            if (this.valueLabel === true) {
                return this.value
            } else if (this.valueLabel === false) {
                return ''
            } else if (typeof this.valueLabel === 'function') {
                return this.valueLabel(this.value)
            } else {
                return this.valueLabel
            }
        }
    },
    watch: {
        modelValue(newValue) {
            if (!this.checked) {
                if (newValue === this.uncheckedValue) {
                    return
                }
                this.checked = true
            } else if (this.uncheckedValue !== undefined && newValue === this.uncheckedValue && newValue < this.min) {
                this.checked = false
                return
            }
            this.sliderValue = newValue
            this.slider.setValue(newValue)
        },
        value(newValue, oldValue) {
            if (newValue !== this.modelValue) {
                this.$emit('update:modelValue', newValue)
            }
        },
        ticks(newValue) {
            this.slider.setAttribute('ticks', [...newValue])
        },
        min(newValue) {
            this.slider.setAttribute('min', newValue)
            this.slider.relayout()
        },
        max(newValue) {
            this.slider.setAttribute('max', newValue)
            this.slider.relayout()
        },
        rangeHighlights(newValue, oldValue) {
            console.log('slider highlights changed, recreating', oldValue, newValue)
            // see https://github.com/seiyria/bootstrap-slider/issues/693
            // this.slider.setAttribute('rangeHighlights', [...newValue])
            this.recreate()
        }
    },
    mounted() {
        this.recreate()
    },
    methods: {
        sliderChanged(val) {
            if(this.sliderValue === val)
                return
            this.sliderValue = val
        },
        recreate() {
            if (this.slider !== null) {
                this.slider.destroy()
            }
            let options = {
                value: this.modelValue,
            }
            if (this.min !== undefined)
                options.min = this.min
            if (this.max !== undefined)
                options.max = this.max
            if (this.step !== undefined)
                options.step = this.step
            if (this.ticks !== undefined)
                options.ticks = [...this.ticks]
            if (this.rangeHighlights !== undefined)
                options.rangeHighlights = [...this.rangeHighlights]

            options = {...options, ...this.options}
            // console.log('creating slider', this.$el, options)

            this.slider = new Slider(this.$refs.el, options)
            this.slider.getElement().style.flexGrow = '1'
            // this.slider.getElement().style.width = '100%'
            this.slider.on('slideStop', val => this.sliderChanged(val))
            if (!this.noUpdateDuringDrag)
                this.slider.on('slide', val => this.sliderChanged(val))
            setTimeout(() => this.slider.relayout(), 0)
            setTimeout(() => this.slider.relayout(), 10)  // do this twice to avoid weird effects with multiple sliders
            // make sure tick labels etc are updated on resize
            new ResizeObserver(() => setTimeout(() => this.slider.relayout(), 0)).observe(this.slider.getElement())

            // this.sliderValue = options.value // disabled since it breaks sliders that are initially unchecked

            if (options.rangeHighlights) { // create tooltips if the marker entries contain a label field
                const elements = this.slider.getElement().querySelectorAll('.slider-rangeHighlight')
                console.assert(elements.length == options.rangeHighlights.length)
                for (let i = 0; i < elements.length; i++) {
                    if (!options.rangeHighlights[i].label)
                        continue
                    elements[i].title = options.rangeHighlights[i].label
                }
            }
        },
        relayout() {
            this.slider.relayout()
        }
    },
}
</script>

<style>
    /* range hightlights do seem to sometimes get hidden permanently if the range changes */
    .slider-rangeHighlight.k { background: #000; opacity: 0.3; display: block !important; }
    .slider-rangeHighlight.C0 { background: #1f77b4; opacity: 0.3; display: block !important; }
    .slider-rangeHighlight.C1 { background: #ff7f0e; opacity: 0.3; display: block !important; }
    .slider-rangeHighlight.C2 { background: #2ca02c; opacity: 0.3; display: block !important; }
    .slider-rangeHighlight.C3 { background: #d62728; opacity: 0.3; display: block !important; }
    .slider-rangeHighlight.C4 { background: #9467bd; opacity: 0.3; display: block !important; }
    .slider-rangeHighlight.C5 { background: #8c564b; opacity: 0.3; display: block !important; }
    .slider-rangeHighlight.C6 { background: #e377c2; opacity: 0.3; display: block !important; }
    .slider-rangeHighlight.C7 { background: #7f7f7f; opacity: 0.3; display: block !important; }
    .slider-rangeHighlight.C8 { background: #bcbd22; opacity: 0.3; display: block !important; }
    .slider-rangeHighlight.C9 { background: #17becf; opacity: 0.3; display: block !important; }

    .unchecked .slider-selection, .unchecked .slider-tick.in-selection {
	    background: #afdcff;
    }
    .unchecked .slider-handle {
	    background: #77a0c4;
    }
</style>
