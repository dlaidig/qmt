<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <div class="card mb-3">
        <div class="card-header" :style="collapsible ? 'cursor: pointer;' : ''" @click="click()">
            {{ title }}
            <slot name="header"></slot>
            <UiInfoTooltipIcon v-if="infoTooltip" class="float-end" :tooltip="infoTooltip" placement="bottom" />
        </div>
        <div class="collapse" ref="collapse">
            <div class="card-body">
                <slot></slot>
            </div>
        </div>
    </div>
</template>

<script>
import * as bootstrap from "bootstrap"
import UiInfoTooltipIcon from './UiInfoTooltipIcon.vue'

export default {
    name: 'UiPanel',
    components: { UiInfoTooltipIcon },
    props: {
        title: { type: String, required: true },
        collapsed: { type: Boolean, default: false }, // can be used with v-model
        collapsible: { type: Boolean, default: true },
        infoTooltip: { type: String, default: '' },
    },
    emits: [
        'update:collapsed',
        'firstShow',  // emitted on first uncollapse when panel is initially collapsed
    ],
    data() {
        return {
            collapsedState: this.collapsed,
            bsCollapse: {},
            waitForFirstShow: this.collapsed,
        }
    },
    methods: {
        click() {
            if (this.collapsible)
                this.toggle()
        },
        toggle() {
            if (this.$refs.collapse.classList.contains('show')) {
                this.bsCollapse.hide()
                this.$emit('update:collapsed', true)
            } else {
                this.bsCollapse.show()
                this.$emit('update:collapsed', false)
                if (this.waitForFirstShow) {
                    this.$emit('firstShow')
                    this.waitForFirstShow = false
                }
            }
        },
    },
    mounted() {
        if (!this.collapsed)
            this.$refs.collapse.classList.add('show')
        this.bsCollapse = new bootstrap.Collapse(this.$refs.collapse, {toggle: false})
    },
    watch: {
        collapsed(newValue) { // called if collapsed was changed from parent, e.g. with :collapse or v-model
            if (newValue && this.$refs.collapse.classList.contains('show')) {
                this.bsCollapse.hide()
            } else if (!newValue && !this.$refs.collapse.classList.contains('show')) {
                this.bsCollapse.show()
            }
        }
    },
}
</script>

<style scoped>

</style>
