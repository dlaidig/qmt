<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <div :style="mainStyle">
        <div ref="split0" :style="{overflow: scroll[0] ? 'auto' : 'hidden', position: 'relative'}"><slot name="first">first</slot></div>
        <div ref="split1" :style="{overflow: scroll[1] ? 'auto' : 'hidden', position: 'relative'}"><slot name="second">second</slot></div>
        <div v-if="hasThirdSlot" ref="split2" :style="{overflow: scroll[2] ? 'auto' : 'hidden'}"><slot name="third">third</slot></div>
    </div>
</template>

<script>
import Split from 'split.js'

export default {
    name: 'UiSplitter',
    props: {
        direction: {
            type: String,
            default: 'horizontal',
            validator: val => ['horizontal', 'vertical'].indexOf(val) !== -1,
        },
        sizes: {type: Array, default: () => []},
        scroll: {type: Array, default: [true, true, true]},
        fixed: {type: Boolean, default: false},
    },
    data() {
        return {
            split: null,
            sizeState: this.sizes,
        }
    },
    computed: {
        hasThirdSlot() {
            return !!this.$slots['third']
        },
        mainStyle() {
            if (this.direction === 'horizontal')
                return 'display: flex; flex-direction: row; height: 100%;'
            return 'height: 100%'
        }
    },
    watch: {
        fixed() {
            this.recreate()
        },
        direction(newValue, oldValue) {
            if (newValue !== oldValue) {
                this.recreate()
            }
        },
        sizes(newValue, oldValue) {
            if (newValue.every((v, i) => v === oldValue[i]))
                return

            this.sizeState = newValue
            const sizes = this.normalizeSizes(newValue)
            // console.log('sizes changed', newValue, oldValue, sizes)
            if (this.split) {
                this.split.setSizes(sizes)
            } else {
                const items = this.getItems()
                for (let i = 0; i < items.length; i++) {
                    items[i].style[this.direction === 'horizontal' ? 'width' : 'height'] = sizes[i] + '%'
                }
            }
        },
    },
    mounted() {
        this.recreate()
    },
    methods: {
        getItems() {
            const items = [this.$refs.split0, this.$refs.split1]
            if (this.hasThirdSlot) {
                items.push(this.$refs.split2)
            }
            return items
        },
        normalizeSizes(sizes) {
            var scale = sizes.reduce((a, b) => a + b, 0) / 100
            if (scale < 0.01)
                scale = 0
            return sizes.map(e => e/scale)
        },
        recreate() {
            const items = this.getItems()

            if (this.split) {
                this.split.destroy()
                this.split = null
            }
            for (let i = 0; i < items.length; i++) {
                items[i].style['height'] = ''
                items[i].style['width'] = ''
            }

            const options = {direction: this.direction}
            if (this.sizeState.length === items.length) {
                options.sizes = this.normalizeSizes(this.sizeState)
            } else if (this.sizeState.length !== 0) {
                console.log('UiSplitter: sizes length does not match slot count', this.sizeState, this.sizeState.length)
            }

            if (this.fixed) {
                if (this.sizeState.length === 0) {
                    options.sizes = Array(items.length).fill(100/items.length)
                }
                for (let i = 0; i < items.length; i++) {
                    items[i].style[this.direction === 'horizontal' ? 'width' : 'height'] = options.sizes[i] + '%'
                }
                return
            }

            options.onDragEnd = sizes => {
                // console.log(sizes)
                this.sizeState = sizes
                this.$emit('update:sizes', sizes)
            }

            this.split = Split(items, options)
        }
    },
}
</script>

<style>
/* Unfortunately, scoped styles do not work here. */

.gutter {
    background-color: #eee;
    background-repeat: no-repeat;
    background-position: 50%;
}
.gutter.gutter-horizontal {
    background-image: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAeCAYAAADkftS9AAAAIklEQVQoU2M4c+bMfxAGAgYYmwGrIIiDjrELjpo5aiZeMwF+yNnOs5KSvgAAAABJRU5ErkJggg==');
    cursor: col-resize;
}
.gutter.gutter-vertical {
    background-image: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB4AAAAFAQMAAABo7865AAAABlBMVEVHcEzMzMzyAv2sAAAAAXRSTlMAQObYZgAAABBJREFUeF5jOAMEEAIEEFwAn3kMwcB6I2AAAAAASUVORK5CYII=');
    cursor: row-resize;
}
</style>
