<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <i
        class="bi bi-info-circle" :title="tooltip" ref="icon"
        data-bs-toggle="tooltip" data-bs-html="true" :data-bs-placement="placement"
    ></i>
</template>

<script>
import { bootstrap } from '../vendor.js'

export default {
    name: 'UiInfoTooltipIcon',
    props: {
        tooltip: { type: String, default: '' },
        placement: { type: String, default: 'top' }, // data-bs-placement="top"
    },
    mounted() {
        new bootstrap.Tooltip(this.$refs.icon)
    },
}
</script>

<style scoped>

</style>
