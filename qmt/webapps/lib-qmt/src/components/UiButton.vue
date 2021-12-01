<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <button
        type="button"
        class="btn" :class="'btn-'+variant"
    >
        <slot></slot>
    </button>
</template>

<script>
export default {
    name: 'UiButton',
    props: {
        variant: { type: String, default: 'primary' },
    }
}
</script>

<style scoped>

</style>
