<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <div class="dropdown" :class="direction">
        <button class="btn dropdown-toggle" :class="'btn-'+btnVariant" type="button" data-bs-toggle="dropdown" :data-bs-auto-close="autoClose">
            <slot name="button"><i class="bi bi-list"></i></slot>
        </button>
        <div class="dropdown-menu" >
            <slot name="default"></slot>
        </div>
    </div>
</template>

<script>
export default {
    name: 'UiDropdownMenu',
    props: {
        btnVariant: {type: String, default: 'outline-primary'},
        direction: {type: String, default: ''}, // '', dropup, dropstart, dropend
        autoClose: {type: String, default: 'outside'}, // true, false, outside, inside
    },
}
</script>

<style scoped>

</style>
