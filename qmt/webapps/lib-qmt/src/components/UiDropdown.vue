<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <select class="form-select" @input="$emit('update:modelValue', $event.target.value)">
        <option v-for="option in options" :value="option" :selected="option === modelValue">{{ option }}</option>
    </select>
</template>

<script>
export default {
    name: 'UiDropdown',
    props: {
        modelValue: [Number, String],
        options: {type: Array, default: []},
    },
}
</script>

<style scoped>

</style>
