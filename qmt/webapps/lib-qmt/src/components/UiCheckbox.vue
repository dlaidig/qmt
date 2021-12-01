<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <div class="form-check" :class="{'form-switch': toggle, 'form-check-inline': inline}">
        <label class="form-check-label">
            <input class="form-check-input" type="checkbox" :checked="modelValue" @change="$emit('update:modelValue', $event.target.checked)" >
            <slot></slot>
        </label>
    </div>
</template>

<script>
export default {
    name: 'UiCheckbox',
    props: {
        modelValue: { type: Boolean, default: false },
        toggle: { type: Boolean, default: false },
        inline: { type: Boolean, default: false },
    }
}
</script>

<style scoped>

</style>
