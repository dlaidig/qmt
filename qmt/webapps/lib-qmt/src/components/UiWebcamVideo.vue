<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT
-->

<template>
    <video autoplay ref="video" class="w-100 h-100"></video>
</template>

<script>
export default {
    name: 'UiWebcamVideo',
    mounted() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => this.$refs.video.srcObject = stream)
                .catch(error => console.warn('error accessing webcam', error))
        } else {
            console.warn('navigator.mediaDevices.getUserMedia not available, cannot show webcam video ' +
                '(webcam access is not yet possible in the PySide2-based viewer, use chromium instead.)')
        }
    },
    beforeUnmount() {
        const stream = this.$refs.video.srcObject
        if (stream) {
            for (let track of stream.getTracks()) {
                track.stop()
            }
        }
        this.$refs.video.srcObject = null
    },
}
</script>

<style scoped>

</style>
