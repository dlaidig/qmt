// SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
//
// SPDX-License-Identifier: MIT

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import qmt from 'lib-qmt/vite-plugin-qmt'

export default defineConfig({
    plugins: [
        qmt(),
        vue()
    ],
    base: '',
})
