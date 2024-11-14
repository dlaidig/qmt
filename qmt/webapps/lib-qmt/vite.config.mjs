// SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
//
// SPDX-License-Identifier: MIT

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import esbuild from 'esbuild'
import path from 'node:path'

// https://github.com/vitejs/vite/issues/6555#issuecomment-2179483448
const minifyBundle = () => ({
	name: 'minify-bundle',
	async generateBundle(_, bundle)
	{
		for (const asset of Object.values(bundle))
		{
			if (asset.type == 'chunk')
				asset.code = (await esbuild.transform(asset.code, {minify: true})).code
		}
	}
})

export default defineConfig({
    plugins: [vue(), minifyBundle()],
    build: {
        sourcemap: true,
        lib: {
            entry: path.resolve(__dirname, 'src/index.js'),
            formats: ['es'],
            name: 'lib-qmt',
        },
        // rollupOptions: {
            // make sure to externalize deps that shouldn't be bundled
            // into your library
            // external: ['vue'],
            // output: {
            //     // Provide global variables to use in the UMD build
            //     // for externalized deps
            //     globals: {
            //         vue: 'Vue'
            //     },
            // },
        // },
    },
    resolve: {
        alias: {
            'jquery': path.resolve(__dirname, 'src/jquery-stub.cjs'),
            'vue': 'vue/dist/vue.esm-bundler.js',
        },
    },
    define: {
        'process.env': {}
    }
    // note that minify will not work with vite 2.6, cf. https://github.com/vitejs/vite/issues/5167 and
    // https://github.com/vitejs/vite/issues/6555
    // this works: https://github.com/vuejs/petite-vue/pull/112
//     esbuild: {
//         minify: true,
//     },
})
