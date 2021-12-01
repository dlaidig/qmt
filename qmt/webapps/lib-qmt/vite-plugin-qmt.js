// SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
//
// SPDX-License-Identifier: MIT

const path = require('path')

module.exports = function qmt(options) {
    const defaults = {
        aliases: true, // define module aliases
        babylon: true, // if false, replace babylonjs with stub to reduce build size
        server: true, // define server config with proxy for assets
    }
    options = {...defaults, ...options}
    return {
        name: 'qmt',
        config() {
            const config = {
                build: {
                    chunkSizeWarningLimit: 10000,
                }
            }
            if (options.aliases) {
                config.resolve = {
                    alias: {
                        'vue': 'vue/dist/vue.esm-bundler.js',
                        '/lib-qmt/lib-qmt.js': 'lib-qmt/src/index.js',
                        '/lib-qmt/style.css': 'lib-qmt/src/style.css',
                        'jquery': path.resolve(__dirname, 'src/jquery-stub.cjs'),
                    },
                }
                if (!options.babylon) {
                    config.resolve.alias['@babylonjs/core/Legacy/legacy'] = path.resolve(__dirname, 'src/babylonjs-stub.js')
                }
            }
            if (options.server) {
                config.server = {
                    proxy: {
                        '^/lib-qmt/.*': {
                            target: 'http://localhost:3000/@fs' + path.resolve(__dirname, '..'),
                            changeOrigin: true,
                        }
                    },
                    fs: {
                        strict: true,
                        allow: ['.', path.resolve(__dirname, '../lib-qmt')],
                    }
                }
            }
            return config
        }
    }
}

