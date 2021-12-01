# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import argparse
import asyncio
import logging
import os
import sys

import qmt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


async def watchFile(webapp, path, target, lastmtime):
    logger.info(f'watching {target} file "{path}" for changes')
    while True:
        mtime = os.path.getmtime(path)
        if mtime > lastmtime:
            logger.info(f'{target} file "{path}" changed, reloading!')
            setattr(webapp, target, path)
            lastmtime = mtime
        await asyncio.sleep(0.5)


def run(path, data, config, watch, datasource, chromium, headless, noLib):
    show = 'window'
    if chromium:
        show = 'chromium'
    if headless:
        show = 'none'
    webapp = qmt.Webapp(path, show=show)
    if not os.path.isfile(webapp.path):
        logger.critical(f'path not found ({webapp.path})')
        exit(1)
    if noLib:
        webapp.noLib = True
    if watch:
        if data is not None:
            asyncio.get_event_loop().create_task(watchFile(webapp, data, 'data', os.path.getmtime(data)))
        if config is not None:
            asyncio.get_event_loop().create_task(watchFile(webapp, config, 'config', os.path.getmtime(config)))
    if datasource:
        sys.path.insert(0, '.')
        source = qmt.dataSourceFromJson(datasource)
        webapp.setupOnlineLoop(source)
    webapp.data = data
    webapp.config = config
    webapp.run()


def main():
    parser = argparse.ArgumentParser(description='qmt webapp command line tool')
    parser.add_argument('-d', '--data', help='path to data file (mat/json)')
    parser.add_argument('-c', '--config', help='path to config file (mat/json)')
    parser.add_argument('-w', '--watch', help='auto-reload data/config when file changes', action='store_true')
    parser.add_argument('-s', '--datasource', help='json string describing a data source object to generate')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--chromium', help='run chromium in app mode instead of internal Qt-based window',
                       action='store_true')
    group.add_argument('--headless', help='do not open window', action='store_true')
    parser.add_argument('--no-lib', help='do not serve /lib-qmt and other builtin static files', action='store_true')
    parser.add_argument('path', nargs='?', default='.', help='directory containing index.html, path to html file or '
                                                             'special url (e.g. /view/imubox), default: %(default)s')
    args = parser.parse_args()
    run(args.path, args.data, args.config, args.watch, args.datasource, args.chromium, args.headless, args.no_lib)
