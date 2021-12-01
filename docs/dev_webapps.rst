.. SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
..
.. SPDX-License-Identifier: MIT

.. _dev_webapps:

Webapp development
##################

The aim of this page is to give some overview and some hints that will make it easier to understand the architecture
used for the ``qmt`` webapps and to develop custom webapps.

Data flow
=========

The purpose of most webapps is to visualize data, either in real-time or from data files. In some cases, data processing
needs to be interactive, and commands (e.g., button clicks) or parameters (e.g., from a check box or slider) need to be
sent to the Python processing code. In the webapps, functionality for managing this data flow is provided by the
``Backend`` javascript class.

Real-time data streaming
------------------------

For real-time data streaming, a websocket connection between the Python backend (via the :class:`qmt.Webapp` class) and
the webapp is used.

The Python backend mainly sends *samples* to the webapp in a regular interval. A sample is a dictionary/object that
always has a key ``t`` for the time and otherwise application-defined data, e.g., quaternions. Furthermore, the
Python backend can send *commands* to trigger application-defined action. A command is a list. The first entry is the
command name as a string and optional further entries are application-defined.

The webapp can send *params* (parameters) and *commands* to the Python backend. Parameters are a dictionary/object with
application-defined values that influence the data processing. Params can be sent at irregular intervals when the values
are changed by the user.

Playback of recorded data
-------------------------

For recorded data, the playback of json files is supported. The data files must have a key ``t`` that is a time vector.
The other measurement data must be provided in lists with the same length as the time vector.

The ``Backend`` class automatically handles playback of those data files and generates samples with the data for the
current time. The webapp can show playback controls to let the user pause playback, seek, and so on.

Data is usually loaded from a ``data.json`` file. It is also possible to update the data via a websocket connection with
the special command ``setData``. This is automatically handled by the :class:`qmt.Webapp` class.

Configuration
-------------

Furthermore, webapps can be adjusted with an application-specific configuration file. While data files or samples are
used for time-dependent data, the config is suitable for static information that can change the behavior of the webapp
(signal names in the data file, camera angles, colors, ...).

The configuration is usually loaded from a ``config.json`` file. It is also possible to update the data via a websocket
connection with the special command ``setConfig``. This is automatically handled by the :class:`qmt.Webapp` class.

Used frameworks
===============

The javascript library used by the webapps, ``lib-qmt.js``, bundles many third-party frameworks, mainly:

- Babylon.js (https://www.babylonjs.com/)
- Vue (https://v3.vuejs.org/)
- Bootstrap (https://getbootstrap.com/)
- Bootstrap Icons (https://icons.getbootstrap.com/)
- bootstrap-slider (https://github.com/seiyria/bootstrap-slider)
- Smoothie Charts (https://github.com/joewalnes/smoothie/)
- Chart.js (https://www.chartjs.org/)
- Split.js (https://split.js.org/)
- Emitter (https://github.com/component/emitter/)

Development of custom webapps
=============================

In addition to the bundled webapps, which are served on the special paths ``/demo`` and ``/view``, custom webapps can
be created. In general, it is sufficient to create a single html file (e.g., copy ``/demo/template/index.html``) and
then point the ``qmt-webapp`` CLI tool or the :class:`qmt.Webapp` class to this file.

For a better development experience, also copy and adjust the ``package.json`` and ``vite.config.js`` files. With this,
it is possible to run a development webserver with automatic reloading (also for files inside ``lib-qmt``):

.. code-block:: sh

    npm run dev

Either open the webapp in a regular browser or use the :attr:`qmt.Webapp.devServerUrl` property. Printing JavaScript
debug messages and developer tools is supported with :class:`qmt.Webapp`. See :attr:`qmt.Webapp.jsLogLevel` for more
information.

In a regular browser and with a development server running on a different port, it is still possible to connect to the
:class:`qmt.Webapp` websocket when running with ``show='none'`` and manually providing the websocket location, e.g.:

.. code-block::

    http://localhost:3000/?ws=ws://localhost:8000/ws

All webapps bundled with the `qmt` toolbox are written in a way that does not require building (i.e., only
``lib-qmt.js`` needs to be built once). Unfortunately, this means that Single File Components (SFCs) cannot be used.
Instead, components are created with the object notation and ``Vue.defineComponent``. In combination with
``npm run build``, custom SFCs can be used.
