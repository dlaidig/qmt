<!--
SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>

SPDX-License-Identifier: MIT

Sections:
### Added (for new features)
### Changed (for changes in existing functionality)
### Deprecated (for soon-to-be removed features)
### Removed (for now removed features)
### Fixed (for any bug fixes)
### Security (in case of vulnerabilities)
-->
# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.4] - 2023-08-09
### Added
- Webapps support loading data from a (data) URL passed via the location hash (i.e., after a `#`).
- Support for async command handlers in data sources and blocks.
### Changed
- Improved Chrome/Chromium auto-detection on macOS.
- Updated vite version.
### Fixed
- Do not ignore options when passing a `qmt.Struct` to `qmt.toJson`.
- Detection of in-use ports when starting a webapp server is more robust.
- Initialize `sendQueue` in webapps started via `arun`.
- Remove assert that started to fail in custom SchemeHandler of PySide-based webapp backend.
- Improved robustness of fetch workaround for QWebEngine.

## [0.2.3] - 2022-11-01
### Added
- Ability to use custom IMU boxes in BoxModel and UiIMUScene.
- Option to rotate camera in UiIMUScene on click.
- `qmt.Struct.update` method.
### Changed
- Use version 2.0.0 of the VQF orientation estimation algorithm.
### Fixed
- Make webapp viewer work with PySide 6.4.

## [0.2.2] - 2022-10-04
### Added
- Support for zooming in and out via the context menu of the webapp viewer.
- Support for passing arbitrary options to the IMU boxes of a BoxModel via the `imubox_options` property.
- Support for setting custom application icons for webapps via the `icon` property of `qmt.Webapp`.
## Changed
- Change minimum Python version from 3.7 to 3.8 (and minimum scipy version to 1.8.0).
- Updated JavaScript dependencies of webapp library.
### Fixed
- Fixed array broadcasting in `qmt.headingInclinationAngle`.
- In `UiPlaybackControls`, show range highlights even when they start/end outside of the valid time range.

## [0.2.1] - 2022-05-09
### Changed
- The webapp viewer now uses PySide6 by default and PySide2 as a fallback (set QT_API=PySide2 to always use PySide2).
### Fixed
- The dip plots in the orientation estimation demo now use the correct units. 

## [0.2.0] - 2022-04-01
### Added
- Support for the [VQF orientation estimation algorithm](https://github.com/dlaidig/vqf) (`qmt.oriEstVQF`,
  `qmt.oriEstBasicVQF`, `qmt.oriEstOfflineVQF`, `qmt.OriEstVQFBlock`).
- Orientation estimation demo (`examples/orientation_estimation_demo.py`).
- Simple magnetometer calibration function function `qmt.calibrateMagnetometerSimple`.
- Lowpass filter block `qmt.LowpassFilterBlock`.
- Utility function `qmt.parallel` for simple multiprocessing-based parallel data processing.
### Changed
- Prefix all orientation estimation functions/blocks with `oriEst`/`OriEst` (`qmt.oriEstMadgwick`, `qmt.oriEstMahony`,
 `qmt.OriEstMadgwickBlock`, `qmt.OriEstMahonyBlock`).
- The `ipython` and `notebook` packages are no longer listed as dependencies.
- Improved heading filter behavior and stillness detection in `qmt.headingCorrection`.
### Fixed
- Output correct gyroscope bias estimate in `qmt.oriEstMahony` (the previous estimate was wrongly scaled by Ki and
  the sign was wrong) and in `qmt.oriEstIMU` (the sign was wrong).
- The `filename` parameter of `qmt.setupDebugPlots` now accepts `pathlib.Path` objects in addition to strings.
- Setting values for the `constraint` parameter of `qmt.headingCorrection` is now possible.
- Fixed angle wrapping issue in the `1d_corr` constraint of the `qmt.headingCorrection` function.

## [0.1.0] - 2021-12-01
### Added
- Initial release.

[Unreleased]: https://github.com/dlaidig/qmt/compare/v0.2.4...HEAD
[0.2.3]: https://github.com/dlaidig/qmt/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/dlaidig/qmt/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/dlaidig/qmt/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/dlaidig/qmt/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/dlaidig/qmt/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dlaidig/qmt/releases/tag/v0.1.0
