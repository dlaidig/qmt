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
### Added
- Support for zooming in and out via the context menu of the webapp viewer.
- Support for passing arbitrary options to the IMU boxes of a BoxModel via the `imubox_options` property.
## Changed
- Updated JavaScript dependencies of webapp library.
### Fixed
- Fixed array broadcasting in `qmt.headingInclinationAngle`.

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

[Unreleased]: https://github.com/dlaidig/qmt/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/dlaidig/qmt/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/dlaidig/qmt/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dlaidig/qmt/releases/tag/v0.1.0
