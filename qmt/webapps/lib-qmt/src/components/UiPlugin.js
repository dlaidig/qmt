// SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
//
// SPDX-License-Identifier: MIT

import {
    UiButton,
    UiCheckbox,
    UiDropdown,
    UiDropdownMenu,
    UiFullBodyCameraTools,
    UiFullBodySignalSelection,
    UiHeadingCorrectionPlot,
    UiIMUScene,
    UiInfoTooltipIcon,
    UiKinematicChainDebugControls,
    UiLinePlot,
    UiOriEstParamsMadgwick,
    UiOriEstParamsMahony,
    UiOriEstParamsOriEstIMU,
    UiPanel,
    UiPlaybackControls,
    UiQuatEulerSliders,
    UiRecordButton,
    UiRenderCanvas,
    UiSlider,
    UiSplitter,
    UiVectorSliders,
    UiWebcamVideo
} from '../components'

export const UiPlugin = {
    install(vue) {
        const components = [
            UiButton,
            UiCheckbox,
            UiDropdown,
            UiDropdownMenu,
            UiFullBodyCameraTools,
            UiFullBodySignalSelection,
            UiHeadingCorrectionPlot,
            UiIMUScene,
            UiInfoTooltipIcon,
            UiKinematicChainDebugControls,
            UiLinePlot,
            UiOriEstParamsMadgwick,
            UiOriEstParamsMahony,
            UiOriEstParamsOriEstIMU,
            UiPanel,
            UiPlaybackControls,
            UiQuatEulerSliders,
            UiRecordButton,
            UiRenderCanvas,
            UiSlider,
            UiSplitter,
            UiVectorSliders,
            UiWebcamVideo,
        ]
        for (var component of components) {
            vue.component(component.name, component)
        }
    }
}
