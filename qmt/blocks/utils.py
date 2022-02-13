# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT
import numpy as np

from qmt import Block


class LowpassFilterBlock(Block):
    """
    Second-order Butterworth low-pass filter block.

    This block uses the filter implementation of the VQF orientation estimation algorithm. The cutoff frequency can
    either directly be specified in Hz, or a time constant can be given. This time constant corresponds to the cutoff
    frequency as follows: :math:`f_\\mathrm{c} = \\frac{\\sqrt{2}}{2\\pi\\tau}`.

    For the first :math:`\\tau` seconds, the filter output is the mean of all previous samples, to ensure fast initial
    convergence.
    """
    def __init__(self, Ts, fc=None, tau=None):
        """
        :param Ts: sampling time in seconds
        :param fc: cutoff frequency in Hz
        :param tau: time constant in seconds (either ``fc`` or ``tau`` has to be None)
        """
        super().__init__()
        if tau is None:
            assert fc is not None
            tau = np.sqrt(2)/(2*np.pi*fc)
        else:
            assert fc is None  # either tau or fc has to be set
        self.Ts = Ts
        self.tau = tau
        from vqf import VQF
        self.b, self.a = VQF.filterCoeffs(tau, Ts)
        self.N = None
        self.state = None

    def step(self, signal):
        from vqf import VQF
        signal = np.atleast_1d(np.asarray(signal, float))
        if self.N is None:
            assert signal.ndim == 1
            self.N = signal.shape[0]
            self.state = np.full((max(2, self.N) * 2), np.nan)

        if self.N == 1:
            extended = np.concatenate([signal, np.zeros(1)])  # filterVec needs at least 2 elements
            return VQF.filterVec(extended, self.tau, self.Ts, self.b, self.a, self.state)[0]
        return VQF.filterVec(signal, self.tau, self.Ts, self.b, self.a, self.state)
