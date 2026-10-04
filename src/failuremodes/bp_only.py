"""BP-only decoder: min-sum BP hard decisions without OSD post-processing."""

import numpy as np

from qudec.bposd import BpOsdDecoder


class BpOnlyDecoder(BpOsdDecoder):
    """BpOsdDecoder with the OSD stage removed.

    decode_corrections returns the sign hard decisions of the final BP
    LLRs, so non-convergence stays observable: a syndrome-violating
    output means BP stopped unsatisfied. Relies on qudec's private _bp
    pass, which is pinned to the sibling repository.
    """

    def decode_corrections(self, sx, sz):
        n = self.h_z.shape[1]
        llr_x = np.full(n, np.log((1 - self.p_x) / self.p_x))
        l_x = self._bp(self.h_z, sx, llr_x).numpy()
        llr_z = np.full(n, np.log((1 - self.p_z) / self.p_z))
        l_z = self._bp(self.h_x, sz, llr_z).numpy()
        return (l_x < 0).astype(np.int8), (l_z < 0).astype(np.int8)
