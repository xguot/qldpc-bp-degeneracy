"""GPU decoders: qudec's min-sum BP on CUDA plus the batched GPU OSD.

The BP pass in qudec is device-agnostic torch; these subclasses feed it
CUDA tensors and replace the per-shot numpy OSD with batched_osd, so
both stages run on the GPU. Batches are chunked to bound VRAM (the BP
working tensor is (batch, m, n) float32). On a machine without CUDA the
same code runs on CPU, which the equality tests exploit.
"""

import numpy as np
import torch

from qudec.bposd import BpOsdDecoder

from failuremodes.gpu_osd import batched_osd


class _GpuMixin:
    """Shared device plumbing for the GPU decoder subclasses."""

    def __init__(self, *args, device="cuda", gpu_chunk=10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.gpu_chunk = gpu_chunk

    def _bp_chunked(self, h, s, llr):
        """Run _bp in chunks on the device; return (batch, n) LLRs."""
        h_dev = torch.as_tensor(h).to(self.device)
        llr_dev = torch.as_tensor(llr).to(self.device)
        outs = []
        for i in range(0, s.shape[0], self.gpu_chunk):
            s_chunk = torch.as_tensor(
                s[i:i + self.gpu_chunk]).to(self.device)
            outs.append(self._bp(h_dev, s_chunk, llr_dev))
        return torch.cat(outs, dim=0)

    def _osd_batch(self, h, s, orders):
        """Run the batched OSD on the device; return (batch, n) int8."""
        if h.shape[0] == 0:
            return np.zeros((s.shape[0], h.shape[1]), dtype=np.int8)
        return batched_osd(h, s, orders.cpu().numpy(),
                           self.osd_order, self.device)


class GpuBpOsdDecoder(_GpuMixin, BpOsdDecoder):
    """BP + OSD where both stages run on the GPU."""

    def decode_corrections(self, sx, sz):
        n = self.h_z.shape[1]
        llr_x = np.full(n, np.log((1 - self.p_x) / self.p_x))
        l_x = self._bp_chunked(self.h_z, sx, llr_x)
        llr_z = np.full(n, np.log((1 - self.p_z) / self.p_z))
        l_z = self._bp_chunked(self.h_x, sz, llr_z)
        orders_x = torch.argsort(l_x, dim=1, stable=True)
        orders_z = torch.argsort(l_z, dim=1, stable=True)
        c_x = self._osd_batch(self.h_z, sx, orders_x)
        c_z = self._osd_batch(self.h_x, sz, orders_z)
        return c_x, c_z


class GpuBpOnlyDecoder(_GpuMixin, BpOsdDecoder):
    """BP hard decisions on the GPU, no OSD stage."""

    def decode_corrections(self, sx, sz):
        n = self.h_z.shape[1]
        llr_x = np.full(n, np.log((1 - self.p_x) / self.p_x))
        l_x = self._bp_chunked(self.h_z, sx, llr_x).cpu().numpy()
        llr_z = np.full(n, np.log((1 - self.p_z) / self.p_z))
        l_z = self._bp_chunked(self.h_x, sz, llr_z).cpu().numpy()
        return (l_x < 0).astype(np.int8), (l_z < 0).astype(np.int8)
