"""Tests for the batched GPU OSD against qudec's CPU implementation."""

import unittest

import numpy as np
import torch

from qudec.bposd import BpOsdDecoder
from qudec.codes import logicals, medium_code, steane_code
from qudec.noise import sample_iid_errors

from failuremodes.bp_only import BpOnlyDecoder
from failuremodes.constructed import hgp_repetition
from failuremodes.gpu_decoders import GpuBpOnlyDecoder, GpuBpOsdDecoder
from failuremodes.gpu_osd import batched_osd

CODES = {
    "steane": steane_code,
    "hgp33": lambda: hgp_repetition(3, 3),
    "bb72": medium_code,
}


def random_orders(n, batch, seed):
    rng = np.random.default_rng(seed)
    return np.argsort(rng.random((batch, n)), axis=1).astype(np.int64)


class TestBatchedOsd(unittest.TestCase):
    def test_matches_qudec_on_random_inputs(self):
        for name, builder in CODES.items():
            h_x, h_z = builder()
            l_x, l_z = logicals(h_x, h_z)
            h = h_z
            m, n = h.shape
            rng = np.random.default_rng(42)
            e = (rng.random((64, n)) < 0.2).astype(np.int8)
            s = (h @ e.T).T % 2
            orders = random_orders(n, 64, 7)
            for osd_order in (0, 1):
                dec = BpOsdDecoder(h_x, h_z, l_x, l_z, 0.1, 0.1,
                                   osd_order=osd_order)
                expected = np.stack(
                    [dec._osd(h, s[i], orders[i]) for i in range(64)])
                got = batched_osd(h, s, orders, osd_order, device="cpu")
                self.assertTrue(np.array_equal(got, expected),
                                f"{name} osd_order={osd_order} mismatch")

    def test_zero_syndrome_zero_correction(self):
        for builder in CODES.values():
            h_x, h_z = builder()
            m, n = h_z.shape
            s = np.zeros((8, m), dtype=np.int8)
            orders = random_orders(n, 8, 3)
            for osd_order in (0, 1):
                got = batched_osd(h_z, s, orders, osd_order, device="cpu")
                self.assertTrue(np.all(got == 0))

    def test_corrections_satisfy_syndrome(self):
        for builder in CODES.values():
            h_x, h_z = builder()
            m, n = h_z.shape
            rng = np.random.default_rng(5)
            e = (rng.random((32, n)) < 0.3).astype(np.int8)
            s = (h_z @ e.T).T % 2
            orders = random_orders(n, 32, 11)
            got = batched_osd(h_z, s, orders, 1, device="cpu")
            self.assertTrue(np.all((h_z @ got.T).T % 2 == s))


class TestGpuDecoders(unittest.TestCase):
    def test_matches_cpu_decoder_exactly(self):
        for name, builder in CODES.items():
            h_x, h_z = builder()
            l_x, l_z = logicals(h_x, h_z)
            n = h_x.shape[1]
            for osd_order in (0, 1):
                cpu = BpOsdDecoder(h_x, h_z, l_x, l_z, 0.1, 0.1,
                                   osd_order=osd_order)
                gpu = GpuBpOsdDecoder(h_x, h_z, l_x, l_z, 0.1, 0.1,
                                      osd_order=osd_order, device="cpu",
                                      gpu_chunk=64)
                e_x, e_z = sample_iid_errors(n, 0.1, 32, "depolarizing",
                                             seed=9)
                sx = (h_z @ e_x.T) % 2
                sz = (h_x @ e_z.T) % 2
                c_cpu, _ = cpu.decode_corrections(
                    sx.T.astype(np.int8), sz.T.astype(np.int8))
                c_gpu, _ = gpu.decode_corrections(
                    sx.T.astype(np.int8), sz.T.astype(np.int8))
                self.assertTrue(np.array_equal(c_cpu, c_gpu),
                                f"{name} osd_order={osd_order} mismatch")

    def test_bp_only_matches_cpu(self):
        for name, builder in CODES.items():
            h_x, h_z = builder()
            l_x, l_z = logicals(h_x, h_z)
            n = h_x.shape[1]
            cpu = BpOnlyDecoder(h_x, h_z, l_x, l_z, 0.1, 0.1)
            gpu = GpuBpOnlyDecoder(h_x, h_z, l_x, l_z, 0.1, 0.1,
                                   device="cpu", gpu_chunk=64)
            e_x, e_z = sample_iid_errors(n, 0.1, 32, "x", seed=2)
            sx = (h_z @ e_x.T) % 2
            sz = (h_x @ e_z.T) % 2
            c_cpu, _ = cpu.decode_corrections(
                sx.T.astype(np.int8), sz.T.astype(np.int8))
            c_gpu, _ = gpu.decode_corrections(
                sx.T.astype(np.int8), sz.T.astype(np.int8))
            self.assertTrue(np.array_equal(c_cpu, c_gpu), name)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_device_matches_cpu(self):
        h_x, h_z = medium_code()
        l_x, l_z = logicals(h_x, h_z)
        n = h_x.shape[1]
        cpu = BpOsdDecoder(h_x, h_z, l_x, l_z, 0.1, 0.1, osd_order=1)
        gpu = GpuBpOsdDecoder(h_x, h_z, l_x, l_z, 0.1, 0.1, osd_order=1,
                              device="cuda", gpu_chunk=64)
        e_x, e_z = sample_iid_errors(n, 0.1, 32, "depolarizing", seed=4)
        sx = (h_z @ e_x.T) % 2
        sz = (h_x @ e_z.T) % 2
        c_cpu, _ = cpu.decode_corrections(
            sx.T.astype(np.int8), sz.T.astype(np.int8))
        c_gpu, _ = gpu.decode_corrections(
            sx.T.astype(np.int8), sz.T.astype(np.int8))
        self.assertTrue(np.array_equal(c_cpu, c_gpu))


if __name__ == "__main__":
    unittest.main()
