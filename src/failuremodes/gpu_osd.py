"""Batched ordered statistics decoding on GF(2) in torch, for GPU decoding.

Implements the same OSD post-processing as qudec.bposd._osd with a
swap-free elimination that keeps the pivot selection identical: for each
column in the shot's reliability order, the first unused row (in
increasing row index) holding a 1 becomes the pivot and the column is
eliminated from every other row, including previous pivot rows. Rows are
bit-packed into int64 words so the elimination is pure integer XOR,
vectorized across the batch. Device-agnostic: runs on CUDA when the
tensors are placed there and on CPU otherwise, which the tests exploit.
"""

import numpy as np
import torch


def pack_rows(h, device="cpu"):
    """Pack the rows of a binary (m, n) matrix into int64 words."""
    h = np.asarray(h, dtype=np.int8)
    m, n = h.shape
    n_words = (n + 63) // 64
    out = torch.zeros((m, n_words), dtype=torch.int64, device=device)
    for i in range(m):
        for w in range(n_words):
            chunk = h[i, w * 64:(w + 1) * 64]
            val = 0
            for b, v in enumerate(chunk):
                if v:
                    val |= 1 << b
            if val >= 1 << 63:
                val -= 1 << 64
            out[i, w] = val
    return out


def batched_osd(h, syndromes, orders, osd_order, device="cpu"):
    """Return (B, n) int8 corrections, identical to qudec's _osd output.

    h: (m, n) int8 check matrix; syndromes: (B, m) int8; orders: (B, n)
    int64 column permutations (most likely error first); osd_order: 0
    for OSD-0, 1 for OSD-0 plus the combination sweep.
    """
    m, n = h.shape
    B = orders.shape[0]
    n_words = (n + 63) // 64
    orders = torch.as_tensor(orders, dtype=torch.int64, device=device)
    rows = pack_rows(h, device).unsqueeze(0).expand(B, m, n_words)
    rows = rows.contiguous()
    synd = torch.as_tensor(syndromes, dtype=torch.int64, device=device)
    used = torch.zeros((B, m), dtype=torch.bool, device=device)
    row_idx = torch.arange(m, device=device).unsqueeze(0).expand(B, m)
    pivot_rows = torch.full((B, n), -1, dtype=torch.int64, device=device)
    for k in range(n):
        col = orders[:, k]
        w_idx = (col >> 6)[:, None, None].expand(B, m, 1)
        word = rows.gather(2, w_idx).squeeze(2)
        bit_all = ((word >> (col & 63)[:, None]) & 1).bool()
        candidates = torch.where(bit_all & ~used, row_idx,
                                 torch.full_like(row_idx, m + 1))
        pivot = candidates.min(dim=1).values
        valid = pivot < m
        pivot_rows[:, k] = torch.where(
            valid, pivot, torch.full_like(pivot, -1))
        used = used | (row_idx == pivot[:, None])
        target = bit_all & (row_idx != pivot[:, None]) & valid[:, None]
        pivot_safe = pivot.clamp(min=0, max=m - 1)
        pivot_words = rows.gather(
            1, pivot_safe[:, None, None].expand(B, 1, n_words))
        rows = torch.where(target[:, :, None], rows ^ pivot_words, rows)
        pivot_synd = synd.gather(1, pivot_safe[:, None])
        synd = torch.where(target, synd ^ pivot_synd, synd)

    e = torch.zeros((B, n), dtype=torch.int64, device=device)
    for k in range(n):
        p = pivot_rows[:, k]
        ok = p >= 0
        val = synd.gather(1, p.clamp(min=0)[:, None]).squeeze(1)
        e[:, k] = torch.where(ok, val, torch.zeros_like(val))

    if osd_order >= 1:
        pr = pivot_rows.clamp(min=0)
        pivot_cell = pivot_rows >= 0
        piv_words = rows.gather(1, pr[:, :, None].expand(B, n, n_words))
        for k in range(n):
            col_k = orders[:, k]
            word = piv_words.gather(
                2, (col_k >> 6)[:, None, None].expand(B, n, 1)).squeeze(2)
            rref_col = ((word >> (col_k & 63)[:, None]) & 1) * pivot_cell
            e_piv = e * pivot_cell
            cand = (e_piv ^ rref_col) * pivot_cell
            w_cand = cand.sum(dim=1)
            w_piv = e_piv.sum(dim=1)
            e_k = e[:, k]
            better = (w_cand + 1 < w_piv + 2 * e_k) & ~pivot_cell[:, k]
            update_cell = pivot_cell & better[:, None]
            e = torch.where(update_cell, cand, e)
            e[:, k] = torch.where(better, e_k ^ 1, e_k)

    corr = torch.zeros((B, n), dtype=torch.int64, device=device)
    corr = corr.scatter(1, orders, e)
    return corr.cpu().numpy().astype(np.int8)
