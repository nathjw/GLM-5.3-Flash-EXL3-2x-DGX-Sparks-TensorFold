"""Actual GLM FP8 cache writes, relocation and attention through external storage."""
from types import SimpleNamespace
import gc

import pytest
import torch

from tensorfold.families.glm5_next.cuda.display_kv import Settings
from tensorfold.families.glm5_next.cuda.forward import Caches
from tensorfold.families.glm5_next.cuda import kv8, latent

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires NVIDIA DRM and CUDA')


def test_cache_plane_relocation_and_memory_accounting():
    w = SimpleNamespace(cfg=SimpleNamespace(heads=64, kv_lora=512, index_dim=128), world=2,
                        device='cuda', layers=[SimpleNamespace(kind='dsa', index=i) for i in (3, 7)],
                        mtp=None, meta={'long_context': True})
    rows = 65536
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    normal = Caches(w, rows, streams=4, ring=64, kv='fp8')
    normal_bytes = torch.cuda.memory_allocated() - base
    del normal
    gc.collect()
    setting = Settings(True, 3, max_bytes=64 << 20)
    owner = setting.allocate(rows, 512)
    external = Caches(w, rows, streams=4, ring=64, kv='fp8', display=owner)
    assert abs(normal_bytes - (torch.cuda.memory_allocated() - base) - rows * 528) < 2 << 20
    plane = external.arena.planes[external.kc[0]].tensor
    assert plane.data_ptr() == owner.tensor.data_ptr()
    reference = torch.arange(rows, dtype=torch.int32, device='cuda').to(torch.uint8)
    plane.copy_(reference[:, None].expand(-1, 528))
    for src, dst, n in ((0, 2048, 8192), (4096, 0, 8192), (0, 16384, 4096)):
        reference[dst:dst+n] = reference[src:src+n].clone()
        external.arena.copy(src, dst, n)
        assert torch.equal(plane[:, 0], reference)
        assert torch.equal(plane[:, -1], reference)
    view = external.arena.view(0, 4096, 8192)
    assert view.data_ptr() == plane.data_ptr() + 4096 * 528
    del external, owner, plane
    gc.collect()
    assert torch.equal(view[:, 0], reference[4096:12288])


@pytest.mark.parametrize('queries', [1, 4, 16, 128])
def test_fp8_attention_exact_and_graph_replay(queries):
    torch.manual_seed(7)
    rows, width, heads = 16384, 512, 32
    owner = Settings(True, 3, max_bytes=16 << 20).allocate(rows, width)
    codes = kv8.quantize_rows(torch.randn((rows, width), device='cuda', dtype=torch.bfloat16))
    owner.tensor.copy_(codes)
    q = torch.randn((queries, heads, width), dtype=torch.bfloat16, device='cuda')
    picks = torch.randint(0, rows, (queries, 2051), dtype=torch.int32, device='cuda').sort(dim=1).values
    counts = torch.full((queries,), 2051, dtype=torch.int32, device='cuda')
    want = torch.empty_like(q)
    got = torch.empty_like(q)
    latent.sparse_attention(q, codes, picks, counts, want, 256**-0.5)
    latent.sparse_attention(q, owner.tensor, picks, counts, got, 256**-0.5)
    assert torch.equal(want, got)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        latent.sparse_attention(q, owner.tensor, picks, counts, got, 256**-0.5)
    owner.retain_graph(graph)
    for _ in range(3):
        q.add_(0.03125)
        latent.sparse_attention(q, codes, picks, counts, want, 256**-0.5)
        graph.replay()
        assert torch.equal(want, got)
