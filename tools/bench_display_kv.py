#!/usr/bin/env python3
"""Paired FP8 latent attention timings at a realistic pool size (no weights required).

This measures ONE cache plane/kernel, not whole-model throughput or quality. Each
pair uses identical queries, selected token positions and quantized cache bytes.
Cold trials evict L2 between calls to avoid hiding the I/O backing cost in cache.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics

import torch
from tensorfold.families.glm5_next.cuda import kv8, latent
from tensorfold.families.glm5_next.cuda.display_kv import Settings


def available():
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                    if x.startswith('MemAvailable:'))) * 1024


def timed(fn, flush, repeats):
    times = []
    for _ in range(repeats):
        if flush is not None:
            flush.add_(1)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main(args):
    torch.manual_seed(17)
    from tensorfold.cuda.display_memory import _extension
    _extension()  # compilation/context overhead precedes memory measurements
    torch.cuda.synchronize()
    before = available()
    owner = Settings(True, 3).allocate(args.pool, 512)
    torch.cuda.synchronize()
    display_available = available()
    normal = torch.zeros_like(owner.tensor)
    torch.cuda.synchronize()
    normal_available = available()
    pattern = kv8.quantize_rows(torch.randn((16384, 512), dtype=torch.bfloat16, device='cuda'))
    for start in range(0, args.pool, len(pattern)):
        end = min(args.pool, start + len(pattern))
        normal[start:end].copy_(pattern[:end-start])
        owner.tensor[start:end].copy_(pattern[:end-start])
    del pattern
    gc.collect()
    flush = torch.empty(128 << 20, dtype=torch.uint8, device='cuda').zero_()
    records = []
    for context in (2048, 65536, 1048576):
        for queries in (1, 4, 16, 128):
            q = torch.randn((queries, 32, 512), dtype=torch.bfloat16, device='cuda')
            out = torch.empty_like(q)
            if context == 2048:
                scratch = latent.LatentScratch(queries, 32, latent.chunks_for(2560), 'cuda')
                pos = torch.tensor([context-queries], dtype=torch.int32, device='cuda')
                def run(cache):
                    latent.attention(q, cache, pos, scratch, scale=256**-0.5, nch=scratch.nch, out=out)
            else:
                picks = torch.randint(0, context, (queries, 2051), device='cuda', dtype=torch.int32).sort(dim=1).values
                counts = torch.full((queries,), 2051, device='cuda', dtype=torch.int32)
                def run(cache):
                    latent.sparse_attention(q, cache, picks, counts, out, 256**-0.5)
            run(normal)
            expected = out.clone()
            run(owner.tensor)
            assert torch.equal(out, expected), (context, queries)
            graphs = {}
            for name, cache in (('normal', normal), ('display', owner.tensor)):
                for _ in range(3):
                    run(cache)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(cache)
                if name == 'display':
                    owner.retain_graph(graph)
                graphs[name] = graph
            for cold in (False, True):
                # Alternate the ordering over three pairs to reduce clock/order bias.
                paired = {'normal': [], 'display': []}
                for trial in range(3):
                    order = ('normal', 'display') if trial % 2 == 0 else ('display', 'normal')
                    for name in order:
                        paired[name].append(timed(graphs[name].replay, flush if cold else None, args.repeats))
                a, b = (statistics.median(paired[name]) for name in ('normal', 'display'))
                records.append(dict(context=context, queries=queries, cold_l2=cold, normal_ms=a,
                                    display_ms=b, ratio=b/a, extra_ms=b-a, bitwise_equal=True))
            del graphs, graph
    result = dict(plane=owner.receipt, pool_tokens=args.pool, torch=torch.__version__, cuda=torch.version.cuda,
                  memory=dict(before=before, after_display=display_available, after_normal=normal_available),
                  records=records, passed=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', type=int, default=2922496)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.pool < 1048576 or args.pool % 2048 or args.repeats < 1:
        parser.error('pool must be at least 1048576 and a multiple of 2048; repeats must be positive')
    main(args)
