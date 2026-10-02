#!/usr/bin/env python3
"""Synthetic GLM + DFlash2 qualification of the recipe's actual scheduler and cache path.

Uses TensorFold's Apache-2.0 test checkpoint generator from a patched source checkout.
No production checkpoint or model download is needed. One GPU emulates two ranks; the
real two-node checkpoint qualification remains a separate gate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--display', type=int, choices=(0, 1), required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--rank', type=int, choices=(0, 1), default=None,
                    help='Omit to emulate both ranks on one GPU; specify on each host for actual two-rank execution')
parser.add_argument('--master', default='192.168.100.10')
parser.add_argument('--master-port', type=int, default=29586)
parser.add_argument('--dense', choices=('q4', 'fp8'), default='q4')
args = parser.parse_args()
sys.path.insert(0, str(args.source / 'tests/cuda'))
# These are identical in both subprocesses except the allocation backing.
os.environ.update(TF_GLM_KV='fp8', TF_GLM_DENSE=args.dense, TF_GLM_MTP='0', TF_GLM_KVB='bf16',
                  TF_GLM_DISPLAY_KV=str(args.display), TF_GLM_DISPLAY_KV_PLANE='1',
                  TF_GLM_DISPLAY_KV_MIB='16', TF_GLM_KV_POOL_TOKENS='16384',
                  TF_GLM_CACHE_GIB='0.5', TF_GLM_CACHE_ENTRIES='4', TF_GLM_PREFILL_ROWS='64',
                  TF_GLM_PROMPT_GRID='64', TF_GLM_SHARED_PREFIX='64',
                  TF_GLM_DFLASH_POLICY='fnc7:0.3', TF_GLM_KDA_CHUNKED='1', TF_GLM_L2PF='0',
                  TF_GLM_MULTI_LONE='0', TF_GLM_MULTI_PREFILL='1')

import numpy as np
import torch
from test_glm_engine import _checkpoint, _drafter, _TwoCopies
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda.engine import GlmEngine


with tempfile.TemporaryDirectory(prefix='display-kv-synthetic-') as tmp:
    root = Path(tmp)
    _checkpoint(root / 'model', exl3=True, mtp=False)
    _drafter(root / 'draft')
    engine = GlmEngine(root / 'model', rank=args.rank or 0, master=args.master, port=args.master_port,
                       drafter=root / 'draft', comm=_TwoCopies() if args.rank is None else None,
                       parallel=4, context=4096)
    assert engine.e.caches.rows == 16384
    assert bool(engine.display_storage) == bool(args.display)
    if args.rank == 1:
        args.output.write_text(json.dumps({'rank': 1, 'ready': True, 'capacity': engine.capacity_plan}) + '\n')
        engine.follow()
        raise SystemExit(0)
    rng = np.random.default_rng(12)
    prompts = [list(map(int, rng.integers(0, 999, n))) for n in (128, 256, 512, 2304)]
    results = []

    def run(prompt, draft=True, gate=None):
        if gate is not None:
            gate.wait()
        out = []
        # Use the server's engine entry point: it supplies the GLM policy to the
        # scheduler. Direct scheduler.submit without glm.code silently selects serial.
        engine.request.policy = 'fnc7:0.3'
        engine.request.stop_eos = False
        stats = engine.generate(prompt, 24, Sampling(77, 1.0, 20, 0.95),
                                lambda new: out.extend(new), draft=draft)
        if draft:
            assert stats['drafted'] > 0, 'test did not exercise DFlash2 drafting'
        return {'tokens': out, 'cached': stats['cached'], 'drafted': stats['drafted']}

    # Dense and sparse attention, then the same requests through DFlash2.
    for prompt in prompts:
        serial = run(prompt, False)
        drafted = run(prompt)
        assert drafted['tokens'] == serial['tokens'], 'DFlash2 changed serial output'
        results.append({'serial': serial, 'drafted': drafted})
    # Four live streams with distinct extents and a shared prefix.
    shared = prompts[0][:96]
    concurrent_prompts = [shared + p[96:] for p in prompts]
    barrier = threading.Barrier(4)
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(run, p, True, barrier) for p in concurrent_prompts]
        concurrent = [job.result() for job in jobs]
    for prompt, got in zip(concurrent_prompts, concurrent):
        assert run(prompt, False)['tokens'] == got['tokens'], 'concurrent differs from serial'
    # Retained turns and eviction beyond the four retained entries.
    prime = run(prompts[0])
    repeat = run(prompts[0])
    assert repeat['cached'] > 0 and repeat['tokens'] == prime['tokens'], 'identical resend missed retained prefix'
    continuation = prompts[0] + prime['tokens'] + [12, 34]
    warm = run(continuation)
    assert warm['cached'] > 0, 'continuation did not reuse cached state'
    for i in range(8):
        run([900-i] * (128 + i * 64))
    cold = run(continuation)
    assert cold['cached'] == 0, 'eviction test did not evict the retained state'
    assert warm['tokens'] == cold['tokens'], 'retained/evicted cache changed output'
    # Run the same decode once with captured verification and once eager.
    saved = engine.multi.verify.graphs
    engine.multi.verify.graphs = {}
    eager = run(prompts[0])
    engine.multi.verify.graphs = saved
    captured = run(prompts[0])
    assert eager['tokens'] == captured['tokens'], 'eager differs from graph replay'
    torch.cuda.synchronize()
    result = {'passed': True, 'display': args.display, 'dense': args.dense,
              'real_two_rank': args.rank is not None, 'capacity': engine.capacity_plan,
              'cases': results, 'concurrent': concurrent, 'repeat': repeat,
              'continuation': warm, 'after_eviction': cold,
              'eager': eager, 'captured': captured,
              'torch': torch.__version__, 'cuda': torch.version.cuda}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'display': args.display, 'output': str(args.output)}), flush=True)
