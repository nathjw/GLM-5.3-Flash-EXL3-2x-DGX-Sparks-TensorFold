# Experimental Display-KV

The `feature/display-kv` branch adds an optional complete FP8 DSA latent plane
backed by the Spark's DRM display reservation. It is disabled by default and
remains an experiment until the full checkpoint qualification below passes.

The branch starts at MiaAI recipe commit
`4bbf2f3d83f8b5363832368021c5b56b4ced3d92`, retaining all 53 patches. Patch 0054
owns GLM placement, capacity accounting and graph lifetime. The generic allocator
is independently implemented in `nathjw/TensorFold`, with the upstream release,
repository and exact source commit recorded separately as `TF_VERSION`, `TF_REPO`
and `TF_REF`. No AGPL allocator code was copied.

## Build without downloading the model

```bash
./scripts/build-image.sh
```

This creates `tensorfold-glm53-display:v0.6.0`. It reuses the digest-pinned released
image's CUDA 13.3 / PyTorch 26.07 environment, removes its patched TensorFold
package, installs the exact fork commit, and applies the full 54-patch stack.
Removing the old package directory is necessary because pip does not track files
created by the original patches. The image identity includes source commit,
repository, release, base image, patches, extras and builder script. Building it
does not download weights or change a running server. `scripts/prepare.sh` also
uses this builder when the image is missing.

The image, container name and state directory are distinct from the released
baseline. The checkpoint revisions and weight files are the baseline's. For
cluster preparation with already complete checkpoints, use `scripts/prepare.sh`
as usual. Do not run it alongside another checkpoint preparation.

## Run

Set cluster connection values in `scripts/local.sh` as in the upstream README.
Both hosts need `nvidia_drm modeset=1 fbdev=0`, an inactive display manager and
access to the chosen `/dev/dri/cardN`. The launcher checks these settings and
passes only the selected DRM card to both containers. It does not change host
boot settings or reboot. Runtime allocation verifies CUDA/DRM PCI identity and
must succeed on both ranks before loading weights.

```bash
DISPLAY_KV=1 DISPLAY_KV_PLANE=3 DISPLAY_KV_MIB=1792 ./start.sh
```

`DISPLAY_KV_PLANE` is the zero-based model layer, not its position in the list of
DSA layers. Layer 3 is the first DSA layer in the pinned GLM checkpoint. Only a
complete latent plane is moved; its codes, scales, padding, shape and stride are
identical. Indexer caches, weights, other latent planes and scratch remain in
ordinary memory. This version requires `KV=fp8`, the latent cache, and
`PARALLEL=2..4`. One request at a time also works under `PARALLEL=4`.

For a controlled comparison, first record the baseline's actual shared pool,
then use that same multiple of 2048 for both launches:

```bash
DISPLAY_KV=0 KV_POOL_TOKENS=2922496 ./start.sh
./stop.sh
DISPLAY_KV=1 DISPLAY_KV_PLANE=3 KV_POOL_TOKENS=2922496 ./start.sh
```

The number above is the published example, not a claim about this cluster's
affordable capacity. `KV_POOL_TOKENS=0` permits automatic growth. A requested
fixed size never silently shrinks: a size that cannot hold the context and
reserve, exceeds the display limit, or exceeds ordinary-memory budget fails.
Automatic sizing removes the selected plane's bytes from normal memory,
includes retained state and drafter growth, and caps the whole plane at the
display allocation limit. The full pool can contain at most 3,557,376 tokens
with the 1792 MiB ceiling and 528-byte FP8 latent rows. At 2,922,496 rows,
1,543,077,888 bytes (1.437 GiB) move to display storage. DRM rounds the physical
allocation to a 64 KiB scanline; only the tensor bytes receive a memory credit.

`DISPLAY_KV_CARD` defaults to `/dev/dri/card0`. Failed preflight or allocation
never falls back to ordinary memory. Rollback is `DISPLAY_KV=0` on restart, or
the original recipe and digest-pinned baseline image.

## Qualification

Hardware checks have run on two GB10 nodes with modeset=Y and fbdev=N. Both
passed full 1792 MiB write/readback, CUDA graph replay and repeated allocation
cleanup. The pinned CUDA 13.3 runtime also passed real FP8 attention, overlapping
arena movement, alias ownership and memory-accounting checks. Synthetic EXL3
GLM + DFlash2 checks cover serial/drafted equality, one and four requests,
prefix reuse, eviction and eager/captured verification. These tests use tiny
weights and do not establish full-checkpoint quality or long-context stability.

Tools included here:

- `tests/test_display_kv.py`: CPU sizing and refusal tests.
- `tests/test_display_kv_cuda.py`: actual FP8 kernel and arena tests.
- `tools/bench_display_kv.py --output results.json`: paired per-plane attention
  latency at a 2,922,496-row pool, at 2K, 64K and 1M token positions, with hot
  and cold L2. This is a kernel microbenchmark, not whole-model throughput.
- `tools/check_display_kv_engine.py --source /patched/TensorFold --display 0|1
  --output results.json`: synthetic engine checks using TensorFold's test
  checkpoint generator. `--dense fp8` exercises the alternative dense path.
- `TF_TEST_SOURCE=/patched/TensorFold WORKER=user@worker
  tools/check_display_kv_cluster.sh 0|1`: the same synthetic checks on two real
  GPUs with the recipe's RoCE transport. The source checkout must have the
  recipe patches applied. It creates and removes only its own test containers.

The full qualification still requires complete checkpoint snapshots on both
nodes: Q4 and FP8 baseline boots; the same fixed-pool A/B; native-context
prompts; repeated server restart; startup/free-memory, TTFT, prefill, decode and
four-request throughput measurements; then tasks 546, 530 and 918, two attempts
each, at high effort through the established Claude Code/SWE-Sharp executor-v2
harness, machine grader, verifier and blinded judge. Keep Display-KV optional
until those gates pass. Increased pool capacity is a separate experiment after
the equal-pool comparison.
