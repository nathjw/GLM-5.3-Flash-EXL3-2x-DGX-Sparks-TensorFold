#!/usr/bin/env bash
# Two actual GPUs and the recipe's RoCE transport, using tiny synthetic weights.
# Requires a source checkout with the recipe patches applied (TF_TEST_SOURCE).
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source scripts/config.sh
source scripts/nodes.sh
[[ "${1:-}" =~ ^[01]$ ]] || die 'Usage: TF_TEST_SOURCE=/patched/source tools/check_display_kv_cluster.sh 0|1'
MODE=$1
SOURCE=$(realpath "${TF_TEST_SOURCE:?Set TF_TEST_SOURCE to the patched TensorFold checkout}")
[[ -f "$SOURCE/tests/cuda/test_glm_engine.py" ]] || die 'Synthetic checkpoint helper is missing'
TEST_IMAGE="${TF_TEST_IMAGE:-tensorfold-glm53:v0.6.0}"
TEST_ROOT="${TF_TEST_ROOT:-/tmp/profile26-display-kv-cluster}"
RESULTS="$TEST_ROOT/results-$MODE"
REMOTE_ROOT=/tmp/profile26-display-kv-cluster
HEAD_NAME=profile26-synthetic-rank0
WORKER_NAME=profile26-synthetic-rank1
KEY="${DGX_CLUSTER_SSH_KEY:-$HOME/.ssh/id_ed25519_dgx_cluster}"
started_head=0; started_worker=0
eval "$(ssh-agent -s)" >/dev/null
cleanup() {
  (( started_head == 0 )) || docker rm -f "$HEAD_NAME" >/dev/null 2>&1 || true
  (( started_worker == 0 )) || worker "docker rm -f '$WORKER_NAME'" >/dev/null 2>&1 || true
  ssh-agent -k >/dev/null 2>&1 || true
}
trap cleanup EXIT
ssh-add "$KEY" >/dev/null
need_worker
detect_link
! docker inspect "$HEAD_NAME" >/dev/null 2>&1 || die "Test container $HEAD_NAME already exists"
! worker "docker inspect '$WORKER_NAME' >/dev/null 2>&1" || die "Test container $WORKER_NAME already exists"
mkdir -p "$RESULTS" "$TEST_ROOT/cache"
worker "mkdir -p '$REMOTE_ROOT/source' '$REMOTE_ROOT/tools' '$REMOTE_ROOT/results-$MODE' '$REMOTE_ROOT/cache'"
rsync -a --exclude .git "$SOURCE/" "$WORKER:$REMOTE_ROOT/source/"
rsync -a tools/check_display_kv_engine.py "$WORKER:$REMOTE_ROOT/tools/"
common=(--gpus all --device /dev/dri/card0 --device /dev/infiniband --ipc=host --network host
        --cap-add IPC_LOCK --ulimit memlock=-1 -e MAX_JOBS=2 -e TF_GLM_COMM=roce -e TF_ROCE_MAX_KB=512
        -e PYTHONPATH=/src/src -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e TRITON_CACHE_DIR=/cache/triton)
command='set -e; python -m pip install --quiet pytest==8.4.2; exec python /checks/check_display_kv_engine.py "$@"'
docker run -d --pull=never --name "$HEAD_NAME" "${common[@]}" $(nccl_env "$HEAD_DEV" "$HEAD_HCAS" "$HEAD_GID") \
  -v "$SOURCE:/src:ro" -v "$PWD/tools:/checks:ro" -v "$RESULTS:/results" -v "$TEST_ROOT/cache:/cache" \
  --entrypoint /bin/bash "$TEST_IMAGE" -c "$command" bash --source /src --display "$MODE" --rank 0 \
  --master "$HEAD_ADDR" --master-port 29586 --output /results/rank0.json >/dev/null
started_head=1
remote=(docker run -d --pull=never --name "$WORKER_NAME" "${common[@]}"
        $(nccl_env "$WORKER_DEV" "$WORKER_HCAS" "$WORKER_GID")
        -v "$REMOTE_ROOT/source:/src:ro" -v "$REMOTE_ROOT/tools:/checks:ro"
        -v "$REMOTE_ROOT/results-$MODE:/results" -v "$REMOTE_ROOT/cache:/cache"
        --entrypoint /bin/bash "$TEST_IMAGE" -c "$command" bash --source /src --display "$MODE" --rank 1
        --master "$HEAD_ADDR" --master-port 29586 --output /results/rank1.json)
printf -v remote_command '%q ' "${remote[@]}"
worker "$remote_command" >/dev/null
started_worker=1
log "Synthetic two-rank Display-KV=$MODE running; logs: docker logs $HEAD_NAME; worker: $WORKER_NAME"
code=$(timeout 1800 docker wait "$HEAD_NAME") || code=124
docker logs "$HEAD_NAME" >"$RESULTS/rank0.log" 2>&1
worker "docker logs '$WORKER_NAME' 2>&1" >"$RESULTS/rank1.log"
rsync -a "$WORKER:$REMOTE_ROOT/results-$MODE/" "$RESULTS/"
[[ "$code" == 0 ]] || { tail -60 "$RESULTS/rank0.log"; die "Rank 0 exited $code; results $RESULTS"; }
python3 - "$RESULTS" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
a=json.loads((p/'rank0.json').read_text()); b=json.loads((p/'rank1.json').read_text())
assert a['passed'] and a['real_two_rank'] and b['ready']
assert a['capacity']['pool_tokens'] == b['capacity']['pool_tokens']
print('Two-rank synthetic qualification passed:', p)
PY
