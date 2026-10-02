#!/usr/bin/env bash
# Build only the experimental image; no model downloads, worker copy, or server changes.
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh
nocache=()
case "${1:-}" in
  "") ;;
  --rebuild|--no-cache) nocache=(--no-cache) ;;
  *) echo "Usage: $0 [--rebuild]" >&2; exit 2 ;;
esac
docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || docker pull "$BASE_IMAGE"
PATCHES_HASH=$(image_hash)
docker build "${nocache[@]}" -t "$IMAGE" --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg TF_SPEC="git+${TF_REPO}@${TF_REF}" --build-arg PATCHES_HASH="$PATCHES_HASH" --build-arg EXTRAS="$IMAGE_EXTRAS" \
    --build-arg TF_VERSION="$TF_VERSION" --build-arg TF_REPO="$TF_REPO" --build-arg TF_REF="$TF_REF" \
    -f - patches <<'DOCKERFILE'
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}
ARG TF_SPEC
ARG EXTRAS
RUN test -f /usr/include/drm/drm.h -o -f /usr/include/libdrm/drm.h || \
    (apt-get update && apt-get install -y --no-install-recommends libdrm-dev && rm -rf /var/lib/apt/lists/*)
# The released base contains patch-created files absent from pip's RECORD. A pip
# reinstall alone leaves them behind and makes applying the patch stack fail.
RUN python -c 'import pathlib, shutil, tensorfold; p=pathlib.Path(tensorfold.__file__).parent; assert p.name == "tensorfold" and p.parent.name in ("site-packages", "dist-packages"); shutil.rmtree(p)'
RUN pip install --no-cache-dir --force-reinstall --no-deps "${TF_SPEC}" && pip install --no-cache-dir ${EXTRAS} && tensorfold --version
COPY . /opt/tf-patches
RUN cd "$(python -c 'import os, tensorfold; print(os.path.dirname(os.path.dirname(tensorfold.__file__)))')" && \
    for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; patch -p0 --forward < "$p" || exit 1; done && \
    python -c "import tensorfold.cuda.server, tensorfold.families.glm5_next.cuda.engine, tensorfold.vision.glm, av, xgrammar"
ARG PATCHES_HASH
ARG TF_VERSION
ARG TF_REPO
ARG TF_REF
LABEL tf.patches=${PATCHES_HASH}
LABEL tf.version=${TF_VERSION} tf.repo=${TF_REPO} tf.ref=${TF_REF}
LABEL org.opencontainers.image.source="https://github.com/nathjw/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold"
LABEL org.opencontainers.image.description="Experimental GLM Display-KV; not yet qualified on the full checkpoint"
ENV HF_HOME=/root/.cache/huggingface TORCH_EXTENSIONS_DIR=/cache/torch_extensions TRITON_CACHE_DIR=/cache/triton
WORKDIR /workspace
DOCKERFILE
log "Built $IMAGE: source $TF_REPO@$TF_REF; release $TF_VERSION; identity $PATCHES_HASH"
