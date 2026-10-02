#!/usr/bin/env bash
# Read-only host checks. Never changes boot settings, stops a display manager, or reboots.
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source scripts/config.sh
source scripts/nodes.sh
[[ "$DISPLAY_KV" == 1 ]] || die 'Set DISPLAY_KV=1 for this preflight'
[[ "$DISPLAY_KV_CARD" =~ ^/dev/dri/card[0-9]+$ ]] || die 'DISPLAY_KV_CARD must be /dev/dri/cardN'
[[ "$DISPLAY_KV_PLANE" =~ ^[0-9]+$ ]] || die 'Set DISPLAY_KV_PLANE to a DSA model layer (e.g. 3)'
[[ "$DISPLAY_KV_MIB" =~ ^[0-9]+$ ]] && (( DISPLAY_KV_MIB >= 1 && DISPLAY_KV_MIB <= 1792 )) ||
  die 'DISPLAY_KV_MIB must be 1..1792'
[[ "$KV" == fp8 && "$PARALLEL" =~ ^[234]$ ]] || die 'Display-KV needs KV=fp8 and PARALLEL=2..4'
need_worker
systemctl is-active --quiet display-manager && die 'Head display-manager is active'
worker systemctl is-active --quiet display-manager && die 'Worker display-manager is active'
# The container can read root-only sysfs parameters, with only the selected DRM card passed through.
check='from pathlib import Path; import os
root=Path("/sys/module/nvidia_drm/parameters")
assert (root/"modeset").read_text().strip() in ("Y", "1"), "nvidia_drm modeset must be enabled"
assert (root/"fbdev").read_text().strip() in ("N", "0"), "nvidia_drm fbdev must be disabled"
assert os.access(os.environ["CARD"], os.R_OK | os.W_OK), "DRM card is not readable/writable"
print("DRM host preflight passed:", os.environ["CARD"])'
docker run --rm --pull=never --device "$DISPLAY_KV_CARD" -e CARD="$DISPLAY_KV_CARD" \
  --entrypoint python "$IMAGE" -c "$check"
remote=(docker run --rm --pull=never --device "$DISPLAY_KV_CARD" -e CARD="$DISPLAY_KV_CARD"
        --entrypoint python "$IMAGE" -c "$check")
printf -v command '%q ' "${remote[@]}"
worker "$command"
log 'Both hosts passed Display-KV preflight; runtime allocation must still succeed on both ranks'
