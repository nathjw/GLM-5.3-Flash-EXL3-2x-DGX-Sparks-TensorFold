"""Capacity and fail-closed policy checks; runs against the patched source, without a GPU."""
from types import SimpleNamespace

import pytest

from tensorfold.families.glm5_next.cuda.display_kv import Settings, pool_rows


def test_plane_size_includes_scale_and_padding():
    setting = Settings(True, 3)
    assert setting.offloaded(2922496, 512) == 1543077888
    assert setting.max_rows(512) == 3557376
    assert setting.offloaded(setting.max_rows(512), 512) <= 1792 << 20
    assert (setting.max_rows(512) + 2048) * 528 > 1792 << 20


@pytest.mark.parametrize('values', [
    {'TF_GLM_DISPLAY_KV': 'yes'},
    {'TF_GLM_DISPLAY_KV': '1'},
    {'TF_GLM_DISPLAY_KV': '1', 'TF_GLM_DISPLAY_KV_PLANE': '3', 'TF_GLM_DISPLAY_KV_MIB': '2048'},
    {'TF_GLM_KV_POOL_TOKENS': '-2048'},
    {'TF_GLM_KV_POOL_TOKENS': '8193'},
])
def test_invalid_environment(values):
    with pytest.raises(ValueError):
        Settings.from_env(values)


def test_disabled_does_not_require_display_configuration():
    s = Settings.from_env({'TF_GLM_DISPLAY_KV': '0', 'TF_GLM_DISPLAY_KV_PLANE': 'invalid'})
    assert not s.enabled and s.offloaded(1000000, 512) == 0


def test_layer_and_format_are_validated():
    cfg = SimpleNamespace(kinds=['kda', 'kda', 'kda', 'dsa'], kv_lora=512)
    Settings(True, 3).validate(cfg, parallel=4, latent=True, kv='fp8')
    for setting, parallel, latent, kv in ((Settings(True, 0), 4, True, 'fp8'),
                                         (Settings(True, 3), 1, True, 'fp8'),
                                         (Settings(True, 3), 4, False, 'fp8'),
                                         (Settings(True, 3), 4, True, 'bf16')):
        with pytest.raises(ValueError):
            setting.validate(cfg, parallel=parallel, latent=latent, kv=kv)


def test_auto_pool_obeys_both_memory_budgets():
    s = Settings(True, 3, max_bytes=16 << 20)
    kwargs = dict(capacity=8192, automatic=1000000, width=512,
                  growth_bytes=lambda n: (n - 8192) * 6000 + 100000, allowance=10**12)
    assert pool_rows(s, **kwargs) == s.max_rows(512)
    kwargs['allowance'] = (16384 - 8192) * 6000 + 100000
    assert pool_rows(s, **kwargs) == 16384


@pytest.mark.parametrize('forced,allowance', [(4096, 10**9), (32768, 10**9), (16384, 10)])
def test_forced_pool_never_shrinks_or_overcommits(forced, allowance):
    with pytest.raises(ValueError):
        pool_rows(Settings(True, 3, max_bytes=16 << 20, pool_tokens=forced),
                  capacity=8192, automatic=20000, width=512,
                  growth_bytes=lambda n: (n - 8192) * 6000 + 100000, allowance=allowance)


def test_fixed_pool_works_with_display_off():
    assert pool_rows(Settings(pool_tokens=16384), capacity=8192, automatic=20000, width=512,
                     growth_bytes=lambda n: (n - 8192) * 6000, allowance=10**9) == 16384
