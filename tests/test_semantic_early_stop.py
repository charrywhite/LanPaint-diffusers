"""Unit tests for the semantic early-stop wiring.

No model weights required. Checks the config defaults, the config -> model
options mapping (the guarantee that it is a no-op unless enabled), and that the
upstream LanPaint early stopper actually honors the option we pass.

Run with: pytest tests/test_semantic_early_stop.py
"""

import pytest

from lanpaint_pipeline.pipeline import LanPaintConfig, semantic_stop_options


def test_config_defaults_disable_early_stop():
    c = LanPaintConfig()
    assert c.semantic_stop_threshold == 0.0
    assert c.semantic_stop_patience == 1
    # default config produces no model option -> behavior identical to before
    assert semantic_stop_options(c) == {}


def test_options_populated_only_when_threshold_set():
    off = LanPaintConfig(semantic_stop_threshold=0.0, semantic_stop_patience=3)
    assert semantic_stop_options(off) == {}

    on = LanPaintConfig(semantic_stop_threshold=0.02, semantic_stop_patience=2)
    opts = semantic_stop_options(on)
    assert opts == {"lanpaint_semantic_stop": {"threshold": 0.02, "patience": 2}}


def test_upstream_stopper_honors_the_option():
    """Contract we rely on: the LanPaint package reads lanpaint_semantic_stop
    from model_options and enables/disables accordingly."""
    import torch
    from LanPaint.earlystop import LanPaintEarlyStopper

    # latent_mask: 1 = keep, 0 = edit. Needs a real edit region for the stopper
    # to have something to watch.
    mask = torch.ones(1, 1, 8, 8)
    mask[..., :4] = 0.0
    common = dict(latent_mask=mask, abt=torch.tensor(0.5), default_threshold=0.0,
                  default_patience=1, default_distance_fn=None)

    # from_options returns None when disabled, a stopper instance when enabled.
    assert LanPaintEarlyStopper.from_options(model_options={}, **common) is None
    enabled = LanPaintEarlyStopper.from_options(
        model_options={"lanpaint_semantic_stop": {"threshold": 0.05, "patience": 1}}, **common)
    assert enabled is not None and enabled.enabled is True
