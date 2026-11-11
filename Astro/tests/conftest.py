import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

from hopper.params import load_params

CONFIG = os.path.join(ROOT, "config.yaml")


@pytest.fixture(scope="session")
def params():
    return load_params(CONFIG)


@pytest.fixture(scope="session")
def make_params():
    return lambda overrides=None: load_params(CONFIG, overrides)


@pytest.fixture(scope="session")
def single_stage_params_factory():
    """The tracker regression suite exercises its supported one-stage interface."""
    def make(overrides=None):
        cfg = dict(overrides or {})
        engine = dict(cfg.get("engine", {}))
        engine["stages"] = dict(ascent=[dict(large=3)], braking=[dict(large=3)])
        cfg["engine"] = engine
        return load_params(CONFIG, cfg)
    return make
