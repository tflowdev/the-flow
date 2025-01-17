import argparse
from typing import List, Sequence, Type

import pytest

from theflow.contrib import add_contrib_callback_args
from theflow.contribs.aim import AimCallback
from theflow.contribs.comet import CometCallback
from theflow.contribs.mlflow import MlflowCallback
from theflow.contribs.wandb import WandbCallback


@pytest.mark.parametrize(
    "sys_argv,expected",
    [
        ([], []),
        (["--mlflow"], [MlflowCallback]),
        (["--aim"], [AimCallback]),
        (["--comet"], [CometCallback]),
        (["--wandb"], [WandbCallback]),
    ],
)
def test_add_contrib_callback_args(sys_argv: Sequence[str], expected: List[Type]):
    parser = argparse.ArgumentParser()
    add_contrib_callback_args(parser)
    args = parser.parse_args(sys_argv)
    callbacks = args.callbacks or []

    assert len(callbacks) == len(expected)
    for callback, expected_cls in zip(callbacks, expected):
        assert isinstance(callback, expected_cls)
