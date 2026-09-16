"""Offline tests of the experiment-owned remote regression entry point."""
import importlib.util
from pathlib import Path
import pytest
import torch

spec=importlib.util.spec_from_file_location("remote_regression",Path(__file__).parents[1]/"scripts/remote_regression.py")
regression=importlib.util.module_from_spec(spec)
spec.loader.exec_module(regression)


@pytest.mark.parametrize("method",["none","third","jackknife"])
@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_remote_ragged_harness_matches_independent_reference(method,dtype):
    report=regression.ragged_case("cpu",dtype,method,1.,1e-5)
    assert report["zero_padding_gradient"]
    assert report["exact_full_low_empty_fallback"]
    assert report["fallback_rows"]==(1 if method=="none" else 5)
    assert report["clipped_rows"]==(0 if method=="none" else 3)


@pytest.mark.parametrize("method",["third","jackknife"])
def test_remote_zero_and_near_zero_backward(method):
    for delta in (0.,1e-7):
        regression.coincidence_case("cpu",torch.float32,method,delta)


def test_cuda_request_cannot_silently_fall_back(monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    with pytest.raises(RuntimeError,match="no CPU fallback"):
        regression.run_checks("cuda")
