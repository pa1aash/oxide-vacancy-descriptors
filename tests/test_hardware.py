"""Hardware record of every cluster result."""
from types import SimpleNamespace

import torch

from dftgnn import hardware as H


def test_cpu_record_has_hosts_and_no_gpu(monkeypatch):
    monkeypatch.setenv("NODE_NAME", "node-a.example")
    r = H.hardware_info("cpu")
    assert r["gpu_model"] is None and r["gpu_uuid"] is None and r["device"] == "cpu"
    assert r["node_hostname"] == "node-a.example" and r["pod_hostname"]


def test_cuda_record_from_torch_and_smi(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "NVIDIA A10")
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: SimpleNamespace(uuid="4ead8915-27fd", total_memory=23028 * 2**20))
    monkeypatch.setattr(H, "_smi", lambda *a: "615.71.09\n" if a else "| CUDA Version: 13.0 |")
    r = H.hardware_info("cuda")
    assert r["gpu_model"] == "NVIDIA A10" and r["gpu_uuid"] == "GPU-4ead8915-27fd"
    assert r["driver"] == "615.71.09" and r["gpu_memory"] == "23028 MiB" and r["cuda_driver_version"] == "13.0"


def test_missing_nvidia_smi_gives_none(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "NVIDIA L4")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: SimpleNamespace(total_memory=2**34))
    monkeypatch.setattr(H, "_smi", lambda *a: None)
    r = H.hardware_info("cuda:0")
    assert r["gpu_model"] == "NVIDIA L4" and r["driver"] is None and r["gpu_uuid"] is None


def _cuda(monkeypatch, banner):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "NVIDIA A10")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: SimpleNamespace(total_memory=2**34))
    monkeypatch.setattr(H, "_smi", lambda *a: "615.71.09\n" if a else banner)


def test_cuda_version_from_the_driver_api_when_the_banner_has_none(monkeypatch):
    _cuda(monkeypatch, "| NVIDIA-SMI 615.71.09   Driver Version: 615.71.09 |")
    monkeypatch.setattr(torch._C, "_cuda_getDriverVersion", lambda: 13020, raising=False)
    r = H.hardware_info("cuda")
    assert r["driver"] == "615.71.09" and r["cuda_driver_version"] == "13.2"


def test_cuda_driver_version_banner_variant(monkeypatch):
    _cuda(monkeypatch, "| NVIDIA-SMI 615.71.09  Driver Version: 615.71.09  CUDA Driver Version : 13.3 |")
    assert H.hardware_info("cuda")["cuda_driver_version"] == "13.3"


def test_cuda_version_none_without_banner_or_driver_api(monkeypatch):
    _cuda(monkeypatch, None)
    monkeypatch.setattr(H, "_cuda_driver_api_version", lambda: None)
    assert H.hardware_info("cuda")["cuda_driver_version"] is None
