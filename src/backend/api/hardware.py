"""
AetherAgent Hardware Status Router.
Exposes hardware profile and live utilization data.
"""

from fastapi import APIRouter
from src.core.hardware_manager import (
    HardwareMonitor,
    build_and_save_config,
    load_runtime_config,
)

router = APIRouter(prefix="/hardware", tags=["Hardware"])

# Start background health monitor (alerts logged to console)
_monitor = HardwareMonitor(
    alert_callback=lambda msg: print(f"[HARDWARE ALERT] {msg}"),
    check_interval=15,
)


@router.get("/profile")
async def get_hardware_profile():
    """Return the full hardware profile and inference configuration."""
    config = load_runtime_config()
    return {
        "profile": {
            "environment": "WSL2" if config.profile.is_wsl else "Native Linux",
            "architecture": config.profile.architecture,
            "cpu": {
                "model": config.profile.cpu.model_name,
                "physical_cores": config.profile.cpu.physical_cores,
                "logical_cores": config.profile.cpu.logical_cores,
                "avx2": config.profile.cpu.has_avx2,
                "avx512": config.profile.cpu.has_avx512,
            },
            "gpu": {
                "vendor": config.profile.gpu.vendor,
                "model": config.profile.gpu.model_name,
                "vram_total_mb": config.profile.gpu.vram_total_mb,
                "driver_version": config.profile.gpu.driver_version,
                "cuda_version": config.profile.gpu.cuda_version,
            },
            "ram_total_gb": config.profile.ram_total_gb,
            "disk_free_gb": config.profile.disk_free_gb,
        },
        "inference": {
            "backend": config.inference.backend,
            "quantization": config.inference.quantization,
            "n_gpu_layers": config.inference.n_gpu_layers,
            "context_size": config.inference.context_size,
            "batch_size": config.inference.batch_size,
            "threads": config.inference.threads,
            "max_model_params_gb": config.inference.max_model_params_gb,
        },
    }


@router.get("/status")
async def get_hardware_status():
    """Return live hardware utilization (CPU%, GPU%, RAM%, VRAM%)."""
    return _monitor.get_live_status()


@router.post("/reoptimize")
async def reoptimize_hardware():
    """Re-run hardware detection and update inference configuration."""
    config = build_and_save_config()
    return {
        "message": "Hardware profile re-scanned and config updated",
        "inference": {
            "backend": config.inference.backend,
            "quantization": config.inference.quantization,
            "n_gpu_layers": config.inference.n_gpu_layers,
            "context_size": config.inference.context_size,
        },
    }
