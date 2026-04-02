"""
AetherAgent Hardware Manager
Detects CPU, GPU, RAM, Storage, and auto-configures inference parameters.
"""

import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import psutil


@dataclass
class CPUProfile:
    model_name: str = "Unknown"
    physical_cores: int = 0
    logical_cores: int = 0
    max_freq_mhz: float = 0.0
    has_avx2: bool = False
    has_avx512: bool = False
    has_neon: bool = False


@dataclass
class GPUProfile:
    vendor: str = "none"
    model_name: str = "None"
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    driver_version: str = ""
    cuda_version: str = ""
    temperature_c: float = 0.0
    utilization_pct: float = 0.0


@dataclass
class SystemProfile:
    cpu: CPUProfile = field(default_factory=CPUProfile)
    gpu: GPUProfile = field(default_factory=GPUProfile)
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    ram_used_pct: float = 0.0
    swap_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    is_wsl: bool = False
    architecture: str = "x86_64"


@dataclass
class InferenceConfig:
    backend: str = "llama_cpp"
    n_gpu_layers: int = -1
    context_size: int = 8192
    batch_size: int = 512
    threads: int = 8
    quantization: str = "Q5_K_M"
    max_model_params_gb: float = 0.0
    use_mmap: bool = True
    use_mlock: bool = False
    numa: bool = False


@dataclass
class RuntimeConfig:
    profile: SystemProfile = field(default_factory=SystemProfile)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    config_path: str = ""


def detect_cpu() -> CPUProfile:
    profile = CPUProfile()
    cores_per_socket = 0
    sockets = 0
    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("Model name:"):
                profile.model_name = line.split(":", 1)[1].strip()
            elif line.startswith("CPU(s):"):
                profile.logical_cores = int(line.split(":", 1)[1].strip())
            elif line.startswith("Core(s) per socket:"):
                cores_per_socket = int(line.split(":", 1)[1].strip())
            elif line.startswith("Socket(s):"):
                sockets = int(line.split(":", 1)[1].strip())
            elif line.startswith("Thread(s) per core:"):
                pass
            elif line.startswith("CPU max MHz:"):
                try:
                    profile.max_freq_mhz = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("Architecture:"):
                arch = line.split(":", 1)[1].strip()
                if "aarch64" in arch or "arm" in arch:
                    profile.has_neon = True
        if cores_per_socket > 0 and sockets > 0:
            profile.physical_cores = cores_per_socket * sockets
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    if profile.logical_cores == 0:
        profile.logical_cores = psutil.cpu_count(logical=True) or 4
    if profile.physical_cores == 0:
        profile.physical_cores = psutil.cpu_count(logical=False) or profile.logical_cores
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
        profile.has_avx2 = "avx2" in cpuinfo
        profile.has_avx512 = "avx512f" in cpuinfo
        if not profile.has_neon:
            profile.has_neon = "neon" in cpuinfo.lower()
    except FileNotFoundError:
        pass
    return profile


def detect_gpu() -> GPUProfile:
    profile = GPUProfile()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 6:
                profile.vendor = "nvidia"
                profile.model_name = parts[0]
                profile.vram_total_mb = int(float(parts[1]))
                profile.vram_free_mb = int(float(parts[2]))
                profile.driver_version = parts[3]
                profile.temperature_c = float(parts[4])
                profile.utilization_pct = float(parts[5])
        result3 = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        for line in result3.stdout.splitlines():
            if "CUDA Version:" in line:
                profile.cuda_version = line.split("CUDA Version:")[1].strip().split()[0]
                break
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return profile


def detect_system() -> SystemProfile:
    profile = SystemProfile()
    profile.is_wsl = os.path.exists("/proc/version") and "microsoft" in open("/proc/version").read().lower()
    try:
        profile.architecture = subprocess.check_output(["uname", "-m"], text=True).strip()
    except Exception:
        profile.architecture = "x86_64"
    profile.cpu = detect_cpu()
    profile.gpu = detect_gpu()
    vm = psutil.virtual_memory()
    profile.ram_total_gb = round(vm.total / (1024**3), 1)
    profile.ram_available_gb = round(vm.available / (1024**3), 1)
    profile.ram_used_pct = round(vm.percent, 1)
    swap = psutil.swap_memory()
    profile.swap_total_gb = round(swap.total / (1024**3), 1)
    try:
        disk = psutil.disk_usage(os.path.expanduser("~"))
        profile.disk_free_gb = round(disk.free / (1024**3), 1)
    except Exception:
        profile.disk_free_gb = 0.0
    return profile


def calculate_inference_config(profile: SystemProfile) -> InferenceConfig:
    config = InferenceConfig()
    config.threads = max(4, profile.cpu.logical_cores - 2)
    if profile.gpu.vendor == "nvidia" and profile.gpu.vram_total_mb > 0:
        config.backend = "llama_cpp"
        vram_gb = profile.gpu.vram_total_mb / 1024.0
        ram_gb = profile.ram_total_gb
        if profile.gpu.vram_total_mb >= 7500:
            config.n_gpu_layers = -1
            config.quantization = "Q5_K_M"
            config.context_size = 8192
            config.batch_size = 512
            config.max_model_params_gb = round(vram_gb * 0.85, 1)
            config.use_mlock = False
        elif profile.gpu.vram_total_mb >= 6000:
            config.n_gpu_layers = -1
            config.quantization = "Q4_K_M"
            config.context_size = 4096
            config.batch_size = 256
            config.max_model_params_gb = round(vram_gb * 0.80, 1)
        else:
            config.n_gpu_layers = 20
            config.quantization = "Q4_K_M"
            config.context_size = 4096
            config.batch_size = 256
            config.max_model_params_gb = round((vram_gb + (ram_gb * 0.3)) * 0.80, 1)
            config.use_mlock = True
    else:
        config.backend = "llama_cpp"
        config.n_gpu_layers = 0
        config.quantization = "Q4_K_M"
        config.context_size = 4096
        config.batch_size = 128
        config.use_mlock = True
        config.max_model_params_gb = round(max(0, (profile.ram_total_gb - 4) * 0.70), 1)
        config.threads = profile.cpu.logical_cores
    return config


def get_config_path() -> Path:
    if os.path.exists("/proc/version") and "microsoft" in open("/proc/version").read().lower():
        return Path.home() / "AetherAgent" / "runtime_config.json"
    return Path("/etc/aetheragent/runtime_config.json")


def load_runtime_config() -> RuntimeConfig:
    config_path = get_config_path()
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
            # FIX: Manually reconstruct nested dataclasses from dicts
            p = data["profile"]
            i = data["inference"]
            profile = SystemProfile(
                cpu=CPUProfile(**p["cpu"]),
                gpu=GPUProfile(**p["gpu"]),
                ram_total_gb=p["ram_total_gb"],
                ram_available_gb=p["ram_available_gb"],
                ram_used_pct=p["ram_used_pct"],
                swap_total_gb=p["swap_total_gb"],
                disk_free_gb=p["disk_free_gb"],
                is_wsl=p["is_wsl"],
                architecture=p["architecture"],
            )
            inference = InferenceConfig(**i)
            return RuntimeConfig(profile=profile, inference=inference, config_path=str(config_path))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return build_and_save_config()


def build_and_save_config() -> RuntimeConfig:
    profile = detect_system()
    inference = calculate_inference_config(profile)
    config = RuntimeConfig(profile=profile, inference=inference)
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.config_path = str(config_path)
    data = {"profile": asdict(profile), "inference": asdict(inference)}
    with open(config_path, "w") as f:
        json.dump(data, f, indent=2)
    return config


class HardwareMonitor:
    def __init__(self, alert_callback=None, check_interval: int = 10):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._alert_callback = alert_callback
        self._check_interval = check_interval
        self._last_alert_time: dict[str, float] = {}

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _should_alert(self, key: str, cooldown: int = 60) -> bool:
        now = time.time()
        if key in self._last_alert_time and (now - self._last_alert_time[key]) < cooldown:
            return False
        self._last_alert_time[key] = now
        return True

    def _monitor_loop(self):
        while self._running:
            try:
                self._check_gpu_temp()
                self._check_ram_usage()
            except Exception:
                pass
            time.sleep(self._check_interval)

    def _check_gpu_temp(self):
        try:
            gpu = detect_gpu()
            if gpu.vendor == "nvidia" and gpu.temperature_c > 85:
                if self._should_alert("gpu_temp"):
                    self._alert_callback(f"GPU WARNING: {gpu.model_name} at {gpu.temperature_c}C")
        except Exception:
            pass

    def _check_ram_usage(self):
        try:
            vm = psutil.virtual_memory()
            if vm.percent > 90:
                if self._should_alert("ram_usage"):
                    self._alert_callback(f"RAM WARNING: {vm.percent}% used ({vm.available / (1024**3):.1f}GB free)")
        except Exception:
            pass

    def get_live_status(self) -> dict:
        gpu = detect_gpu()
        vm = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=0.5)
        return {
            "cpu_percent": cpu_pct,
            "ram_percent": vm.percent,
            "ram_available_gb": round(vm.available / (1024**3), 2),
            "ram_used_gb": round((vm.total - vm.available) / (1024**3), 2),
            "gpu_percent": gpu.utilization_pct if gpu.vendor == "nvidia" else 0,
            "gpu_temperature_c": gpu.temperature_c if gpu.vendor == "nvidia" else 0,
            "gpu_vram_free_mb": gpu.vram_free_mb if gpu.vendor == "nvidia" else 0,
            "gpu_vram_used_mb": (gpu.vram_total_mb - gpu.vram_free_mb) if gpu.vendor == "nvidia" else 0,
        }


def print_hardware_report():
    config = build_and_save_config()
    p = config.profile
    i = config.inference
    print("=" * 60)
    print("  AETHERAGENT — Hardware Profile Report")
    print("=" * 60)
    env = "WSL2" if p.is_wsl else "Native Linux"
    print(f"  Environment  : {env} ({p.architecture})")
    print(f"  CPU          : {p.cpu.model_name}")
    print(f"  Cores        : {p.cpu.physical_cores}P / {p.cpu.logical_cores}L")
    print(f"  SIMD         : AVX2={'Y' if p.cpu.has_avx2 else 'N'}  AVX-512={'Y' if p.cpu.has_avx512 else 'N'}")
    print(f"  RAM          : {p.ram_total_gb} GB (available: {p.ram_available_gb} GB)")
    print(f"  Swap         : {p.swap_total_gb} GB")
    print(f"  Disk Free    : {p.disk_free_gb} GB")
    if p.gpu.vendor == "nvidia":
        print(f"  GPU          : {p.gpu.model_name}")
        print(f"  VRAM         : {p.gpu.vram_total_mb} MB")
        print(f"  Driver       : {p.gpu.driver_version}")
        print(f"  CUDA         : {p.gpu.cuda_version}")
    else:
        print("  GPU          : None (CPU-only mode)")
    print("-" * 60)
    print("  INFERENCE CONFIGURATION")
    print("-" * 60)
    print(f"  Backend      : {i.backend}")
    print(f"  Quantization : {i.quantization}")
    print(f"  GPU Layers   : {'All' if i.n_gpu_layers == -1 else i.n_gpu_layers}")
    print(f"  Context Size : {i.context_size} tokens")
    print(f"  Batch Size   : {i.batch_size}")
    print(f"  Threads      : {i.threads}")
    print(f"  Max Model    : ~{i.max_model_params_gb} GB params")
    print("=" * 60)
    print(f"  Config saved : {config.config_path}")
    print("=" * 60)


if __name__ == "__main__":
    print_hardware_report()
