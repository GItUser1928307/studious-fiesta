import os
import sys
import platform
import torch


def get_cpu_threads():
    return os.cpu_count() or 2


def get_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        c_ulong = ctypes.c_ulong

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", c_ulong),
                ("dwMemoryLoad", c_ulong),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]
        mem = MEMORYSTATUSEX()
        mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        return mem.ullTotalPhys / (1024 ** 3)
    except Exception:
        return 4.0


def get_avail_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        pass
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        c_ulong = ctypes.c_ulong

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", c_ulong),
                ("dwMemoryLoad", c_ulong),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]
        mem = MEMORYSTATUSEX()
        mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        return mem.ullAvailPhys / (1024 ** 3)
    except Exception:
        return 2.0


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_cpu_name():
    try:
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return name.strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def get_system_info():
    threads = get_cpu_threads()
    total_ram = get_ram_gb()
    avail_ram = get_avail_ram_gb()
    device = get_device()
    cpu_name = get_cpu_name()

    return {
        "cpu": cpu_name,
        "total_ram_gb": total_ram,
        "avail_ram_gb": avail_ram,
        "threads": threads,
        "device": device,
        "pytorch": torch.__version__,
        "python": platform.python_version(),
    }


def print_system_info():
    info = get_system_info()
    print(f"\nSystem Info:")
    print(f"  CPU:     {info['cpu']}")
    print(f"  RAM:     {info['avail_ram_gb']:.1f}GB available / {info['total_ram_gb']:.1f}GB total")
    print(f"  Threads: {info['threads']}")
    print(f"  Device:  {info['device'].upper()}")
    print(f"  PyTorch: {info['pytorch']}")
    return info
