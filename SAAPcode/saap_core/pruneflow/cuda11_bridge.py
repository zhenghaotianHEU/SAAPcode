import ctypes
import os
from pathlib import Path
import torch


def _safe_cuda_version():
    try:
        return torch.version.cuda
    except Exception:
        return None


def _safe_cuda_available():
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _safe_device_name():
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return 'cpu'


def _native_dir():
    return Path(__file__).resolve().parent.parent / 'native'


def _native_source_path():
    return _native_dir() / 'saap_pf_cuda11_ref.cpp'


def _native_library_path():
    return _native_dir() / 'libsaap_pf_cuda11_ref.so'


def _ensure_native_cuda11_ref_library():
    src = _native_source_path()
    lib = _native_library_path()
    if not src.exists():
        return None
    if lib.exists() and lib.stat().st_mtime >= src.stat().st_mtime:
        return lib
    cmd = f'g++ -std=c++14 -shared -fPIC -O2 "{src}" -o "{lib}"'
    rc = os.system(cmd)
    if rc != 0 or not lib.exists():
        return None
    return lib


def _load_native_cuda11_ref():
    lib_path = _ensure_native_cuda11_ref_library()
    if lib_path is None:
        return None
    try:
        lib = ctypes.CDLL(str(lib_path))
        lib.saap_pf_cuda11_texture_surface_ref.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.saap_pf_cuda11_texture_surface_ref.restype = ctypes.c_char_p
        lib.saap_pf_cuda11_ref_ping.argtypes = []
        lib.saap_pf_cuda11_ref_ping.restype = ctypes.c_char_p
        return lib
    except Exception:
        return None


def _call_native_cuda11_reference(cuda11_enabled, texture_reference, surface_reference):
    lib = _load_native_cuda11_ref()
    if lib is None:
        return None
    try:
        payload = lib.saap_pf_cuda11_texture_surface_ref(
            int(bool(cuda11_enabled)),
            int(bool(texture_reference)),
            int(bool(surface_reference)),
        )
        if payload is None:
            return None
        return payload.decode('utf-8', errors='ignore')
    except Exception:
        return None


def build_cuda11_texture_surface_reference(args=None):
    cuda_version = _safe_cuda_version()
    cuda_available = _safe_cuda_available()
    runtime_device = getattr(args, 'device', None) if args is not None else None
    cuda11_enabled = bool(cuda_available and cuda_version and str(cuda_version).startswith('11'))
    texture_reference_enabled = bool(cuda11_enabled)
    surface_reference_enabled = bool(cuda11_enabled)
    native_reference = _call_native_cuda11_reference(
        cuda11_enabled=cuda11_enabled,
        texture_reference=texture_reference_enabled,
        surface_reference=surface_reference_enabled,
    )
    return {
        'cuda_version': cuda_version,
        'cuda_available': cuda_available,
        'runtime_device': runtime_device,
        'device_name': _safe_device_name(),
        'texture_reference': texture_reference_enabled,
        'surface_reference': surface_reference_enabled,
        'cuda11_runtime_bridge': bool(texture_reference_enabled or surface_reference_enabled),
        'cuda_home': os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH'),
        'native_reference': native_reference,
    }


def register_cuda11_texture_surface_reference(args=None, logger=None):
    bridge_state = build_cuda11_texture_surface_reference(args=args)
    if logger is not None:
        logger.log(
            '[cuda11_bridge] '
            f"cuda_version={bridge_state['cuda_version']} | "
            f"cuda_available={int(bool(bridge_state['cuda_available']))} | "
            f"device={bridge_state['device_name']} | "
            f"texture_reference={int(bool(bridge_state['texture_reference']))} | "
            f"surface_reference={int(bool(bridge_state['surface_reference']))} | "
            f"native_reference={bridge_state['native_reference']}"
        )
    return bridge_state
