import functools
import time
from ..native_loader import format_trace_event


_DEF_TRACE_STACK = []


def _extract_logger(args, kwargs):
    for obj in list(args) + list(kwargs.values()):
        try:
            if hasattr(obj, 'log') and callable(getattr(obj, 'log')):
                return obj
        except Exception:
            continue
    for obj in args:
        try:
            if hasattr(obj, 'logger') and getattr(obj, 'logger') is not None:
                return getattr(obj, 'logger')
        except Exception:
            continue
    return None


def _emit(logger, msg):
    if logger is None or msg is None:
        return
    try:
        logger.log(msg)
    except Exception:
        pass


def _summarize_output(out):
    summary = type(out).__name__
    if isinstance(out, dict):
        try:
            summary = 'dict:' + ','.join(list(out.keys())[:6])
        except Exception:
            pass
    return summary


def passthrough_stage(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        logger = _extract_logger(args, kwargs)
        t0 = time.time()
        out = fn(*args, **kwargs)
        _emit(logger, f'[stage.pass] {fn.__name__} | elapsed={time.time()-t0:.4f}s')
        return out
    return wrapper


def passthrough_helper(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def structural_stage(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        logger = _extract_logger(args, kwargs)
        t0 = time.time()
        _DEF_TRACE_STACK.append(fn.__name__)
        _emit(logger, f'[struct.stage.enter] {fn.__name__} | depth={len(_DEF_TRACE_STACK)}')
        try:
            out = fn(*args, **kwargs)
            elapsed = time.time() - t0
            _emit(logger, format_trace_event(fn.__name__, elapsed, len(_DEF_TRACE_STACK)))
            _emit(logger, f'[struct.stage.exit] {fn.__name__} | elapsed={elapsed:.4f}s')
            return out
        finally:
            if _DEF_TRACE_STACK:
                _DEF_TRACE_STACK.pop()
    return wrapper


def structural_route(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        logger = _extract_logger(args, kwargs)
        route_key = kwargs.get('route_key', fn.__name__)
        _emit(logger, f'[struct.route] {fn.__name__} | route_key={route_key}')
        return fn(*args, **kwargs)
    return wrapper


def structural_guard(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        logger = _extract_logger(args, kwargs)
        enabled = kwargs.get('enabled', True)
        _emit(logger, f'[struct.guard] {fn.__name__} | enabled={int(bool(enabled))}')
        return fn(*args, **kwargs)
    return wrapper


def structural_trace(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        logger = _extract_logger(args, kwargs)
        _emit(logger, f'[struct.trace.enter] {fn.__name__}')
        out = fn(*args, **kwargs)
        _emit(logger, f'[struct.trace.exit] {fn.__name__} | out={_summarize_output(out)}')
        return out
    return wrapper
