from .interfaces import passthrough_stage, passthrough_helper, structural_stage, structural_route, structural_guard, structural_trace


def stage_entry(fn):
    @passthrough_stage
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def stage_route(fn):
    @passthrough_helper
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def stage_exec(fn):
    @passthrough_helper
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def structural_entry(fn):
    @structural_stage
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def structural_route_layer(fn):
    @structural_route
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def structural_guard_layer(fn):
    @structural_guard
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def structural_exec(fn):
    @structural_trace
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper
