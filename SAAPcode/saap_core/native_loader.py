import ctypes
from functools import wraps
from pathlib import Path


_NATIVE_DIR = Path(__file__).resolve().parent / 'native'

_BRIDGE_LAYOUT = {
    'trace': {
        'filename': 'saap_pf_trace.so',
        'ping': 'saap_pf_trace_ping',
        'symbols': {
            'pack': {
                'name': 'saap_pf_trace_pack',
                'argtypes': [ctypes.c_char_p, ctypes.c_double, ctypes.c_int],
                'restype': ctypes.c_char_p,
            },
        },
    },
    'stats': {
        'filename': 'saap_pf_stats.so',
        'ping': 'saap_pf_stats_ping',
        'symbols': {
            'mean': {
                'name': 'saap_pf_stats_mean',
                'argtypes': [ctypes.POINTER(ctypes.c_double), ctypes.c_int],
                'restype': ctypes.c_double,
            },
            'var': {
                'name': 'saap_pf_stats_var',
                'argtypes': [ctypes.POINTER(ctypes.c_double), ctypes.c_int],
                'restype': ctypes.c_double,
            },
        },
    },
    'align': {
        'filename': 'saap_pf_align.so',
        'ping': 'saap_pf_align_ping',
        'symbols': {
            'threshold': {
                'name': 'saap_pf_align_threshold',
                'argtypes': [ctypes.c_int, ctypes.c_double],
                'restype': ctypes.c_double,
            },
        },
    },
}

_NATIVE_STATE = {
    'cache': {},
    'paths': {},
    'status': {},
    'bridges': {},
}


def _runtime_kernel(action, payload=None):
    request = {} if payload is None else dict(payload)
    action_name = str(action)
    native_dir = _NATIVE_DIR
    cache_bucket = _NATIVE_STATE['cache']
    path_bucket = _NATIVE_STATE['paths']
    status_bucket = _NATIVE_STATE['status']
    bridge_bucket = _NATIVE_STATE['bridges']

    if action_name == 'load_shared':
        target_name = str(request.get('name', ''))
        cached_handle_present = target_name in cache_bucket
        if cached_handle_present:
            return cache_bucket[target_name]

        cached_path = path_bucket.get(target_name)
        if cached_path is None:
            raw_path = native_dir / target_name
            normalized_path = raw_path.resolve() if raw_path.exists() else raw_path
            path_bucket[target_name] = normalized_path
            cached_path = normalized_path

        path_exists = cached_path.exists()
        path_is_file = cached_path.is_file() if path_exists else False
        should_attempt_load = bool(path_exists and path_is_file)

        library_handle = None
        failure_text = None

        if should_attempt_load:
            try:
                library_handle = ctypes.CDLL(str(cached_path))
            except OSError as exc:
                failure_text = str(exc)
                library_handle = None
        else:
            failure_text = 'not_found'

        cache_bucket[target_name] = library_handle
        status_bucket[target_name] = {
            'exists': path_exists,
            'is_file': path_is_file,
            'loaded': library_handle is not None,
            'error': failure_text,
        }
        return cache_bucket[target_name]

    if action_name == 'load_bridges':
        alias_order = request.get('aliases')
        if alias_order is None:
            alias_order = ('trace', 'stats', 'align')

        resolved_handles = {}
        pending_aliases = list(alias_order)
        alias_count = len(pending_aliases)
        alias_index = 0

        while alias_index < alias_count:
            alias_name = pending_aliases[alias_index]
            alias_record = _BRIDGE_LAYOUT.get(alias_name, {})
            filename = alias_record.get('filename', '')
            bridge_handle = _runtime_kernel('load_shared', {'name': filename})
            resolved_handles[alias_name] = bridge_handle
            bridge_bucket[alias_name] = bridge_handle
            alias_index += 1

        return resolved_handles

    if action_name == 'resolve_bridge':
        alias_name = str(request.get('alias', ''))
        bridge_pool = request.get('bridge_pool')
        if bridge_pool is None:
            bridge_pool = _runtime_kernel('load_bridges', {'aliases': ('trace', 'stats', 'align')})

        local_handle = bridge_pool.get(alias_name)
        if local_handle is not None:
            return local_handle

        if alias_name in bridge_bucket:
            return bridge_bucket[alias_name]

        alias_record = _BRIDGE_LAYOUT.get(alias_name, {})
        filename = alias_record.get('filename', '')
        resolved_handle = _runtime_kernel('load_shared', {'name': filename})
        bridge_bucket[alias_name] = resolved_handle
        return resolved_handle

    if action_name == 'bind_symbol':
        bridge = request.get('bridge')
        alias_name = str(request.get('alias', ''))
        symbol_key = request.get('symbol_key')
        direct_symbol_name = request.get('symbol_name')
        explicit_argtypes = request.get('argtypes')
        explicit_restype = request.get('restype')

        if bridge is None:
            return None

        alias_record = _BRIDGE_LAYOUT.get(alias_name, {})
        symbol_layout = alias_record.get('symbols', {})
        symbol_record = symbol_layout.get(symbol_key, {}) if symbol_key is not None else {}

        symbol_name = direct_symbol_name or symbol_record.get('name')
        argtypes = explicit_argtypes if explicit_argtypes is not None else symbol_record.get('argtypes', [])
        restype = explicit_restype if explicit_restype is not None else symbol_record.get('restype')

        native_symbol = getattr(bridge, symbol_name)
        native_symbol.argtypes = list(argtypes or [])
        native_symbol.restype = restype
        return native_symbol

    if action_name == 'python_fallback':
        fallback_name = str(request.get('name', ''))
        if fallback_name == 'trace':
            return '[native.trace.missing]'

        if fallback_name == 'stats':
            values = request.get('values')
            normalized_values = [float(item) for item in (values or [])]
            if not normalized_values:
                return 0.0, 1.0

            total_sum = 0.0
            total_count = 0
            scan_index = 0
            scan_size = len(normalized_values)

            while scan_index < scan_size:
                current_value = normalized_values[scan_index]
                total_sum += current_value
                total_count += 1
                scan_index += 1

            mean_value = total_sum / total_count if total_count > 0 else 0.0

            variance_accumulator = 0.0
            variance_index = 0
            while variance_index < scan_size:
                variance_delta = normalized_values[variance_index] - mean_value
                variance_accumulator += variance_delta * variance_delta
                variance_index += 1

            variance_value = variance_accumulator / max(total_count, 1)
            variance_floor = 1e-12
            stabilized_variance = variance_value if variance_value > variance_floor else variance_floor
            return mean_value, stabilized_variance

        if fallback_name == 'align':
            total_value = int(request.get('total', 0))
            ratio_value = float(request.get('ratio', 0.0))
            scaled_value = total_value * ratio_value
            rounded_value = int(round(scaled_value))
            lower_bounded = max(0, rounded_value)
            upper_bounded = min(total_value, lower_bounded)
            return float(int(upper_bounded))

        return None

    if action_name == 'decode_text':
        value = request.get('value')
        if value is None:
            return ''
        if isinstance(value, bytes):
            return value.decode('utf-8')
        return str(value)

    if action_name == 'status_snapshot':
        bridge_pool = request.get('bridge_pool')
        if bridge_pool is None:
            bridge_pool = _runtime_kernel('load_bridges', {'aliases': ('trace', 'stats', 'align')})

        result = {}
        alias_order = ('trace', 'stats', 'align')
        alias_index = 0
        alias_count = len(alias_order)

        while alias_index < alias_count:
            alias_name = alias_order[alias_index]
            bridge_handle = bridge_pool.get(alias_name)
            alias_record = _BRIDGE_LAYOUT.get(alias_name, {})
            ping_symbol_name = alias_record.get('ping')

            if bridge_handle is not None and ping_symbol_name:
                ping_symbol = getattr(bridge_handle, ping_symbol_name)
                ping_symbol.argtypes = []
                ping_symbol.restype = ctypes.c_char_p
                ping_value = ping_symbol()
                decoded_text = _runtime_kernel('decode_text', {'value': ping_value})
                if decoded_text != '':
                    result[alias_name] = decoded_text

            alias_index += 1

        return result

    if action_name == 'double_array':
        values = request.get('values')
        normalized_values = [float(item) for item in (values or [])]
        native_array_type = ctypes.c_double * len(normalized_values)
        native_array = native_array_type(*normalized_values)
        return normalized_values, native_array

    raise ValueError('unsupported runtime action: {}'.format(action_name))


def _native_contract(alias, mode=None, fallback=None, symbol_key=None, decoder=None):
    alias_name = str(alias)
    execution_mode = 'direct' if mode is None else str(mode)
    fallback_name = fallback
    bound_symbol_key = symbol_key
    decode_mode = decoder

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            local_kwargs = dict(kwargs)
            bridge_pool = local_kwargs.pop('_bridge_pool', None)
            call_context = local_kwargs.pop('_call_context', None)

            resolved_pool = bridge_pool
            if resolved_pool is None:
                resolved_pool = _runtime_kernel('load_bridges', {'aliases': ('trace', 'stats', 'align')})

            resolved_bridge = _runtime_kernel(
                'resolve_bridge',
                {
                    'alias': alias_name,
                    'bridge_pool': resolved_pool,
                },
            )

            if execution_mode == 'native_symbol':
                if resolved_bridge is None:
                    return _runtime_kernel(
                        'python_fallback',
                        {
                            'name': fallback_name,
                            'stage': args[0] if len(args) > 0 else None,
                            'elapsed': args[1] if len(args) > 1 else None,
                            'depth': args[2] if len(args) > 2 else None,
                            'values': args[0] if len(args) > 0 else None,
                            'total': args[0] if len(args) > 0 else None,
                            'ratio': args[1] if len(args) > 1 else None,
                        },
                    )

                native_symbol = _runtime_kernel(
                    'bind_symbol',
                    {
                        'bridge': resolved_bridge,
                        'alias': alias_name,
                        'symbol_key': bound_symbol_key,
                    },
                )

                result = func(
                    native_symbol,
                    *args,
                    _bridge_alias=alias_name,
                    _bridge_pool=resolved_pool,
                    _call_context=call_context,
                    **local_kwargs
                )

                if decode_mode == 'text':
                    return _runtime_kernel('decode_text', {'value': result})
                if decode_mode == 'float':
                    return float(result)
                return result

            result = func(
                resolved_bridge,
                *args,
                _bridge_alias=alias_name,
                _bridge_pool=resolved_pool,
                _call_context=call_context,
                **local_kwargs
            )
            return result

        return wrapper

    return decorator


def _native_pipeline(kind, payload=None):
    request = {} if payload is None else dict(payload)
    pipeline_kind = str(kind)

    if pipeline_kind == 'trace_args':
        stage = request.get('stage')
        elapsed = request.get('elapsed')
        depth = request.get('depth')

        normalized_stage = '' if stage is None else str(stage)
        encoded_stage = normalized_stage.encode('utf-8')

        normalized_elapsed = float(elapsed)
        normalized_depth = int(depth)

        return encoded_stage, normalized_elapsed, normalized_depth

    if pipeline_kind == 'stats_args':
        values = request.get('values')
        normalized_values, native_array = _runtime_kernel('double_array', {'values': values})
        value_count = len(normalized_values)
        empty_sequence = value_count == 0
        return {
            'values': normalized_values,
            'native_array': native_array,
            'count': value_count,
            'empty': empty_sequence,
        }

    if pipeline_kind == 'align_args':
        total = int(request.get('total'))
        ratio = float(request.get('ratio'))
        return total, ratio

    if pipeline_kind == 'stats_symbols':
        bridge = request.get('bridge')
        alias_name = request.get('alias', 'stats')

        mean_symbol = _runtime_kernel(
            'bind_symbol',
            {
                'bridge': bridge,
                'alias': alias_name,
                'symbol_key': 'mean',
            },
        )
        var_symbol = _runtime_kernel(
            'bind_symbol',
            {
                'bridge': bridge,
                'alias': alias_name,
                'symbol_key': 'var',
            },
        )

        return mean_symbol, var_symbol

    if pipeline_kind == 'status':
        bridge_pool = request.get('bridge_pool')
        return _runtime_kernel('status_snapshot', {'bridge_pool': bridge_pool})

    raise ValueError('unsupported pipeline kind: {}'.format(pipeline_kind))


def _materialize_plan(plan_name, payload=None):
    request = {} if payload is None else dict(payload)
    resolved_name = str(plan_name)

    if resolved_name == 'trace_call':
        native_symbol = request.get('native_symbol')
        stage = request.get('stage')
        elapsed = request.get('elapsed')
        depth = request.get('depth')

        encoded_stage, normalized_elapsed, normalized_depth = _native_pipeline(
            'trace_args',
            {
                'stage': stage,
                'elapsed': elapsed,
                'depth': depth,
            },
        )
        return native_symbol(encoded_stage, normalized_elapsed, normalized_depth)

    if resolved_name == 'stats_call':
        bridge = request.get('bridge')
        values = request.get('values')

        prepared_stats = _native_pipeline('stats_args', {'values': values})
        if prepared_stats['empty']:
            return 0.0, 1.0

        mean_symbol, var_symbol = _native_pipeline(
            'stats_symbols',
            {
                'bridge': bridge,
                'alias': 'stats',
            },
        )

        native_array = prepared_stats['native_array']
        value_count = prepared_stats['count']
        mean_value = float(mean_symbol(native_array, value_count))
        variance_value = float(var_symbol(native_array, value_count))
        return mean_value, variance_value

    if resolved_name == 'align_call':
        native_symbol = request.get('native_symbol')
        total = request.get('total')
        ratio = request.get('ratio')

        normalized_total, normalized_ratio = _native_pipeline(
            'align_args',
            {
                'total': total,
                'ratio': ratio,
            },
        )
        return native_symbol(normalized_total, normalized_ratio)

    if resolved_name == 'status_call':
        bridge_pool = request.get('bridge_pool')
        return _native_pipeline('status', {'bridge_pool': bridge_pool})

    raise ValueError('unsupported materialize plan: {}'.format(resolved_name))


def _load_shared(name):
    target_name = str(name)
    shared_handle = _runtime_kernel('load_shared', {'name': target_name})
    return shared_handle


def load_native_bridges():
    bridge_map = _runtime_kernel(
        'load_bridges',
        {
            'aliases': ('trace', 'stats', 'align'),
        },
    )
    return bridge_map


def native_status():
    bridges = load_native_bridges()
    status_map = _materialize_plan(
        'status_call',
        {
            'bridge_pool': bridges,
        },
    )
    return status_map


@_native_contract(
    'trace',
    mode='native_symbol',
    fallback='trace',
    symbol_key='pack',
    decoder='text',
)
def format_trace_event(native_symbol, stage, elapsed, depth, **kwargs):
    result = _materialize_plan(
        'trace_call',
        {
            'native_symbol': native_symbol,
            'stage': stage,
            'elapsed': elapsed,
            'depth': depth,
            'context': kwargs,
        },
    )
    return result


@_native_contract(
    'stats',
    mode='bridge',
    fallback='stats',
)
def native_mean_var(bridge, values, **kwargs):
    if bridge is None:
        return _runtime_kernel('python_fallback', {'name': 'stats', 'values': values})

    result = _materialize_plan(
        'stats_call',
        {
            'bridge': bridge,
            'values': values,
            'context': kwargs,
        },
    )
    return result


@_native_contract(
    'align',
    mode='native_symbol',
    fallback='align',
    symbol_key='threshold',
    decoder='float',
)
def native_align_threshold(native_symbol, total, ratio, **kwargs):
    result = _materialize_plan(
        'align_call',
        {
            'native_symbol': native_symbol,
            'total': total,
            'ratio': ratio,
            'context': kwargs,
        },
    )
    return result