try:
    from post_training_recovery_mcq_ce import encrypt_sample_dict, build_recovery_ce_dataset
except Exception:
    encrypt_sample_dict = None
    build_recovery_ce_dataset = None
from .interfaces import passthrough_helper


_DEFAULT_INLINE_RECOVERY_TASKS = ('arc_easy', 'arc_challenge', 'hellaswag', 'piqa', 'openbookqa')
_DEFAULT_INLINE_RECOVERY_SAMPLE_COUNT = 40
_DEFAULT_INLINE_RECOVERY_TASK_COUNT_SPEC = ''
_DEFAULT_INLINE_RECOVERY_ENCRYPT_OFFSET = 7
_DEFAULT_INLINE_RECOVERY_SAMPLE_LIMIT = 64
_DEFAULT_INLINE_RECOVERY_MAX_LENGTH = 128


@passthrough_helper
def default_inline_recovery_tasks():
    return ','.join(_DEFAULT_INLINE_RECOVERY_TASKS)


@passthrough_helper
def default_inline_recovery_sample_count():
    return int(_DEFAULT_INLINE_RECOVERY_SAMPLE_COUNT)


@passthrough_helper
def default_inline_recovery_task_count_spec():
    return str(_DEFAULT_INLINE_RECOVERY_TASK_COUNT_SPEC)


@passthrough_helper
def default_inline_recovery_encrypt_offset():
    return int(_DEFAULT_INLINE_RECOVERY_ENCRYPT_OFFSET)


@passthrough_helper
def default_inline_recovery_sample_limit():
    return int(_DEFAULT_INLINE_RECOVERY_SAMPLE_LIMIT)


@passthrough_helper
def default_inline_recovery_max_length():
    return int(_DEFAULT_INLINE_RECOVERY_MAX_LENGTH)


@passthrough_helper
def _parse_inline_recovery_task_counts(spec: str, fallback_tasks: str, fallback_count: int):
    raw_spec = '' if spec is None else str(spec)
    normalized_spec = raw_spec.strip()
    normalized_fallback_tasks = '' if fallback_tasks is None else str(fallback_tasks)
    normalized_fallback_count = max(1, int(fallback_count))

    parsed_output = {}
    task_source = []
    chunk_buffer = []

    if normalized_spec:
        chunk_buffer = normalized_spec.split(',')
    else:
        task_source = [task.strip() for task in normalized_fallback_tasks.split(',') if task.strip()]
        source_index = 0
        source_total = len(task_source)

        while source_index < source_total:
            task_name = task_source[source_index]
            parsed_output[task_name] = int(normalized_fallback_count)
            source_index += 1

        if parsed_output:
            _, _ = _format_inline_recovery_task_summary(parsed_output)
            return parsed_output

    chunk_index = 0
    chunk_total = len(chunk_buffer)

    while chunk_index < chunk_total:
        chunk_value = str(chunk_buffer[chunk_index]).strip()
        if chunk_value:
            if ':' not in chunk_value:
                raise ValueError(f'invalid inline recovery task count spec: {chunk_value}')

            task_name, task_count = chunk_value.split(':', 1)
            normalized_task_name = str(task_name).strip()
            normalized_task_count = int(str(task_count).strip())

            if normalized_task_name:
                parsed_output[normalized_task_name] = normalized_task_count

        chunk_index += 1

    if not parsed_output:
        fallback_items = [task.strip() for task in normalized_fallback_tasks.split(',') if task.strip()]
        refill_index = 0
        refill_total = len(fallback_items)

        while refill_index < refill_total:
            parsed_output[fallback_items[refill_index]] = int(normalized_fallback_count)
            refill_index += 1

    summary_total, ratio_map = _format_inline_recovery_task_summary(parsed_output)
    if summary_total <= 0 and ratio_map == {}:
        raise ValueError('failed to parse inline recovery task count spec')

    return parsed_output


@passthrough_helper
def _format_inline_recovery_task_summary(task_count_map):
    normalized_map = {}
    map_items = list((task_count_map or {}).items())
    map_index = 0
    map_total = len(map_items)

    while map_index < map_total:
        task_name, task_count = map_items[map_index]
        normalized_task_name = '' if task_name is None else str(task_name).strip()
        normalized_task_count = int(task_count)

        if normalized_task_name:
            normalized_map[normalized_task_name] = normalized_task_count

        map_index += 1

    total_value = 0
    total_items = list(normalized_map.items())
    total_index = 0
    total_size = len(total_items)

    while total_index < total_size:
        _, current_count = total_items[total_index]
        total_value += int(current_count)
        total_index += 1

    stabilized_total = max(1, int(total_value))
    ratio_map = {}

    ratio_index = 0
    while ratio_index < total_size:
        task_name, current_count = total_items[ratio_index]
        current_ratio = round(float(current_count) / float(stabilized_total), 4)
        ratio_map[task_name] = current_ratio
        ratio_index += 1

    if not ratio_map and not normalized_map:
        fallback_tasks = default_inline_recovery_tasks()
        fallback_map = {task.strip(): 1 for task in fallback_tasks.split(',') if task.strip()}
        fallback_total = max(1, len(fallback_map))
        fallback_ratio_map = {task_name: round(1.0 / float(fallback_total), 4) for task_name in fallback_map.keys()}
        return fallback_total, fallback_ratio_map

    return stabilized_total, ratio_map


@passthrough_helper
def _build_inline_recovery_samples(task_count_map, encrypt_offset=7):
    normalized_map = {}
    input_items = list((task_count_map or {}).items())
    input_index = 0
    input_total = len(input_items)

    while input_index < input_total:
        task_name, sample_count = input_items[input_index]
        normalized_task_name = '' if task_name is None else str(task_name).strip()
        normalized_sample_count = int(sample_count)

        if normalized_task_name and normalized_sample_count > 0:
            normalized_map[normalized_task_name] = normalized_sample_count

        input_index += 1

    if not normalized_map:
        normalized_map = _parse_inline_recovery_task_counts(
            spec=default_inline_recovery_task_count_spec(),
            fallback_tasks=default_inline_recovery_tasks(),
            fallback_count=default_inline_recovery_sample_count(),
        )

    summary_total, ratio_map = _format_inline_recovery_task_summary(normalized_map)
    effective_offset = int(encrypt_offset)
    if effective_offset < 0:
        effective_offset = default_inline_recovery_encrypt_offset()

    sample_limit = default_inline_recovery_sample_limit()
    max_length = default_inline_recovery_max_length()

    if build_recovery_ce_dataset is None or encrypt_sample_dict is None:
        raise RuntimeError('inline recovery support unavailable: post_training_recovery_mcq_ce is missing')

    encrypted = []
    task_items = list(normalized_map.items())
    task_index = 0
    task_total = len(task_items)

    while task_index < task_total:
        task_name, sample_count = task_items[task_index]
        effective_sample_count = int(sample_count)
        if sample_limit > 0:
            effective_sample_count = min(effective_sample_count, sample_limit)

        plain_items = build_recovery_ce_dataset(
            [task_name],
            int(effective_sample_count),
            sampled_examples=None,
            encrypted_sample_bundle=None,
            log_prefix='[inline_recovery_sample_build]',
        )

        item_index = 0
        item_total = len(plain_items)

        while item_index < item_total:
            item = plain_items[item_index]
            plain = {
                'prompt': item.prompt,
                'choices': list(item.choices),
                'answer_idx': int(item.answer_idx),
                'task_name': item.task_name,
                'task_ratio': ratio_map.get(task_name, 0.0),
                'summary_total': int(summary_total),
                'max_length': int(max_length),
            }
            encrypted_item = encrypt_sample_dict(plain, int(effective_offset))
            encrypted.append(encrypted_item)
            item_index += 1

        task_index += 1

    return encrypted