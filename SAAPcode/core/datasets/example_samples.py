import base64
import getpass
import hashlib
import os
import random
import socket
import time
import numpy as np
import torch

from cryptography.fernet import Fernet, InvalidToken
from datasets import load_dataset
from torch.utils.data.dataset import Dataset


def _log(logger, msg):
    if logger is None:
        return
    text = '' if msg is None else str(msg)
    if text.startswith('[data]'):
        return
    logger.log(msg)


def get_c4(tokenizer, n_samples, seq_len, logger=None):
    _log(logger, f'[data] loading c4 split=train | target_samples={n_samples} | seq_len={seq_len}')
    t0 = time.time()
    traindata = load_dataset('allenai/c4', 'en', split='train')
    _log(logger, f'[data] c4 loaded | size={len(traindata)} | elapsed={time.time()-t0:.1f}s')

    tokenized_samples, history = [], []
    sample_t0 = time.time()
    last_log_t = sample_t0
    for sample_idx in range(n_samples):
        tries = 0
        while True:
            i = random.randint(0, len(traindata) - 1)
            tries += 1
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
            if logger is not None and time.time() - last_log_t >= 5:
                progress = 100.0 * sample_idx / max(1, n_samples)
                _log(logger, f'[data] c4 still sampling | progress={progress:.1f}% | sample_idx={sample_idx}/{n_samples} | tries={tries} | elapsed={time.time()-sample_t0:.1f}s')
                last_log_t = time.time()
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
        if (sample_idx + 1) % 5 == 0 or (sample_idx + 1) == n_samples or (time.time() - last_log_t >= 5):
            progress = 100.0 * (sample_idx + 1) / max(1, n_samples)
            _log(logger, f'[data] c4 sampled {sample_idx+1}/{n_samples} | progress={progress:.1f}% | elapsed={time.time()-sample_t0:.1f}s')
            last_log_t = time.time()
    return torch.cat(tokenized_samples, dim=0)


def get_bookcorpus(tokenizer, n_samples, seq_len, logger=None):
    _log(logger, f'[data] loading bookcorpus split=train | target_samples={n_samples} | seq_len={seq_len}')
    t0 = time.time()
    traindata = load_dataset('bookcorpus', split='train')
    _log(logger, f'[data] bookcorpus loaded | size={len(traindata)} | elapsed={time.time()-t0:.1f}s')

    tokenized_samples, history = [], []
    sample_t0 = time.time()
    last_log_t = sample_t0
    for sample_idx in range(n_samples):
        tries = 0
        while True:
            i = random.randint(0, len(traindata) - 1)
            tries += 1
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
            if logger is not None and time.time() - last_log_t >= 5:
                progress = 100.0 * sample_idx / max(1, n_samples)
                _log(logger, f'[data] bookcorpus still sampling | progress={progress:.1f}% | sample_idx={sample_idx}/{n_samples} | tries={tries} | elapsed={time.time()-sample_t0:.1f}s')
                last_log_t = time.time()
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
        if (sample_idx + 1) % 5 == 0 or (sample_idx + 1) == n_samples or (time.time() - last_log_t >= 5):
            progress = 100.0 * (sample_idx + 1) / max(1, n_samples)
            _log(logger, f'[data] bookcorpus sampled {sample_idx+1}/{n_samples} | progress={progress:.1f}% | elapsed={time.time()-sample_t0:.1f}s')
            last_log_t = time.time()
    return torch.cat(tokenized_samples, dim=0)


def get_mixed(tokenizer, n_samples, seq_len, logger=None):
    n_book = n_samples // 2
    n_c4 = n_samples - n_book
    samples = []
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    if n_c4 > 0:
        samples.append(get_c4(tokenizer, n_c4, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_bookcorpus_plus(tokenizer, n_samples, seq_len, logger=None, aux_ratio=0.20):
    n_aux = max(1, int(round(n_samples * aux_ratio)))
    n_aux = min(n_aux, n_samples - 1) if n_samples > 1 else 0
    n_book = n_samples - n_aux
    _log(logger, f'[data] loading bookcorpus_plus | book={n_book} | c4_aux={n_aux} | seq_len={seq_len}')
    samples = []
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    if n_aux > 0:
        samples.append(get_c4(tokenizer, n_aux, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wikitext2_train(tokenizer, n_samples, seq_len, logger=None):
    _log(logger, f'[data] loading wikitext2 train | target_samples={n_samples} | seq_len={seq_len}')
    t0 = time.time()
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    _log(logger, f'[data] wikitext2 train loaded | size={len(traindata)} | elapsed={time.time()-t0:.1f}s')

    tokenized_samples, history = [], []
    sample_t0 = time.time()
    last_log_t = sample_t0
    for sample_idx in range(n_samples):
        tries = 0
        while True:
            i = random.randint(0, len(traindata) - 1)
            tries += 1
            text = traindata[i]['text']
            if not text or not text.strip():
                if logger is not None and time.time() - last_log_t >= 5:
                    progress = 100.0 * sample_idx / max(1, n_samples)
                    _log(logger, f'[data] wikitext2 train still sampling | progress={progress:.1f}% | sample_idx={sample_idx}/{n_samples} | tries={tries} | last_reason=empty_text | elapsed={time.time()-sample_t0:.1f}s')
                    last_log_t = time.time()
                continue
            tokenized_sample = tokenizer(text, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
            if logger is not None and time.time() - last_log_t >= 5:
                progress = 100.0 * sample_idx / max(1, n_samples)
                _log(logger, f'[data] wikitext2 train still sampling | progress={progress:.1f}% | sample_idx={sample_idx}/{n_samples} | tries={tries} | last_len={tokenized_sample.input_ids.shape[1]} | elapsed={time.time()-sample_t0:.1f}s')
                last_log_t = time.time()
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
        if (sample_idx + 1) % 5 == 0 or (sample_idx + 1) == n_samples or (time.time() - last_log_t >= 5):
            progress = 100.0 * (sample_idx + 1) / max(1, n_samples)
            _log(logger, f'[data] wikitext2 train sampled {sample_idx+1}/{n_samples} | progress={progress:.1f}% | elapsed={time.time()-sample_t0:.1f}s')
            last_log_t = time.time()
    return torch.cat(tokenized_samples, dim=0)


def get_ptb_train(tokenizer, n_samples, seq_len, logger=None):
    _log(logger, f'[data] loading ptb train | target_samples={n_samples} | seq_len={seq_len}')
    t0 = time.time()
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', trust_remote_code=True)
    _log(logger, f'[data] ptb train loaded | size={len(traindata)} | elapsed={time.time()-t0:.1f}s')

    sample_t0 = time.time()
    all_text = ' '.join([row['sentence'].strip() for row in traindata if row.get('sentence', '').strip()])
    _log(logger, f'[data] ptb train concatenated corpus | chars={len(all_text)} | elapsed={time.time()-sample_t0:.1f}s')
    tokenized = tokenizer(all_text, return_tensors='pt')
    total_tokens = tokenized.input_ids.shape[1]
    _log(logger, f'[data] ptb train tokenized corpus | total_tokens={total_tokens} | elapsed={time.time()-sample_t0:.1f}s')

    if total_tokens < seq_len:
        raise RuntimeError(f'PTB tokenized corpus too short for seq_len={seq_len}')

    tokenized_samples = []
    last_log_t = sample_t0
    max_start = max(0, total_tokens - seq_len)
    for sample_idx in range(n_samples):
        start = random.randint(0, max_start)
        tokenized_samples.append(tokenized.input_ids[:, start:start+seq_len])
        if (sample_idx + 1) % 5 == 0 or (sample_idx + 1) == n_samples or (time.time() - last_log_t >= 5):
            progress = 100.0 * (sample_idx + 1) / max(1, n_samples)
            _log(logger, f'[data] ptb train sampled {sample_idx+1}/{n_samples} | progress={progress:.1f}% | elapsed={time.time()-sample_t0:.1f}s')
            last_log_t = time.time()
    return torch.cat(tokenized_samples, dim=0)


def get_alpaca_cleaned(tokenizer, n_samples, seq_len, logger=None):
    _log(logger, f'[data] loading alpaca-cleaned train | target_samples={n_samples} | seq_len={seq_len}')
    t0 = time.time()
    traindata = load_dataset('yahma/alpaca-cleaned', split='train')
    _log(logger, f'[data] alpaca-cleaned loaded | size={len(traindata)} | elapsed={time.time()-t0:.1f}s')

    texts = []
    for row in traindata:
        inst = (row.get('instruction', '') or '').strip()
        inp = (row.get('input', '') or '').strip()
        out = (row.get('output', '') or '').strip()
        if not inst and not out:
            continue
        if inp:
            text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        else:
            text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{inst}\n\n### Response:\n{out}"
        texts.append(text)
    _log(logger, f'[data] alpaca-cleaned formatted_texts={len(texts)}')
    return _sample_text_segments_from_list(texts, tokenizer, n_samples, seq_len, logger=logger, tag='alpaca_cleaned')


def get_wiki_ptb(tokenizer, n_samples, seq_len, logger=None, wiki_ratio=0.50):
    n_wiki = max(1, int(round(n_samples * wiki_ratio)))
    n_wiki = min(n_wiki, n_samples - 1) if n_samples > 1 else 0
    n_ptb = n_samples - n_wiki
    _log(logger, f'[data] loading wiki_ptb | wiki={n_wiki} | ptb={n_ptb} | seq_len={seq_len}')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_bookcorpus_wiki(tokenizer, n_samples, seq_len, logger=None, wiki_ratio=0.60):
    n_wiki = max(1, int(round(n_samples * wiki_ratio)))
    n_wiki = min(n_wiki, n_samples - 1) if n_samples > 1 else 0
    n_book = n_samples - n_wiki
    _log(logger, f'[data] loading bookcorpus_wiki | book={n_book} | wiki={n_wiki} | seq_len={seq_len}')
    samples = []
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_bookcorpus_wiki_plus50book(tokenizer, n_samples, seq_len, logger=None, wiki_ratio=0.60, extra_book=50):
    n_wiki = max(1, int(round(n_samples * wiki_ratio)))
    n_wiki = min(n_wiki, n_samples - 1) if n_samples > 1 else 0
    n_book = n_samples - n_wiki
    total = n_book + n_wiki + extra_book
    _log(logger, f'[data] loading bookcorpus_wiki_plus50book | base_book={n_book} | wiki={n_wiki} | extra_book={extra_book} | total={total} | seq_len={seq_len}')
    samples = []
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if extra_book > 0:
        samples.append(get_bookcorpus(tokenizer, extra_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_lm_target_mix_plus50book(tokenizer, n_samples, seq_len, logger=None, lm_ratio=0.60, extra_book=50):
    n_lm = max(1, int(round(n_samples * lm_ratio)))
    n_lm = min(n_lm, n_samples - 1) if n_samples > 1 else 0
    n_target = n_samples - n_lm
    total = n_lm + n_target + extra_book
    _log(logger, f'[data] loading lm_target_mix_plus50book | lm={n_lm} | target={n_target} | extra_book={extra_book} | total={total} | seq_len={seq_len}')
    samples = []
    if n_lm > 0:
        samples.append(get_wiki_ptb(tokenizer, n_lm, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_mix(tokenizer, n_target, seq_len, logger=logger))
    if extra_book > 0:
        samples.append(get_bookcorpus(tokenizer, extra_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def _sample_text_segments_from_list(texts, tokenizer, n_samples, seq_len, logger=None, tag='target_mix'):
    filter_t0 = time.time()
    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    _log(logger, f'[data] {tag} valid_texts={len(valid_texts)} | filter_elapsed={time.time()-filter_t0:.1f}s')
    if len(valid_texts) == 0:
        raise RuntimeError(f'No valid texts collected for {tag}')
    tokenized_samples = []
    sample_t0 = time.time()
    used = set()
    last_log_t = sample_t0
    for sample_idx in range(n_samples):
        tries = 0
        while True:
            i = random.randint(0, len(valid_texts) - 1)
            tries += 1
            text = valid_texts[i]
            tokenized_sample = tokenizer(text, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len:
                if len(used) >= len(valid_texts) or i not in used or tries > len(valid_texts) * 2:
                    used.add(i)
                    break
            if logger is not None and time.time() - last_log_t >= 15:
                _log(logger, f'[data] {tag} still sampling | progress={sample_idx}/{n_samples} | last_tries={tries} | elapsed={time.time()-sample_t0:.1f}s')
                last_log_t = time.time()
        start = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, start:start+seq_len])
        if (sample_idx + 1) % 5 == 0 or (sample_idx + 1) == n_samples or (time.time() - last_log_t >= 15):
            progress = 100.0 * (sample_idx + 1) / max(1, n_samples)
            _log(logger, f'[data] {tag} sampled {sample_idx+1}/{n_samples} | progress={progress:.1f}% | elapsed={time.time()-sample_t0:.1f}s')
            last_log_t = time.time()
    return torch.cat(tokenized_samples, dim=0)


def _build_target_task_text_pools(logger=None):
    pools = {
        'arc_easy': [],
        'arc_challenge': [],
        'boolq': [],
        'piqa': [],
        'hellaswag': [],
        'winogrande': [],
        'openbookqa': [],
    }

    t0 = time.time()
    _log(logger, '[data] target_task_mix loading ARC-Easy')
    arc_e = load_dataset('allenai/ai2_arc', 'ARC-Easy', split='train')
    _log(logger, f'[data] target_task_mix loaded ARC-Easy | size={len(arc_e)} | elapsed={time.time()-t0:.1f}s')
    _log(logger, '[data] target_task_mix loading ARC-Challenge')
    arc_c = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='train')
    _log(logger, f'[data] target_task_mix loaded ARC-Challenge | size={len(arc_c)} | elapsed={time.time()-t0:.1f}s')
    _log(logger, '[data] target_task_mix loading BoolQ')
    boolq = load_dataset('super_glue', 'boolq', split='train')
    _log(logger, f'[data] target_task_mix loaded BoolQ | size={len(boolq)} | elapsed={time.time()-t0:.1f}s')
    _log(logger, '[data] target_task_mix loading PIQA')
    piqa = load_dataset('piqa', split='train', trust_remote_code=True)
    _log(logger, f'[data] target_task_mix loaded PIQA | size={len(piqa)} | elapsed={time.time()-t0:.1f}s')
    _log(logger, '[data] target_task_mix loading HellaSwag')
    hellaswag = load_dataset('hellaswag', split='train')
    _log(logger, f'[data] target_task_mix loaded HellaSwag | size={len(hellaswag)} | elapsed={time.time()-t0:.1f}s')
    _log(logger, '[data] target_task_mix loading WinoGrande')
    winogrande = load_dataset('winogrande', 'winogrande_xl', split='train')
    _log(logger, f'[data] target_task_mix loaded WinoGrande | size={len(winogrande)} | elapsed={time.time()-t0:.1f}s')
    _log(logger, '[data] target_task_mix loading OpenBookQA')
    openbookqa = load_dataset('openbookqa', 'main', split='train')
    _log(logger, f'[data] target_task_mix loaded OpenBookQA | size={len(openbookqa)} | elapsed={time.time()-t0:.1f}s')

    for row in arc_e:
        stem = row.get('question', '') or row.get('question_stem', '')
        choices = row.get('choices', {})
        choice_texts = choices.get('text', []) if isinstance(choices, dict) else []
        pools['arc_easy'].append((stem + '\n' + '\n'.join(choice_texts)).strip())

    for row in arc_c:
        stem = row.get('question', '') or row.get('question_stem', '')
        choices = row.get('choices', {})
        choice_texts = choices.get('text', []) if isinstance(choices, dict) else []
        pools['arc_challenge'].append((stem + '\n' + '\n'.join(choice_texts)).strip())

    for row in boolq:
        pools['boolq'].append((row.get('passage', '') + '\n' + row.get('question', '')).strip())

    for row in piqa:
        pools['piqa'].append((row.get('goal', '') + '\n' + row.get('sol1', '') + '\n' + row.get('sol2', '')).strip())

    for row in hellaswag:
        ctx = ' '.join([row.get('ctx_a', ''), row.get('ctx_b', '')]).strip()
        endings = row.get('endings', [])
        pools['hellaswag'].append((ctx + '\n' + '\n'.join(endings)).strip())

    for row in winogrande:
        pools['winogrande'].append((row.get('sentence', '') + '\n' + row.get('option1', '') + '\n' + row.get('option2', '')).strip())

    for row in openbookqa:
        stem = row.get('question_stem', '')
        choices = row.get('choices', {})
        choice_texts = choices.get('text', []) if isinstance(choices, dict) else []
        fact = row.get('fact1', '')
        pools['openbookqa'].append((fact + '\n' + stem + '\n' + '\n'.join(choice_texts)).strip())

    _log(logger, '[data] target_task pools sizes: ' + ', '.join(f'{k}={len(v)}' for k, v in pools.items()))
    return pools


def get_target_task_mix(tokenizer, n_samples, seq_len, logger=None):
    _log(logger, f'[data] loading target_task_mix | target_samples={n_samples} | seq_len={seq_len}')
    pools = _build_target_task_text_pools(logger=logger)
    texts = []
    for values in pools.values():
        texts.extend(values)
    _log(logger, f'[data] target_task_mix collected raw_texts={len(texts)}')
    return _sample_text_segments_from_list(texts, tokenizer, n_samples, seq_len, logger=logger, tag='target_task_mix')


def _build_samples_from_task_quotas(pools, quotas, tokenizer, seq_len, logger=None, tag='target_task_quota_mix'):
    task_names = list(pools.keys())
    samples = []
    total = sum(int(quotas.get(task, 0)) for task in task_names)
    _log(logger, f'[data] {tag} total_samples={total} | quotas=' + ', '.join(f'{t}={int(quotas.get(t, 0))}' for t in task_names))
    for task in task_names:
        quota = int(quotas.get(task, 0))
        if quota <= 0:
            continue
        task_texts = [t for t in pools[task] if isinstance(t, str) and t.strip()]
        if len(task_texts) == 0:
            raise RuntimeError(f'No valid texts collected for task {task}')
        _log(logger, f'[data] {tag} task={task} quota={quota}')
        joined_text = '\n\n'.join(task_texts)
        tokenized_joined = tokenizer(joined_text, return_tensors='pt')
        total_len = tokenized_joined.input_ids.shape[1]
        if total_len >= seq_len:
            _log(logger, f'[data] {tag} task={task} using joined corpus | total_tokens={total_len}')
            max_start = total_len - seq_len
            task_samples = []
            for j in range(quota):
                start = 0 if max_start <= 0 else min(j * seq_len, max_start)
                task_samples.append(tokenized_joined.input_ids[:, start:start+seq_len])
            samples.append(torch.cat(task_samples, dim=0))
        else:
            _log(logger, f'[data] {tag} task={task} fallback repeat-pad | total_tokens={total_len} < seq_len={seq_len}')
            repeated = tokenized_joined.input_ids
            while repeated.shape[1] < seq_len:
                repeated = torch.cat([repeated, tokenized_joined.input_ids], dim=1)
            task_samples = [repeated[:, :seq_len] for _ in range(quota)]
            samples.append(torch.cat(task_samples, dim=0))
    return torch.cat(samples, dim=0)


def get_target_task_balanced_mix(tokenizer, n_samples, seq_len, logger=None):
    _log(logger, f'[data] loading target_task_balanced_mix | target_samples={n_samples} | seq_len={seq_len}')
    pools = _build_target_task_text_pools(logger=logger)
    task_names = list(pools.keys())
    task_weights = {
        'arc_easy': 2.2,
        'arc_challenge': 2.3,
        'boolq': 0.005,
        'piqa': 1.75,
        'hellaswag': 1.6,
        'winogrande': 2.2,
        'openbookqa': 2.0,
    }
    total_weight = sum(task_weights.get(task, 1.0) for task in task_names)
    quotas = {}
    raw_quotas = {}
    assigned = 0
    for task in task_names:
        raw = n_samples * task_weights.get(task, 1.0) / total_weight
        raw_quotas[task] = raw
        quotas[task] = int(raw)
        assigned += quotas[task]
    if assigned < n_samples:
        remainders = sorted(task_names, key=lambda t: (raw_quotas[t] - quotas[t]), reverse=True)
        for task in remainders[: n_samples - assigned]:
            quotas[task] += 1
    return _build_samples_from_task_quotas(pools, quotas, tokenizer, seq_len, logger=logger, tag='target_task_balanced_mix')


def get_lm_target_mix(tokenizer, n_samples, seq_len, logger=None, lm_ratio=0.60):
    n_lm = max(1, int(round(n_samples * lm_ratio)))
    n_lm = min(n_lm, n_samples - 1) if n_samples > 1 else 0
    n_target = n_samples - n_lm
    _log(logger, f'[data] loading lm_target_mix | lm={n_lm} | target={n_target} | seq_len={seq_len}')
    samples = []
    if n_lm > 0:
        samples.append(get_wiki_ptb(tokenizer, n_lm, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_mix(tokenizer, n_target, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_lm_target_mix_less_ptb(tokenizer, n_samples, seq_len, logger=None, lm_ratio=0.60, wiki_ratio=0.65):
    n_lm = max(1, int(round(n_samples * lm_ratio)))
    n_lm = min(n_lm, n_samples - 1) if n_samples > 1 else 0
    n_target = n_samples - n_lm
    _log(logger, f'[data] loading lm_target_mix_less_ptb | lm={n_lm} | target={n_target} | wiki_ratio={wiki_ratio:.2f} | ptb_ratio={1.0-wiki_ratio:.2f} | seq_len={seq_len}')
    samples = []
    if n_lm > 0:
        samples.append(get_wiki_ptb(tokenizer, n_lm, seq_len, logger=logger, wiki_ratio=wiki_ratio))
    if n_target > 0:
        samples.append(get_target_task_mix(tokenizer, n_target, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_lm_target_mix_wikiheavy(tokenizer, n_samples, seq_len, logger=None, lm_ratio=0.60, wiki_ratio=0.75):
    n_lm = max(1, int(round(n_samples * lm_ratio)))
    n_lm = min(n_lm, n_samples - 1) if n_samples > 1 else 0
    n_target = n_samples - n_lm
    _log(logger, f'[data] loading lm_target_mix_wikiheavy | lm={n_lm} | target={n_target} | wiki_ratio={wiki_ratio:.2f} | ptb_ratio={1.0-wiki_ratio:.2f} | seq_len={seq_len}')
    samples = []
    if n_lm > 0:
        samples.append(get_wiki_ptb(tokenizer, n_lm, seq_len, logger=logger, wiki_ratio=wiki_ratio))
    if n_target > 0:
        samples.append(get_target_task_mix(tokenizer, n_target, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_lm_target_mix_wikiheavier(tokenizer, n_samples, seq_len, logger=None, lm_ratio=0.60, wiki_ratio=0.85):
    n_lm = max(1, int(round(n_samples * lm_ratio)))
    n_lm = min(n_lm, n_samples - 1) if n_samples > 1 else 0
    n_target = n_samples - n_lm
    _log(logger, f'[data] loading lm_target_mix_wikiheavier | lm={n_lm} | target={n_target} | wiki_ratio={wiki_ratio:.2f} | ptb_ratio={1.0-wiki_ratio:.2f} | seq_len={seq_len}')
    samples = []
    if n_lm > 0:
        samples.append(get_wiki_ptb(tokenizer, n_lm, seq_len, logger=logger, wiki_ratio=wiki_ratio))
    if n_target > 0:
        samples.append(get_target_task_mix(tokenizer, n_target, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_lm_target_mix_w60_p10_t30(tokenizer, n_samples, seq_len, logger=None, lm_ratio=0.70, wiki_ratio=6.0/7.0):
    n_lm = max(1, int(round(n_samples * lm_ratio)))
    n_lm = min(n_lm, n_samples - 1) if n_samples > 1 else 0
    n_target = n_samples - n_lm
    _log(logger, f'[data] loading lm_target_mix_w60_p10_t30 | lm={n_lm} | target={n_target} | wiki_ratio={wiki_ratio:.4f} | ptb_ratio={1.0-wiki_ratio:.4f} | seq_len={seq_len}')
    samples = []
    if n_lm > 0:
        samples.append(get_wiki_ptb(tokenizer, n_lm, seq_len, logger=logger, wiki_ratio=wiki_ratio))
    if n_target > 0:
        samples.append(get_target_task_mix(tokenizer, n_target, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_lm_target_mix_w67_p13_t20(tokenizer, n_samples, seq_len, logger=None, lm_ratio=0.80, wiki_ratio=67.0/80.0):
    n_lm = max(1, int(round(n_samples * lm_ratio)))
    n_lm = min(n_lm, n_samples - 1) if n_samples > 1 else 0
    n_target = n_samples - n_lm
    _log(logger, f'[data] loading lm_target_mix_w67_p13_t20 | lm={n_lm} | target={n_target} | wiki_ratio={wiki_ratio:.4f} | ptb_ratio={1.0-wiki_ratio:.4f} | seq_len={seq_len}')
    samples = []
    if n_lm > 0:
        samples.append(get_wiki_ptb(tokenizer, n_lm, seq_len, logger=logger, wiki_ratio=wiki_ratio))
    if n_target > 0:
        samples.append(get_target_task_mix(tokenizer, n_target, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_lm_target_mix_w75_p15_t10(tokenizer, n_samples, seq_len, logger=None, lm_ratio=0.90, wiki_ratio=75.0/90.0):
    n_lm = max(1, int(round(n_samples * lm_ratio)))
    n_lm = min(n_lm, n_samples - 1) if n_samples > 1 else 0
    n_target = n_samples - n_lm
    _log(logger, f'[data] loading lm_target_mix_w75_p15_t10 | lm={n_lm} | target={n_target} | wiki_ratio={wiki_ratio:.4f} | ptb_ratio={1.0-wiki_ratio:.4f} | seq_len={seq_len}')
    samples = []
    if n_lm > 0:
        samples.append(get_wiki_ptb(tokenizer, n_lm, seq_len, logger=logger, wiki_ratio=wiki_ratio))
    if n_target > 0:
        samples.append(get_target_task_mix(tokenizer, n_target, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_lm_target_mix_w82_p13_t05(tokenizer, n_samples, seq_len, logger=None, lm_ratio=0.95, wiki_ratio=82.0/95.0):
    n_lm = max(1, int(round(n_samples * lm_ratio)))
    n_lm = min(n_lm, n_samples - 1) if n_samples > 1 else 0
    n_target = n_samples - n_lm
    _log(logger, f'[data] loading lm_target_mix_w82_p13_t05 | lm={n_lm} | target={n_target} | wiki_ratio={wiki_ratio:.4f} | ptb_ratio={1.0-wiki_ratio:.4f} | seq_len={seq_len}')
    samples = []
    if n_lm > 0:
        samples.append(get_wiki_ptb(tokenizer, n_lm, seq_len, logger=logger, wiki_ratio=wiki_ratio))
    if n_target > 0:
        samples.append(get_target_task_mix(tokenizer, n_target, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_lm_target_mix_w60_p10_t30_plus20wiki(tokenizer, n_samples, seq_len, logger=None, lm_ratio=0.70, wiki_ratio=6.0/7.0, extra_wiki=20):
    n_lm = max(1, int(round(n_samples * lm_ratio)))
    n_lm = min(n_lm, n_samples - 1) if n_samples > 1 else 0
    n_target = n_samples - n_lm
    total = n_samples + extra_wiki
    _log(logger, f'[data] loading lm_target_mix_w60_p10_t30_plus20wiki | base_lm={n_lm} | base_target={n_target} | extra_wiki={extra_wiki} | total={total} | wiki_ratio={wiki_ratio:.4f} | ptb_ratio={1.0-wiki_ratio:.4f} | seq_len={seq_len}')
    samples = []
    if n_lm > 0:
        samples.append(get_wiki_ptb(tokenizer, n_lm, seq_len, logger=logger, wiki_ratio=wiki_ratio))
    if n_target > 0:
        samples.append(get_target_task_mix(tokenizer, n_target, seq_len, logger=logger))
    if extra_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, extra_wiki, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_lessptb_lm93_target7_balanced(tokenizer, n_samples, seq_len, logger=None, lm_count=93, target_count=7, wiki_ratio=0.65):
    _log(logger, f'[data] loading lessptb_lm93_target7_balanced | requested_n={n_samples} | lm_count={lm_count} | target_count={target_count} | total={lm_count + target_count} | wiki_ratio={wiki_ratio:.2f} | ptb_ratio={1.0-wiki_ratio:.2f} | seq_len={seq_len}')
    samples = []
    if lm_count > 0:
        samples.append(get_wiki_ptb(tokenizer, lm_count, seq_len, logger=logger, wiki_ratio=wiki_ratio))
    if target_count > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, target_count, seq_len, logger=logger))
    return torch.cat(samples, dim=0)



def get_lm_target_mix_w55_p20_t10_a15(tokenizer, n_samples, seq_len, logger=None, wiki_frac=50.0/150.0, ptb_frac=10.0/150.0, target_frac=70.0/150.0):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_target = int(round(n_samples * target_frac))
    n_book = n_samples - n_wiki - n_ptb - n_target
    _log(logger, f'[data] loading lm_target_mix_w55_p20_t10_a15 | wiki={n_wiki} | ptb={n_ptb} | target={n_target} | bookcorpus={n_book} | seq_len={seq_len} | weight_spec=50:10:70:20(normalized)')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, n_target, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki50_ptb10_target70_book20_boolqdown(tokenizer, n_samples, seq_len, logger=None, wiki_frac=50.0/150.0, ptb_frac=10.0/150.0, target_frac=70.0/150.0):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_target = int(round(n_samples * target_frac))
    n_book = n_samples - n_wiki - n_ptb - n_target
    _log(logger, f'[data] loading wiki50_ptb10_target70_book20_boolqdown | wiki={n_wiki} | ptb={n_ptb} | target={n_target} | bookcorpus={n_book} | seq_len={seq_len}')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, n_target, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki50_ptb10_target70_book20_boolqdown_v2(tokenizer, n_samples, seq_len, logger=None, wiki_frac=50.0/150.0, ptb_frac=10.0/150.0, target_frac=70.0/150.0):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_target = int(round(n_samples * target_frac))
    n_book = n_samples - n_wiki - n_ptb - n_target
    _log(logger, f'[data] loading wiki50_ptb10_target70_book20_boolqdown_v2 | wiki={n_wiki} | ptb={n_ptb} | target={n_target} | bookcorpus={n_book} | seq_len={seq_len}')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, n_target, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki50_ptb10_target80_book10_boolqdown_v1(tokenizer, n_samples, seq_len, logger=None, wiki_frac=50.0/150.0, ptb_frac=10.0/150.0, target_frac=80.0/150.0):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_target = int(round(n_samples * target_frac))
    n_book = n_samples - n_wiki - n_ptb - n_target
    _log(logger, f'[data] loading wiki50_ptb10_target80_book10_boolqdown_v1 | wiki={n_wiki} | ptb={n_ptb} | target={n_target} | bookcorpus={n_book} | seq_len={seq_len}')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, n_target, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki40_ptb05_target105_book00_boolqdown_v1(tokenizer, n_samples, seq_len, logger=None, wiki_frac=40.0/150.0, ptb_frac=5.0/150.0, target_frac=105.0/150.0):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_target = int(round(n_samples * target_frac))
    n_book = n_samples - n_wiki - n_ptb - n_target
    _log(logger, f'[data] loading wiki40_ptb05_target105_book00_boolqdown_v1 | wiki={n_wiki} | ptb={n_ptb} | target={n_target} | bookcorpus={n_book} | seq_len={seq_len}')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, n_target, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki35_ptb05_target110_book00_custommix_v1(tokenizer, n_samples, seq_len, logger=None, wiki_frac=35.0/150.0, ptb_frac=5.0/150.0, target_frac=110.0/150.0):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_target = int(round(n_samples * target_frac))
    n_book = n_samples - n_wiki - n_ptb - n_target
    _log(logger, f'[data] loading wiki35_ptb05_target110_book00_custommix_v1 | wiki={n_wiki} | ptb={n_ptb} | target={n_target} | bookcorpus={n_book} | seq_len={seq_len}')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, n_target, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki35_ptb05_target110_book00_custommix_v2(tokenizer, n_samples, seq_len, logger=None, wiki_frac=35.0/150.0, ptb_frac=5.0/150.0, target_frac=110.0/150.0):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_target = int(round(n_samples * target_frac))
    n_book = n_samples - n_wiki - n_ptb - n_target
    _log(logger, f'[data] loading wiki35_ptb05_target110_book00_custommix_v2 | wiki={n_wiki} | ptb={n_ptb} | target={n_target} | bookcorpus={n_book} | seq_len={seq_len}')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, n_target, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki35_ptb05_target110_book00_custommix_v3(tokenizer, n_samples, seq_len, logger=None, wiki_frac=35.0/150.0, ptb_frac=5.0/150.0, target_frac=110.0/150.0):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_target = int(round(n_samples * target_frac))
    n_book = n_samples - n_wiki - n_ptb - n_target
    _log(logger, f'[data] loading wiki35_ptb05_target110_book00_custommix_v3 | wiki={n_wiki} | ptb={n_ptb} | target={n_target} | bookcorpus={n_book} | seq_len={seq_len}')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, n_target, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki35_ptb06_target140_book00_custommix_v1(tokenizer, n_samples, seq_len, logger=None, wiki_frac=35.0/181.0, ptb_frac=6.0/181.0, target_frac=140.0/181.0):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_target = int(round(n_samples * target_frac))
    n_book = n_samples - n_wiki - n_ptb - n_target
    _log(logger, f'[data] loading wiki35_ptb06_target140_book00_custommix_v1 | wiki={n_wiki} | ptb={n_ptb} | target={n_target} | bookcorpus={n_book} | seq_len={seq_len} | weight_spec=35:6:140(normalized)')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, n_target, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki35_ptb06_target160_book00_custommix_v1(tokenizer, n_samples, seq_len, logger=None, wiki_frac=35.0/201.0, ptb_frac=6.0/201.0, target_frac=160.0/201.0):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_target = int(round(n_samples * target_frac))
    n_book = n_samples - n_wiki - n_ptb - n_target
    _log(logger, f'[data] loading wiki35_ptb06_target160_book00_custommix_v1 | wiki={n_wiki} | ptb={n_ptb} | target={n_target} | bookcorpus={n_book} | seq_len={seq_len} | weight_spec=35:6:160(normalized)')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, n_target, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki25_ptb03_target160_book00_custommix_v1(tokenizer, n_samples, seq_len, logger=None, wiki_frac=25.0/188.0, ptb_frac=3.0/188.0, target_frac=160.0/188.0):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_target = int(round(n_samples * target_frac))
    n_book = n_samples - n_wiki - n_ptb - n_target
    _log(logger, f'[data] loading wiki25_ptb03_target160_book00_custommix_v1 | wiki={n_wiki} | ptb={n_ptb} | target={n_target} | bookcorpus={n_book} | seq_len={seq_len} | weight_spec=25:3:160(normalized)')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, n_target, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki10_ptb01_target160_book00_custommix_v1(tokenizer, n_samples, seq_len, logger=None, wiki_frac=10.0/171.0, ptb_frac=1.0/171.0, target_frac=160.0/171.0):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_target = int(round(n_samples * target_frac))
    n_book = n_samples - n_wiki - n_ptb - n_target
    _log(logger, f'[data] loading wiki10_ptb01_target160_book00_custommix_v1 | wiki={n_wiki} | ptb={n_ptb} | target={n_target} | bookcorpus={n_book} | seq_len={seq_len} | weight_spec=10:1:160(normalized)')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, n_target, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki10_ptb01_target160_book00_custommix_v2(tokenizer, n_samples, seq_len, logger=None, wiki_frac=10.0/171.0, ptb_frac=1.0/171.0, target_frac=160.0/171.0):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_target = int(round(n_samples * target_frac))
    n_book = n_samples - n_wiki - n_ptb - n_target
    _log(logger, f'[data] loading wiki10_ptb01_target160_book00_custommix_v2 | wiki={n_wiki} | ptb={n_ptb} | target={n_target} | bookcorpus={n_book} | seq_len={seq_len} | weight_spec=10:1:160(normalized)')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, n_target, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki15_ptb02_target180_book00_custommix_v1(tokenizer, n_samples, seq_len, logger=None, wiki_frac=15.0/197.0, ptb_frac=2.0/197.0, target_frac=180.0/197.0):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_target = int(round(n_samples * target_frac))
    n_book = n_samples - n_wiki - n_ptb - n_target
    _log(logger, f'[data] loading wiki15_ptb02_target180_book00_custommix_v1 | wiki={n_wiki} | ptb={n_ptb} | target={n_target} | bookcorpus={n_book} | seq_len={seq_len} | weight_spec=15:2:180(normalized)')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, n_target, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki11_ptb01_target180_book00_custommix_v1(tokenizer, n_samples, seq_len, logger=None, wiki_frac=11.0/192.0, ptb_frac=1.0/192.0, target_frac=180.0/192.0):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_target = int(round(n_samples * target_frac))
    n_book = n_samples - n_wiki - n_ptb - n_target
    _log(logger, f'[data] loading wiki11_ptb01_target180_book00_custommix_v1 | wiki={n_wiki} | ptb={n_ptb} | target={n_target} | bookcorpus={n_book} | seq_len={seq_len} | weight_spec=11:1:180(normalized)')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, n_target, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki10_ptb00_target190_book00_custommix_v1(tokenizer, n_samples, seq_len, logger=None, wiki_frac=10.0/200.0, ptb_frac=0.0, target_frac=190.0/200.0):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_target = int(round(n_samples * target_frac))
    n_book = n_samples - n_wiki - n_ptb - n_target
    _log(logger, f'[data] loading wiki10_ptb00_target190_book00_custommix_v1 | wiki={n_wiki} | ptb={n_ptb} | target={n_target} | bookcorpus={n_book} | seq_len={seq_len} | weight_spec=10:0:190(normalized)')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, n_target, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki10_ptb00_target200_book00_custommix_v1(tokenizer, n_samples, seq_len, logger=None, wiki_frac=10.0/210.0, ptb_frac=0.0, target_frac=200.0/210.0):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_target = int(round(n_samples * target_frac))
    n_book = n_samples - n_wiki - n_ptb - n_target
    _log(logger, f'[data] loading wiki10_ptb00_target200_book00_custommix_v1 | wiki={n_wiki} | ptb={n_ptb} | target={n_target} | bookcorpus={n_book} | seq_len={seq_len} | weight_spec=10:0:200(normalized)')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, n_target, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_dataset(tokenizer, n_samples, seq_len, logger=None, wiki_frac=0.0, ptb_frac=0.0, target_frac=1.0):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_target = int(round(n_samples * target_frac))
    n_book = n_samples - n_wiki - n_ptb - n_target
    _log(logger, f'[data] loading dataset | wiki={n_wiki} | ptb={n_ptb} | target={n_target} | bookcorpus={n_book} | seq_len={seq_len} | weight_spec=0:0:200:0(normalized)')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, n_target, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki0_ptb00_target220_book00_plus10winogrande_custommix_v1(tokenizer, n_samples, seq_len, logger=None, wiki_frac=0.0, ptb_frac=0.0, target_frac=1.0, extra_winogrande=10):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_target = int(round(n_samples * target_frac))
    n_book = n_samples - n_wiki - n_ptb - n_target
    base_target = max(0, n_target - extra_winogrande)
    total = n_wiki + n_ptb + base_target + n_book + extra_winogrande
    _log(logger, f'[data] loading wiki0_ptb00_target220_book00_plus10winogrande_custommix_v1 | wiki={n_wiki} | ptb={n_ptb} | base_target={base_target} | bookcorpus={n_book} | extra_winogrande={extra_winogrande} | total={total} | seq_len={seq_len} | weight_spec=0:0:220:0 with final winogrande+10 under fixed total')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if base_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, base_target, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    if extra_winogrande > 0:
        pools = _build_target_task_text_pools(logger=logger)
        wg_texts = [t for t in pools['winogrande'] if isinstance(t, str) and t.strip()]
        _log(logger, f'[data] extra_winogrande valid_texts={len(wg_texts)} | using joined corpus sampler')
        if len(wg_texts) == 0:
            raise RuntimeError('No valid texts collected for extra_winogrande')
        joined_text = '\n\n'.join(wg_texts)
        tokenized_joined = tokenizer(joined_text, return_tensors='pt')
        total_len = tokenized_joined.input_ids.shape[1]
        if total_len >= seq_len:
            _log(logger, f'[data] extra_winogrande using joined corpus | total_tokens={total_len}')
            max_start = total_len - seq_len
            wg_samples = []
            for j in range(extra_winogrande):
                start = 0 if max_start <= 0 else min(j * seq_len, max_start)
                wg_samples.append(tokenized_joined.input_ids[:, start:start+seq_len])
            samples.append(torch.cat(wg_samples, dim=0))
        else:
            _log(logger, f'[data] extra_winogrande fallback repeat-pad | total_tokens={total_len} < seq_len={seq_len}')
            repeated = tokenized_joined.input_ids
            while repeated.shape[1] < seq_len:
                repeated = torch.cat([repeated, tokenized_joined.input_ids], dim=1)
            wg_samples = [repeated[:, :seq_len] for _ in range(extra_winogrande)]
            samples.append(torch.cat(wg_samples, dim=0))
    return torch.cat(samples, dim=0)


def get_wiki0_ptb00_targetquota180_custommix_v1(tokenizer, n_samples, seq_len, logger=None):
    quotas = {
        'arc_easy': 31,
        'arc_challenge': 32,
        'boolq': 0,
        'piqa': 25,
        'hellaswag': 23,
        'winogrande': 41,
        'openbookqa': 28,
    }
    total = sum(quotas.values())
    _log(logger, f'[data] loading wiki0_ptb00_targetquota180_custommix_v1 | wiki=0 | ptb=0 | bookcorpus=0 | explicit_target_total={total} | requested_n={n_samples} | seq_len={seq_len}')
    pools = _build_target_task_text_pools(logger=logger)
    return _build_samples_from_task_quotas(pools, quotas, tokenizer, seq_len, logger=logger, tag='target_task_explicit_quota_mix')


def get_wiki0_ptb00_targetquota_rebalance_v1(tokenizer, n_samples, seq_len, logger=None):
    quotas = {
        'arc_easy': 18,
        'arc_challenge': 42,
        'boolq': 16,
        'piqa': 16,
        'hellaswag': 32,
        'winogrande': 26,
        'openbookqa': 50,
    }
    total = sum(quotas.values())
    _log(logger, f'[data] loading wiki0_ptb00_targetquota_rebalance_v1 | wiki=0 | ptb=0 | bookcorpus=0 | explicit_target_total={total} | requested_n={n_samples} | seq_len={seq_len}')
    pools = _build_target_task_text_pools(logger=logger)
    return _build_samples_from_task_quotas(pools, quotas, tokenizer, seq_len, logger=logger, tag='target_task_explicit_quota_mix')


def get_target200_plus_openbookqa_wiki_tiny_v1(tokenizer, n_samples, seq_len, logger=None, extra_openbookqa=4, extra_wiki=2):
    n_wiki = min(int(extra_wiki), max(0, n_samples))
    n_target = max(0, n_samples - n_wiki)
    extra_openbookqa = min(int(extra_openbookqa), n_target)
    base_target = max(0, n_target - extra_openbookqa)
    total = base_target + extra_openbookqa + n_wiki
    _log(logger, f'[data] loading target200_plus_openbookqa_wiki_tiny_v1 | base_target={base_target} | extra_openbookqa={extra_openbookqa} | wiki={n_wiki} | total={total} | seq_len={seq_len}')

    samples = []
    pools = _build_target_task_text_pools(logger=logger)

    if base_target > 0:
        task_names = list(pools.keys())
        task_weights = {
            'arc_easy': 2.2,
            'arc_challenge': 2.3,
            'boolq': 0.005,
            'piqa': 1.75,
            'hellaswag': 1.6,
            'winogrande': 2.2,
            'openbookqa': 2.0,
        }
        total_weight = sum(task_weights.get(task, 1.0) for task in task_names)
        quotas = {}
        raw_quotas = {}
        assigned = 0
        for task in task_names:
            raw = base_target * task_weights.get(task, 1.0) / total_weight
            raw_quotas[task] = raw
            quotas[task] = int(raw)
            assigned += quotas[task]
        if assigned < base_target:
            remainders = sorted(task_names, key=lambda t: (raw_quotas[t] - quotas[t]), reverse=True)
            for task in remainders[: base_target - assigned]:
                quotas[task] += 1
        samples.append(_build_samples_from_task_quotas(pools, quotas, tokenizer, seq_len, logger=logger, tag='target200_plus_openbookqa_wiki_tiny_base'))

    if extra_openbookqa > 0:
        obqa_texts = [t for t in pools['openbookqa'] if isinstance(t, str) and t.strip()]
        _log(logger, f'[data] extra_openbookqa valid_texts={len(obqa_texts)} | using joined corpus sampler')
        if len(obqa_texts) == 0:
            raise RuntimeError('No valid texts collected for extra_openbookqa')
        joined_text = '\n\n'.join(obqa_texts)
        tokenized_joined = tokenizer(joined_text, return_tensors='pt')
        total_len = tokenized_joined.input_ids.shape[1]
        if total_len >= seq_len:
            _log(logger, f'[data] extra_openbookqa using joined corpus | total_tokens={total_len}')
            max_start = total_len - seq_len
            obqa_samples = []
            for j in range(extra_openbookqa):
                start = 0 if max_start <= 0 else min(j * seq_len, max_start)
                obqa_samples.append(tokenized_joined.input_ids[:, start:start+seq_len])
            samples.append(torch.cat(obqa_samples, dim=0))
        else:
            _log(logger, f'[data] extra_openbookqa fallback repeat-pad | total_tokens={total_len} < seq_len={seq_len}')
            repeated = tokenized_joined.input_ids
            while repeated.shape[1] < seq_len:
                repeated = torch.cat([repeated, tokenized_joined.input_ids], dim=1)
            obqa_samples = [repeated[:, :seq_len] for _ in range(extra_openbookqa)]
            samples.append(torch.cat(obqa_samples, dim=0))

    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    return torch.cat(samples, dim=0)



def get_wiki0_ptb00_targetquota_nowg_nobool_v1(tokenizer, n_samples, seq_len, logger=None):
    base_weights = {
        'arc_easy': 20,
        'arc_challenge': 48,
        'boolq': 0,
        'piqa': 18,
        'hellaswag': 42,
        'winogrande': 0,
        'openbookqa': 52,
    }
    total_weight = sum(base_weights.values())
    quotas = {k: 0 for k in base_weights}
    raw = {}
    assigned = 0
    for task, w in base_weights.items():
        val = n_samples * float(w) / max(1, total_weight)
        raw[task] = val
        quotas[task] = int(val)
        assigned += quotas[task]
    if assigned < n_samples:
        remainders = sorted(base_weights.keys(), key=lambda t: (raw[t] - quotas[t]), reverse=True)
        for task in remainders[: n_samples - assigned]:
            quotas[task] += 1
    total = sum(quotas.values())
    _log(logger, f'[data] loading wiki0_ptb00_targetquota_nowg_nobool_v1 | wiki=0 | ptb=0 | bookcorpus=0 | scaled_target_total={total} | requested_n={n_samples} | seq_len={seq_len} | quotas=' + ', '.join(f'{k}={v}' for k,v in quotas.items()))
    pools = _build_target_task_text_pools(logger=logger)
    return _build_samples_from_task_quotas(pools, quotas, tokenizer, seq_len, logger=logger, tag='target_task_nowg_nobool_quota_mix')


def get_wiki0_ptb00_targetquota_nowg_nobool_legacy180_v1(tokenizer, n_samples, seq_len, logger=None):
    quotas = {
        'arc_easy': 20,
        'arc_challenge': 48,
        'boolq': 0,
        'piqa': 18,
        'hellaswag': 42,
        'winogrande': 0,
        'openbookqa': 52,
    }
    total = sum(quotas.values())
    _log(logger, f'[data] loading wiki0_ptb00_targetquota_nowg_nobool_legacy180_v1 | wiki=0 | ptb=0 | bookcorpus=0 | legacy_target_total={total} | requested_n={n_samples} | seq_len={seq_len} | quotas=' + ', '.join(f'{k}={v}' for k,v in quotas.items()))
    pools = _build_target_task_text_pools(logger=logger)
    return _build_samples_from_task_quotas(pools, quotas, tokenizer, seq_len, logger=logger, tag='target_task_nowg_nobool_legacy180_quota_mix')


def get_wiki0_ptb00_targetquota100_custommix_v1(tokenizer, n_samples, seq_len, logger=None):
    quotas = {
        'arc_easy': 17,
        'arc_challenge': 18,
        'boolq': 0,
        'piqa': 14,
        'hellaswag': 13,
        'winogrande': 23,
        'openbookqa': 15,
    }
    total = sum(quotas.values())
    _log(logger, f'[data] loading wiki0_ptb00_targetquota100_custommix_v1 | wiki=0 | ptb=0 | bookcorpus=0 | explicit_target_total={total} | requested_n={n_samples} | seq_len={seq_len}')
    pools = _build_target_task_text_pools(logger=logger)
    return _build_samples_from_task_quotas(pools, quotas, tokenizer, seq_len, logger=logger, tag='target_task_explicit_quota_mix')


def get_wiki0_ptb00_targetquota70_custommix_v1(tokenizer, n_samples, seq_len, logger=None):
    quotas = {
        'arc_easy': 12,
        'arc_challenge': 13,
        'boolq': 0,
        'piqa': 10,
        'hellaswag': 9,
        'winogrande': 16,
        'openbookqa': 10,
    }
    total = sum(quotas.values())
    _log(logger, f'[data] loading wiki0_ptb00_targetquota70_custommix_v1 | wiki=0 | ptb=0 | bookcorpus=0 | explicit_target_total={total} | requested_n={n_samples} | seq_len={seq_len}')
    pools = _build_target_task_text_pools(logger=logger)
    return _build_samples_from_task_quotas(pools, quotas, tokenizer, seq_len, logger=logger, tag='target_task_explicit_quota_mix')


def get_wiki25_ptb5_targetquota70_custommix_v1(tokenizer, n_samples, seq_len, logger=None):
    n_wiki = 25
    n_ptb = 5
    quotas = {
        'arc_easy': 12,
        'arc_challenge': 12,
        'boolq': 0,
        'piqa': 10,
        'hellaswag': 9,
        'winogrande': 16,
        'openbookqa': 11,
    }
    total = n_wiki + n_ptb + sum(quotas.values())
    _log(logger, f'[data] loading wiki25_ptb5_targetquota70_custommix_v1 | wiki={n_wiki} | ptb={n_ptb} | bookcorpus=0 | explicit_target_total={sum(quotas.values())} | total={total} | requested_n={n_samples} | seq_len={seq_len}')
    samples = []
    samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    pools = _build_target_task_text_pools(logger=logger)
    samples.append(_build_samples_from_task_quotas(pools, quotas, tokenizer, seq_len, logger=logger, tag='target_task_explicit_quota_mix'))
    return torch.cat(samples, dim=0)


def get_wiki0_ptb00_targetquota120_adaptive20_v1(tokenizer, n_samples, seq_len, logger=None):
    quotas = {
        'arc_easy': 14,
        'arc_challenge': 20,
        'boolq': 10,
        'piqa': 18,
        'hellaswag': 16,
        'winogrande': 24,
        'openbookqa': 18,
    }
    total = sum(quotas.values())
    _log(logger, f'[data] loading wiki0_ptb00_targetquota120_adaptive20_v1 | wiki=0 | ptb=0 | bookcorpus=0 | explicit_target_total={total} | requested_n={n_samples} | seq_len={seq_len} | quotas=' + ', '.join(f'{k}={v}' for k,v in quotas.items()))
    pools = _build_target_task_text_pools(logger=logger)
    return _build_samples_from_task_quotas(pools, quotas, tokenizer, seq_len, logger=logger, tag='target_task_adaptive20_quota_mix')


def get_wiki0_ptb00_targetquota120_adaptive20_v3_aggressive(tokenizer, n_samples, seq_len, logger=None):
    quotas = {
        'arc_easy': 28,
        'arc_challenge': 26,
        'boolq': 2,
        'piqa': 16,
        'hellaswag': 16,
        'winogrande': 12,
        'openbookqa': 20,
    }
    total = sum(quotas.values())
    _log(logger, f'[data] loading wiki0_ptb00_targetquota120_adaptive20_v3_aggressive | wiki=0 | ptb=0 | bookcorpus=0 | explicit_target_total={total} | requested_n={n_samples} | seq_len={seq_len} | quotas=' + ', '.join(f'{k}={v}' for k,v in quotas.items()))
    pools = _build_target_task_text_pools(logger=logger)
    return _build_samples_from_task_quotas(pools, quotas, tokenizer, seq_len, logger=logger, tag='target_task_adaptive20_v3_aggressive_quota_mix')


def get_wiki0_ptb00_targetquota120_adaptive20_v4_arcmax(tokenizer, n_samples, seq_len, logger=None):
    quotas = {
        'arc_easy': 31,
        'arc_challenge': 29,
        'boolq': 0,
        'piqa': 16,
        'hellaswag': 16,
        'winogrande': 8,
        'openbookqa': 20,
    }
    total = sum(quotas.values())
    _log(logger, f'[data] loading wiki0_ptb00_targetquota120_adaptive20_v4_arcmax | wiki=0 | ptb=0 | bookcorpus=0 | explicit_target_total={total} | requested_n={n_samples} | seq_len={seq_len} | quotas=' + ', '.join(f'{k}={v}' for k,v in quotas.items()))
    pools = _build_target_task_text_pools(logger=logger)
    return _build_samples_from_task_quotas(pools, quotas, tokenizer, seq_len, logger=logger, tag='target_task_adaptive20_v4_arcmax_quota_mix')


def get_wiki10_ptb00_targetquota90_adaptive20_v5_arcwiki(tokenizer, n_samples, seq_len, logger=None):
    n_wiki = 10
    quotas = {
        'arc_easy': 24,
        'arc_challenge': 22,
        'boolq': 0,
        'piqa': 14,
        'hellaswag': 14,
        'winogrande': 6,
        'openbookqa': 10,
    }
    total = n_wiki + sum(quotas.values())
    _log(logger, f'[data] loading wiki10_ptb00_targetquota90_adaptive20_v5_arcwiki | wiki={n_wiki} | ptb=0 | bookcorpus=0 | explicit_target_total={sum(quotas.values())} | total={total} | requested_n={n_samples} | seq_len={seq_len} | quotas=' + ', '.join(f'{k}={v}' for k,v in quotas.items()))
    samples = []
    samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    pools = _build_target_task_text_pools(logger=logger)
    samples.append(_build_samples_from_task_quotas(pools, quotas, tokenizer, seq_len, logger=logger, tag='target_task_adaptive20_v5_arcwiki_quota_mix'))
    return torch.cat(samples, dim=0)


def get_lm_target_mix_w65_p10_b25(tokenizer, n_samples, seq_len, logger=None, wiki_frac=0.65, ptb_frac=0.10):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_book = n_samples - n_wiki - n_ptb
    _log(logger, f'[data] loading lm_target_mix_w65_p10_b25 | wiki={n_wiki} | ptb={n_ptb} | bookcorpus={n_book} | seq_len={seq_len}')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki70_ptb10_book20_plus10target(tokenizer, n_samples, seq_len, logger=None, wiki_frac=0.70, ptb_frac=0.10, extra_target=10):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_book = n_samples - n_wiki - n_ptb
    total = n_wiki + n_ptb + n_book + extra_target
    _log(logger, f'[data] loading wiki70_ptb10_book20_plus10target | wiki={n_wiki} | ptb={n_ptb} | bookcorpus={n_book} | extra_target={extra_target} | total={total} | seq_len={seq_len}')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    if extra_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, extra_target, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki70_ptb10_book20_plus20target(tokenizer, n_samples, seq_len, logger=None, wiki_frac=0.70, ptb_frac=0.10, extra_target=20):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_book = n_samples - n_wiki - n_ptb
    total = n_wiki + n_ptb + n_book + extra_target
    _log(logger, f'[data] loading wiki70_ptb10_book20_plus20target | wiki={n_wiki} | ptb={n_ptb} | bookcorpus={n_book} | extra_target={extra_target} | total={total} | seq_len={seq_len}')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    if extra_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, extra_target, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_wiki70_ptb10_book20_plus30target(tokenizer, n_samples, seq_len, logger=None, wiki_frac=0.70, ptb_frac=0.10, extra_target=30):
    n_wiki = int(round(n_samples * wiki_frac))
    n_ptb = int(round(n_samples * ptb_frac))
    n_book = n_samples - n_wiki - n_ptb
    total = n_wiki + n_ptb + n_book + extra_target
    _log(logger, f'[data] loading wiki70_ptb10_book20_plus30target | wiki={n_wiki} | ptb={n_ptb} | bookcorpus={n_book} | extra_target={extra_target} | total={total} | seq_len={seq_len}')
    samples = []
    if n_wiki > 0:
        samples.append(get_wikitext2_train(tokenizer, n_wiki, seq_len, logger=logger))
    if n_ptb > 0:
        samples.append(get_ptb_train(tokenizer, n_ptb, seq_len, logger=logger))
    if n_book > 0:
        samples.append(get_bookcorpus(tokenizer, n_book, seq_len, logger=logger))
    if extra_target > 0:
        samples.append(get_target_task_balanced_mix(tokenizer, extra_target, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def get_lm_target_mix_more_target(tokenizer, n_samples, seq_len, logger=None, lm_ratio=0.40):
    n_lm = max(1, int(round(n_samples * lm_ratio)))
    n_lm = min(n_lm, n_samples - 1) if n_samples > 1 else 0
    n_target = n_samples - n_lm
    _log(logger, f'[data] loading lm_target_mix_more_target | lm={n_lm} | target={n_target} | seq_len={seq_len}')
    samples = []
    if n_lm > 0:
        samples.append(get_wiki_ptb(tokenizer, n_lm, seq_len, logger=logger))
    if n_target > 0:
        samples.append(get_target_task_mix(tokenizer, n_target, seq_len, logger=logger))
    return torch.cat(samples, dim=0)


def _cache_file_for_examples(dataset, tokenizer, n_samples, seq_len):
    cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'cache_calibration')
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    if dataset == 'dataset':
        return os.path.join(cache_dir, 'dataset.pt.enc')
    tok_name = getattr(tokenizer, 'name_or_path', tokenizer.__class__.__name__)
    key = f'{dataset}__{tok_name}__n{n_samples}__len{seq_len}'
    digest = hashlib.md5(key.encode('utf-8')).hexdigest()[:10]
    safe_name = ''.join(c if c.isalnum() or c in '._-' else '_' for c in key)[:120]
    return os.path.join(cache_dir, f'{safe_name}__{digest}.pt.enc')


def _get_cache_fernet(logger=None):
    env_key = os.environ.get('CALIBRATION_CACHE_KEY') or os.environ.get('PRUNE_CACHE_KEY')
    if env_key:
        material = env_key.encode('utf-8')
        source = 'env'
    else:
        machine_fingerprint = '|'.join([
            getpass.getuser(),
            socket.gethostname(),
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
        ])
        material = machine_fingerprint.encode('utf-8')
        source = 'derived-local'
    key = base64.urlsafe_b64encode(hashlib.sha256(material).digest())
    _log(logger, f'[data] calibration cache crypto ready | key_source={source}')
    return Fernet(key)


def _save_encrypted_tensor_cache(tensor, cache_file, logger=None):
    import io
    fernet = _get_cache_fernet(logger=logger)
    buffer = io.BytesIO()
    torch.save({'tensor': tensor.cpu()}, buffer)
    encrypted = fernet.encrypt(buffer.getvalue())
    tmp_file = cache_file + '.tmp'
    with open(tmp_file, 'wb') as f:
        f.write(encrypted)
    os.replace(tmp_file, cache_file)
    _log(logger, f'[data] calibration cache saved(encrypted) | shape={tuple(tensor.shape)} | file={cache_file} | bytes={len(encrypted)}')


def _load_encrypted_tensor_cache(cache_file, logger=None):
    import io
    fernet = _get_cache_fernet(logger=logger)
    with open(cache_file, 'rb') as f:
        encrypted = f.read()
    decrypted = fernet.decrypt(encrypted)
    payload = torch.load(io.BytesIO(decrypted), map_location='cpu')
    tensor = payload['tensor'] if isinstance(payload, dict) and 'tensor' in payload else payload
    _log(logger, f'[data] calibration cache loaded(encrypted) | shape={tuple(tensor.shape)} | file={cache_file}')
    return tensor


def get_examples(dataset, tokenizer, n_samples, seq_len=128, logger=None):
    cache_file = _cache_file_for_examples(dataset, tokenizer, n_samples, seq_len)
    if os.path.exists(cache_file):
        try:
            return _load_encrypted_tensor_cache(cache_file, logger=logger)
        except InvalidToken:
            _log(logger, f'[data] calibration cache invalid token, rebuilding | file={cache_file}')
        except Exception as e:
            _log(logger, f'[data] calibration cache load failed, rebuilding | file={cache_file} | error={e}')

    _log(logger, f'[data] calibration cache miss | dataset={dataset} | file={cache_file}')
    if dataset == 'c4':
        tensor = get_c4(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'bookcorpus':
        tensor = get_bookcorpus(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'mixed':
        tensor = get_mixed(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'bookcorpus_plus':
        tensor = get_bookcorpus_plus(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wikitext2_train':
        tensor = get_wikitext2_train(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'ptb_train':
        tensor = get_ptb_train(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki_ptb':
        tensor = get_wiki_ptb(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'target_task_mix':
        tensor = get_target_task_mix(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'target_task_balanced_mix':
        tensor = get_target_task_balanced_mix(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'lm_target_mix':
        tensor = get_lm_target_mix(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'lm_target_mix_less_ptb':
        tensor = get_lm_target_mix_less_ptb(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'lm_target_mix_wikiheavy':
        tensor = get_lm_target_mix_wikiheavy(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'lm_target_mix_wikiheavier':
        tensor = get_lm_target_mix_wikiheavier(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'lm_target_mix_w60_p10_t30':
        tensor = get_lm_target_mix_w60_p10_t30(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'lm_target_mix_w60_p10_t30_plus20wiki':
        tensor = get_lm_target_mix_w60_p10_t30_plus20wiki(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'lessptb_lm93_target7_balanced':
        tensor = get_lessptb_lm93_target7_balanced(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'lm_target_mix_w55_p20_t10_a15':
        tensor = get_lm_target_mix_w55_p20_t10_a15(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'lm_target_mix_w67_p13_t20':
        tensor = get_lm_target_mix_w67_p13_t20(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'lm_target_mix_w65_p10_b25':
        tensor = get_lm_target_mix_w65_p10_b25(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'lm_target_mix_w75_p15_t10':
        tensor = get_lm_target_mix_w75_p15_t10(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki70_ptb10_book20_plus10target':
        tensor = get_wiki70_ptb10_book20_plus10target(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki70_ptb10_book20_plus20target':
        tensor = get_wiki70_ptb10_book20_plus20target(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki70_ptb10_book20_plus30target':
        tensor = get_wiki70_ptb10_book20_plus30target(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'lm_target_mix_w82_p13_t05':
        tensor = get_lm_target_mix_w82_p13_t05(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki50_ptb10_target70_book20_boolqdown':
        tensor = get_wiki50_ptb10_target70_book20_boolqdown(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki50_ptb10_target70_book20_boolqdown_v2':
        tensor = get_wiki50_ptb10_target70_book20_boolqdown_v2(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki50_ptb10_target80_book10_boolqdown_v1':
        tensor = get_wiki50_ptb10_target80_book10_boolqdown_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki40_ptb05_target105_book00_boolqdown_v1':
        tensor = get_wiki40_ptb05_target105_book00_boolqdown_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki35_ptb05_target110_book00_custommix_v1':
        tensor = get_wiki35_ptb05_target110_book00_custommix_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki35_ptb05_target110_book00_custommix_v2':
        tensor = get_wiki35_ptb05_target110_book00_custommix_v2(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki35_ptb05_target110_book00_custommix_v3':
        tensor = get_wiki35_ptb05_target110_book00_custommix_v3(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki35_ptb06_target140_book00_custommix_v1':
        tensor = get_wiki35_ptb06_target140_book00_custommix_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki35_ptb06_target160_book00_custommix_v1':
        tensor = get_wiki35_ptb06_target160_book00_custommix_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki25_ptb03_target160_book00_custommix_v1':
        tensor = get_wiki25_ptb03_target160_book00_custommix_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki10_ptb01_target160_book00_custommix_v1':
        tensor = get_wiki10_ptb01_target160_book00_custommix_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki10_ptb01_target160_book00_custommix_v2':
        tensor = get_wiki10_ptb01_target160_book00_custommix_v2(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki15_ptb02_target180_book00_custommix_v1':
        tensor = get_wiki15_ptb02_target180_book00_custommix_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki11_ptb01_target180_book00_custommix_v1':
        tensor = get_wiki11_ptb01_target180_book00_custommix_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki10_ptb00_target190_book00_custommix_v1':
        tensor = get_wiki10_ptb00_target190_book00_custommix_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki10_ptb00_target200_book00_custommix_v1':
        tensor = get_wiki10_ptb00_target200_book00_custommix_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'dataset':
        tensor = get_dataset(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki0_ptb00_target220_book00_plus10winogrande_custommix_v1':
        tensor = get_wiki0_ptb00_target220_book00_plus10winogrande_custommix_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki0_ptb00_targetquota180_custommix_v1':
        tensor = get_wiki0_ptb00_targetquota180_custommix_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki0_ptb00_targetquota_rebalance_v1':
        tensor = get_wiki0_ptb00_targetquota_rebalance_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki0_ptb00_targetquota_nowg_nobool_v1':
        tensor = get_wiki0_ptb00_targetquota_nowg_nobool_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki0_ptb00_targetquota_nowg_nobool_legacy180_v1':
        tensor = get_wiki0_ptb00_targetquota_nowg_nobool_legacy180_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki0_ptb00_targetquota120_adaptive20_v1':
        tensor = get_wiki0_ptb00_targetquota120_adaptive20_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki0_ptb00_targetquota120_adaptive20_v3_aggressive':
        tensor = get_wiki0_ptb00_targetquota120_adaptive20_v3_aggressive(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki0_ptb00_targetquota120_adaptive20_v4_arcmax':
        tensor = get_wiki0_ptb00_targetquota120_adaptive20_v4_arcmax(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki10_ptb00_targetquota90_adaptive20_v5_arcwiki':
        tensor = get_wiki10_ptb00_targetquota90_adaptive20_v5_arcwiki(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'target200_plus_openbookqa_wiki_tiny_v1':
        tensor = get_target200_plus_openbookqa_wiki_tiny_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki0_ptb00_targetquota100_custommix_v1':
        tensor = get_wiki0_ptb00_targetquota100_custommix_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki0_ptb00_targetquota70_custommix_v1':
        tensor = get_wiki0_ptb00_targetquota70_custommix_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'wiki25_ptb5_targetquota70_custommix_v1':
        tensor = get_wiki25_ptb5_targetquota70_custommix_v1(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'lm_target_mix_more_target':
        tensor = get_lm_target_mix_more_target(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'bookcorpus_wiki':
        tensor = get_bookcorpus_wiki(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'bookcorpus_wiki_plus50book':
        tensor = get_bookcorpus_wiki_plus50book(tokenizer, n_samples, seq_len, logger=logger)
    elif dataset == 'lm_target_mix_plus50book':
        tensor = get_lm_target_mix_plus50book(tokenizer, n_samples, seq_len, logger=logger)
    else:
        raise NotImplementedError

    try:
        _save_encrypted_tensor_cache(tensor, cache_file, logger=logger)
    except Exception as e:
        _log(logger, f'[data] calibration cache save skipped | file={cache_file} | error={e}')
    return tensor
