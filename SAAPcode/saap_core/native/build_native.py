import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TARGETS = [
    ('saap_pf_trace.cpp', 'saap_pf_trace.so'),
    ('saap_pf_stats.cpp', 'saap_pf_stats.so'),
    ('saap_pf_align.cpp', 'saap_pf_align.so'),
]


def build_one(src_name, so_name):
    src = ROOT / src_name
    out = ROOT / so_name
    cmd = [
        'g++', '-std=c++11', '-shared', '-fPIC', '-O2',
        str(src), '-o', str(out),
    ]
    subprocess.check_call(cmd)
    return out


def build_all():
    built = []
    for src_name, so_name in TARGETS:
        built.append(str(build_one(src_name, so_name)))
    return built


if __name__ == '__main__':
    for item in build_all():
        print(item)
