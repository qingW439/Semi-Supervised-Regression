"""Container-aware CPU memory preflight for full RoBERTa AdamW training."""
from pathlib import Path

GIB=1024**3
MINIMUM_GIB=8  # Conservative operational floor, not a measured sufficiency guarantee.

def available_memory_bytes(root=Path('/sys/fs/cgroup')):
    host=0
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemTotal:'):
            host=int(line.split()[1])*1024
            break
    limits=[host]
    for path in (root/'memory.max',root/'memory/memory.limit_in_bytes'):
        if path.is_file():
            value=path.read_text().strip()
            if value.isdigit(): limits.append(int(value))
    return min(limits)

def require_cpu_memory(minimum_gib=MINIMUM_GIB):
    available=available_memory_bytes()
    if available<minimum_gib*GIB:
        raise RuntimeError(f'CPU RoBERTa full-parameter AdamW needs at least {minimum_gib} GiB '
                           f'container RAM for this runner; detected {available/GIB:.2f} GiB. '
                           'No model update was started.')
    return available
