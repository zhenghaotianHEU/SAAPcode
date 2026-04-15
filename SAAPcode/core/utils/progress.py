import time
from typing import Optional


class StageTimer:
    def __init__(self, name: str, total_steps: Optional[int] = None, logger=None):
        self.name = name
        self.total_steps = total_steps
        self.logger = logger
        self.start_time = time.time()
        self.last_time = self.start_time

    def _emit(self, msg: str):
        if self.logger is not None:
            self.logger.log(msg)

    def update(self, step: int, extra: str = ""):
        now = time.time()
        elapsed = now - self.start_time
        if self.total_steps is not None and step > 0:
            avg = elapsed / step
            remaining = max(self.total_steps - step, 0) * avg
            self._emit(f"[{self.name}] step {step}/{self.total_steps} | elapsed {elapsed:.1f}s | eta {remaining:.1f}s {extra}".rstrip())
        else:
            self._emit(f"[{self.name}] elapsed {elapsed:.1f}s {extra}".rstrip())
        self.last_time = now

    def done(self, extra: str = ""):
        elapsed = time.time() - self.start_time
        self._emit(f"[{self.name}] done in {elapsed:.1f}s {extra}".rstrip())
