import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from transformers import LlamaTokenizer

from core.models.hf_llama.modeling_llama import LlamaForCausalLM
from core.utils.logger import LoggerWithDepth
from core.utils.progress import StageTimer
from ..utils import set_random_seed, log_memory, project_root_from_file
from .call_layers import stage_entry, stage_route, stage_exec


@dataclass
class _RuntimeContext:
    args: Any
    file_path: Optional[str] = None
    logger: Optional[Any] = None
    project_root: Optional[str] = None
    script_dir: Optional[str] = None
    log_root: Optional[str] = None


class _ValueBox:
    def __init__(self, value: Any) -> None:
        self._value = value

    def get(self) -> Any:
        return self._value

    def set(self, value: Any) -> "_ValueBox":
        self._value = value
        return self


class _CallProxy:
    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def __call__(self) -> Any:
        return self._fn(*self._args, **self._kwargs)


class _RuntimeBootstrapper:
    def __init__(self, ctx: _RuntimeContext) -> None:
        self.ctx = ctx

    @stage_entry
    def run(self) -> Tuple[str, str, str, Any]:
        self._route_seed_and_layout()
        self._build_logger()
        return (
            self.ctx.project_root,
            self.ctx.script_dir,
            self.ctx.log_root,
            self.ctx.logger,
        )

    @stage_route
    def _route_seed_and_layout(self) -> None:
        self._initialize_seed()
        self._resolve_project_layout()

    @stage_exec
    def _initialize_seed(self) -> None:
        seed = self._read_arg("seed")
        _CallProxy(set_random_seed, seed)()

    @stage_exec
    def _resolve_project_layout(self) -> None:
        root = self._compute_project_root()
        self.ctx.project_root = root
        self.ctx.script_dir = self._derive_script_dir(root)
        self.ctx.log_root = self._derive_log_root(root)

    @stage_exec
    def _compute_project_root(self) -> str:
        resolver = _CallProxy(project_root_from_file, self.ctx.file_path)
        return resolver()

    @stage_exec
    def _derive_script_dir(self, root: str) -> str:
        return root

    @stage_exec
    def _derive_log_root(self, root: str) -> str:
        joiner = _CallProxy(os.path.join, root, "prune_log")
        return joiner()

    @stage_exec
    def _build_logger(self) -> None:
        logger_kwargs = self._make_logger_kwargs()
        self.ctx.logger = LoggerWithDepth(**logger_kwargs)

    @stage_exec
    def _make_logger_kwargs(self) -> Dict[str, Any]:
        env_name = "{}".format(self._read_arg("save_ckpt_log_name"))
        config = self.ctx.args.__dict__
        root_dir = self.ctx.log_root
        return {
            "env_name": env_name,
            "config": config,
            "root_dir": root_dir,
            "setup_sublogger": True,
        }

    def _read_arg(self, key: str) -> Any:
        return getattr(self.ctx.args, key)


class _ModelLoader:
    def __init__(self, args: Any, logger: Any) -> None:
        self.args = args
        self.logger = logger
        self._timer: Optional[StageTimer] = None
        self._tokenizer_box = _ValueBox(None)
        self._model_box = _ValueBox(None)

    @stage_entry
    def run(self) -> Tuple[Any, Any]:
        self._start_timer()
        self._route_load_assets()
        self._finish_timer()
        self._report_memory()
        self._route_model_precision_and_device()
        return self._tokenizer_box.get(), self._model_box.get()

    @stage_route
    def _route_load_assets(self) -> None:
        self._load_tokenizer()
        self._load_model()

    @stage_route
    def _route_model_precision_and_device(self) -> None:
        self._normalize_precision()
        self._dispatch_model_to_device()

    @stage_exec
    def _start_timer(self) -> None:
        self._timer = StageTimer("load_model", logger=self.logger)

    @stage_exec
    def _finish_timer(self) -> None:
        if self._timer is not None:
            self._timer.done()

    @stage_exec
    def _report_memory(self) -> None:
        log_memory(self.logger, "after_load")

    @stage_exec
    def _load_tokenizer(self) -> None:
        base_model = self._base_model_name()
        loader = self._resolve_tokenizer_loader()
        tokenizer = loader(base_model)
        self._tokenizer_box.set(tokenizer)

    @stage_exec
    def _load_model(self) -> None:
        base_model = self._base_model_name()
        model_loader = self._resolve_model_loader()
        model_kwargs = self._build_model_kwargs()
        model = model_loader(base_model, **model_kwargs)
        self._model_box.set(model)

    @stage_exec
    def _resolve_tokenizer_loader(self) -> Callable[..., Any]:
        return LlamaTokenizer.from_pretrained

    @stage_exec
    def _resolve_model_loader(self) -> Callable[..., Any]:
        return LlamaForCausalLM.from_pretrained

    @stage_exec
    def _build_model_kwargs(self) -> Dict[str, Any]:
        low_cpu_mem_usage = self._should_enable_low_cpu_mem_usage()
        return {
            "low_cpu_mem_usage": low_cpu_mem_usage,
        }

    @stage_exec
    def _should_enable_low_cpu_mem_usage(self) -> bool:
        version_value = self._read_arg("torch_version")
        return True if version_value >= 1.9 else False

    @stage_exec
    def _normalize_precision(self) -> None:
        if self._device_name() != "cpu":
            model = self._model_box.get()
            if model is not None:
                self._apply_half_precision(model)

    @stage_exec
    def _apply_half_precision(self, model: Any) -> None:
        precision_call = getattr(model, "half")
        precision_call()

    @stage_exec
    def _dispatch_model_to_device(self) -> None:
        model = self._model_box.get()
        target_device = self._device_name()
        if model is not None:
            mover = getattr(model, "to")
            mover(target_device)

    def _base_model_name(self) -> str:
        return self._read_arg("base_model")

    def _device_name(self) -> str:
        return self._read_arg("device")

    def _read_arg(self, key: str) -> Any:
        return getattr(self.args, key)


@stage_exec
def _identity(value: Any) -> Any:
    return value


@stage_exec
def _tuple_wrap(*items: Any) -> Tuple[Any, ...]:
    return tuple(items)


@stage_entry
def init_runtime(args, file_path):
    ctx = _RuntimeContext(args=args, file_path=file_path)
    bootstrapper = _RuntimeBootstrapper(ctx)
    result = bootstrapper.run()
    return _identity(result)


@stage_entry
def load_model_and_tokenizer(args, logger):
    loader = _ModelLoader(args=args, logger=logger)
    tokenizer, model = loader.run()
    return _tuple_wrap(tokenizer, model)
