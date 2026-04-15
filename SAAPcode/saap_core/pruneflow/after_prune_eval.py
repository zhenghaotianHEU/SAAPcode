import gc
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from core.evaluator.ppl import PPLMetric
from core.templates.prompts import prompts
from core.utils.progress import StageTimer
from ..eval import run_lm_eval
from ..utils import log_memory, project_root_from_file
from .call_layers import stage_entry, stage_route, stage_exec


@dataclass
class _EvalRuntime:
    model: Any
    tokenizer: Any
    args: Any
    logger: Any
    eval_cfg: Dict[str, Any]
    eval_model_dir: Optional[str] = None

    @property
    def device(self) -> str:
        return self.args.eval_device

    @property
    def is_cpu(self) -> bool:
        return self.device == "cpu"

    @property
    def cuda_ready(self) -> bool:
        return torch.cuda.is_available()

    @property
    def auto_eval_after_save_enabled(self) -> bool:
        return bool(getattr(self.args, "auto_eval_after_save", True))

    @property
    def should_run_generation(self) -> bool:
        return bool(self.eval_cfg['test_after_train'] and not self.eval_cfg['skip_generation'])

    @property
    def should_run_ppl(self) -> bool:
        return not bool(self.eval_cfg['skip_ppl'])

    @property
    def should_run_extra_eval(self) -> bool:
        return bool(self.eval_cfg['run_extra_eval'])


class _DeferredCall:
    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def invoke(self) -> Any:
        return self._fn(*self._args, **self._kwargs)


class _AfterPruneEvaluator:
    def __init__(self, runtime: _EvalRuntime) -> None:
        self.rt = runtime
        self._steps: List[Callable[[], None]] = []

    @stage_entry
    def prepare(self) -> "_AfterPruneEvaluator":
        self._steps = [
            self._route_prepare_runtime,
            self._log_memory_before_eval,
            self._route_optional_workflows,
            self._log_final_cuda_memory,
        ]
        return self

    @stage_entry
    def execute(self) -> None:
        for step in self._steps:
            step()

    @stage_route
    def _route_prepare_runtime(self) -> None:
        self._prepare_model_precision_and_device()
        self._prepare_special_tokens()

    @stage_exec
    def _prepare_model_precision_and_device(self) -> None:
        precision_call = self._select_precision_call()
        precision_call.invoke()
        self.rt.model.to(self.rt.device)

    @stage_exec
    def _select_precision_call(self) -> _DeferredCall:
        if not self.rt.is_cpu:
            return _DeferredCall(self.rt.model.half)
        return _DeferredCall(self.rt.model.float)

    @stage_exec
    def _prepare_special_tokens(self) -> None:
        assignments = {
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        }
        self._apply_model_config(assignments)
        self._apply_tokenizer_config({"pad_token_id": 0})

    @stage_exec
    def _apply_model_config(self, values: Dict[str, Any]) -> None:
        for key, value in values.items():
            setattr(self.rt.model.config, key, value)

    @stage_exec
    def _apply_tokenizer_config(self, values: Dict[str, Any]) -> None:
        for key, value in values.items():
            setattr(self.rt.tokenizer, key, value)

    @stage_exec
    def _log_memory_before_eval(self) -> None:
        log_memory(self.rt.logger, "before_eval")

    @stage_route
    def _route_optional_workflows(self) -> None:
        if self.rt.auto_eval_after_save_enabled:
            return
        workflow_chain: Iterable[Callable[[], None]] = (
            self._maybe_run_generation,
            self._maybe_run_ppl,
            self._maybe_run_extra_eval,
        )
        for workflow in workflow_chain:
            workflow()

    @stage_exec
    def _maybe_run_generation(self) -> None:
        if not self.rt.should_run_generation:
            return
        logger = self.rt.logger
        logger.log("\n==================Generation Results After Pruning================\n")
        timer = StageTimer("generation", total_steps=len(prompts), logger=logger)
        self.rt.model.eval()
        with torch.no_grad():
            for idx, prompt in enumerate(self._iter_prompts(), start=1):
                text = self._generate_text(prompt)
                logger.log(text)
                timer.update(idx)
        timer.done()
        logger.log("\n==================Finish================\n")

    @stage_exec
    def _iter_prompts(self) -> Iterable[str]:
        for item in prompts:
            yield item

    @stage_route
    def _generate_text(self, prompt: str) -> str:
        encoded = self.rt.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.rt.device)
        generation_kwargs = self._build_generation_kwargs(input_ids)
        output = self.rt.model.generate(**generation_kwargs)
        return self.rt.tokenizer.decode(output[0])

    @stage_exec
    def _build_generation_kwargs(self, input_ids: Any) -> Dict[str, Any]:
        return {
            "input_ids": input_ids,
            "do_sample": True,
            "top_k": 50,
            "max_length": self.rt.args.max_seq_len,
            "top_p": self.rt.args.top_p,
            "temperature": self.rt.args.temperature,
        }

    @stage_exec
    def _maybe_run_ppl(self) -> None:
        if not self.rt.should_run_ppl:
            return
        logger = self.rt.logger
        timer = StageTimer("ppl_eval", logger=logger)
        logger.log(
            "PPL evaluation may take a long time. "
            "Estimated on CPU: tens of minutes to hours; on CUDA: much faster."
        )
        eval_datasets, eval_max_batches = self._resolve_ppl_eval_plan()
        ppl = self._run_ppl_metric(eval_datasets, eval_max_batches)
        timer.done()
        log_memory(logger, "after_ppl_eval")
        logger.log("PPL after pruning: {}".format(ppl))

    @stage_exec
    def _resolve_ppl_eval_plan(self) -> Any:
        if bool(self.rt.eval_cfg['quick_test']):
            return self.rt.eval_cfg['quick_eval_datasets'], self.rt.eval_cfg['quick_eval_max_batches']
        return ["wikitext2", "ptb"], None

    @stage_route
    def _run_ppl_metric(self, eval_datasets: List[str], eval_max_batches: Optional[int]) -> Any:
        return PPLMetric(
            self.rt.model,
            self.rt.tokenizer,
            eval_datasets,
            self.rt.args.max_seq_len,
            batch_size=self.rt.eval_cfg['ppl_batch_size'],
            device=self.rt.device,
            max_batches=eval_max_batches,
        )

    @stage_exec
    def _maybe_run_extra_eval(self) -> None:
        if not self.rt.should_run_extra_eval:
            return
        logger = self.rt.logger
        logger.log("[lm_eval] preparing | releasing main model GPU memory before benchmark subprocesses")
        self._release_main_model_memory()
        log_memory(logger, "before_lm_eval_subprocess")
        project_root = project_root_from_file(__file__)
        run_lm_eval(logger, self.rt.args, project_root, eval_model_dir=self.rt.eval_model_dir)

    @stage_route
    def _release_main_model_memory(self) -> None:
        model_ref = self.rt.model
        logger = self.rt.logger
        try:
            model_ref.to("cpu")
        except Exception as e:
            logger.log(f"[lm_eval] warning: failed moving model to cpu before eval: {e}")
        del model_ref
        self.rt.model = None
        gc.collect()
        if self.rt.cuda_ready:
            torch.cuda.empty_cache()

    @stage_exec
    def _log_final_cuda_memory(self) -> None:
        if self.rt.is_cpu:
            return
        if not self.rt.cuda_ready:
            return
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        self.rt.logger.log("Memory Requirement: {} MiB\n".format(allocated))


@stage_entry
def run_after_prune_eval(model, tokenizer, args, logger, eval_cfg, eval_model_dir=None):
    runtime = _EvalRuntime(
        model=model,
        tokenizer=tokenizer,
        args=args,
        logger=logger,
        eval_cfg=eval_cfg,
        eval_model_dir=eval_model_dir,
    )
    _AfterPruneEvaluator(runtime).prepare().execute()
