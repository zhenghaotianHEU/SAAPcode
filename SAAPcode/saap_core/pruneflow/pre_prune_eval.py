import torch
from core.evaluator.ppl import PPLMetric
from core.templates.prompts import prompts
from .call_layers import stage_entry, stage_route, stage_exec


@stage_exec
def _run_generation_before_prune(model, tokenizer, args, logger):
    logger.log("\n==================Generation Results before Pruning================\n")
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(args.device)
            generation_output = model.generate(
                input_ids=input_ids,
                do_sample=True,
                top_k=50,
                max_length=args.max_seq_len,
                top_p=args.top_p,
                temperature=args.temperature,
            )
            logger.log(tokenizer.decode(generation_output[0]))


@stage_exec
def _run_ppl_before_prune(model, tokenizer, args, logger):
    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.device)
    logger.log(f'PPL before pruning: {ppl}')


@stage_route
def _route_pre_prune_eval(model, tokenizer, args, logger, test_before_train):
    if not test_before_train:
        return
    _run_generation_before_prune(model, tokenizer, args, logger)
    _run_ppl_before_prune(model, tokenizer, args, logger)


@stage_entry
def run_pre_prune_eval(model, tokenizer, args, logger, test_before_train):
    return _route_pre_prune_eval(model, tokenizer, args, logger, test_before_train)
