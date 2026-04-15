def apply_runtime(args):
    args.export_calibration_bundle = False
    args.run_inline_recovery_ce = False
    args.run_stage2_full_ce_subprocess = False
    args.inline_recovery_per_device_train_batch_size = 1
    args.inline_recovery_gradient_accumulation_steps = 2
    args.inline_recovery_learning_rate = 1e-5
    args.inline_recovery_num_train_epochs = 1.0
    args.inline_recovery_weight_decay = 0.0
    args.inline_recovery_warmup_ratio = 0.03
    args.inline_recovery_logging_steps = 5
    args.inline_recovery_save_steps = 100
    args.inline_recovery_save_total_limit = 2
    args.inline_recovery_bf16 = False
    args.inline_recovery_fp16 = False
    args.inline_recovery_train_last_k_mlp_layers = 0
    args.inline_recovery_train_mlp_gate_proj = True
    args.inline_recovery_train_mlp_up_proj = True
    args.inline_recovery_train_mlp_down_proj = True
    args.save_model = True
    args.export_hf_after_save = False
    args.auto_eval_after_prune = True
    args.auto_eval_after_save = False
    args.auto_eval_device = 'cuda'
    args.auto_eval_lm_eval_batch_size = 8
    args.auto_eval_ppl_batch_size = 16
    args.auto_eval_ppl_max_seq_len = 128
    return args
