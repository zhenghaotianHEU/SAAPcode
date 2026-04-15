def apply_structural(args):
    args.saap_vector_reduction = 'mean'
    args.saap_element_reduction = 'sum'
    args.saap_beta0_v = 0.0
    args.saap_beta1_v = 1.0
    args.saap_beta2_v = 0.0
    args.saap_beta0_e = 0.0
    args.saap_beta1_e = 1.0
    args.saap_beta2_e = 0.0
    args.disable_saap_alignment = False
    args.saap_alignment_mode = 'quantile'
    args.saap_score_temperature = 1.0
    args.saap_score_floor_quantile = 0.0
    args.saap_module_score_bias = 0.03
    args.saap_use_grad_branch = True
    args.saap_grad_branch_reduction = 'mean'
    args.saap_grad_branch_weight = 0.35
    args.saap_signed_grad_branch = False
    return args
