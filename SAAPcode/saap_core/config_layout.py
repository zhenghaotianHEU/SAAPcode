def layout():
    return {
        't': 1.0,
        'p': 0.95,
        'msl': 128,
        'bals': 4,
        'bale': 32,
        'bmls': 4,
        'bmle': 32,
        'pfnl': 2,
        'pll': True,
    }


def apply_layout(args):
    l = layout()
    args.temperature = l['t']
    args.top_p = l['p']
    args.max_seq_len = l['msl']
    args.block_attention_layer_start = l['bals']
    args.block_attention_layer_end = l['bale']
    args.block_mlp_layer_start = l['bmls']
    args.block_mlp_layer_end = l['bmle']
    args.protect_first_n_layers = l['pfnl']
    args.protect_last_layer = l['pll']
    return args
