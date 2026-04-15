def apply_pruning(args):
    class PruningProfileBuilder:
        def __init__(self, namespace):
            self.args = namespace
            self.profile = {}
            self.runtime = {}
            self.hybrid = {}
            self.structural = {}
            self.taylor = {}
            self.reconstruction = {}

        def build(self):
            self._build_runtime_profile()
            self._build_hybrid_profile()
            self._build_structural_profile()
            self._build_taylor_profile()
            self._build_reconstruction_profile()

            self._compose_profile()
            self._resolve_budget_dependencies()
            self._resolve_policy_dependencies()
            self._finalize_numeric_fields()
            self._materialize()

            return self.args

        def _build_runtime_profile(self):
            runtime = self.runtime
            runtime['iterative_steps'] = 1
            runtime['prune_strategy'] = 'cfsp_ffn_flap'
            runtime['taylor'] = 'param_first'
            runtime['num_examples'] = 36
            runtime['dataset'] = 'dataset'
            runtime['calibration_seq_len'] = 256
            runtime['device'] = 'cuda'
            runtime['eval_device'] = 'cuda'
            runtime['extra_eval_batch_size'] = 4
            runtime['seed'] = 42

        def _build_hybrid_profile(self):
            hybrid = self.hybrid

            sample_budget = 50
            structural_floor = {
                'heads': 8,
                'mlp_neurons': 1024,
            }
            scoring_mix = {
                'mag': 0.25,
                'taylor': 0.50,
                'act': 0.25,
            }

            hybrid['hybrid_nsamples'] = sample_budget
            hybrid['hybrid_min_heads'] = structural_floor['heads']
            hybrid['hybrid_min_mlp_neurons'] = structural_floor['mlp_neurons']
            hybrid['hybrid_edge_protect'] = 0.20
            hybrid['hybrid_mag_weight'] = scoring_mix['mag']
            hybrid['hybrid_taylor_weight'] = scoring_mix['taylor']
            hybrid['hybrid_act_weight'] = scoring_mix['act']

        def _build_structural_profile(self):
            structural = self.structural

            layer_policy = {
                'early_protect': 0.20,
                'late_protect': 0.24,
                'middle_aggressive': 0.00,
            }
            attention_policy = {
                'keep_ratio': 0.995,
                'min_heads': 30,
            }
            rerank_policy = {
                'distill': True,
                'candidate_pool_ratio': 0.98,
                'calib_chunk_size': 6,
                'rerank_batch_size': 30,
                'rerank_max_tokens': 4096,
            }

            structural['cfsp_min_keep_ratio'] = 0.67
            structural['cfsp_budget_temperature'] = 3.2
            structural['cfsp_early_layer_protect'] = layer_policy['early_protect']
            structural['cfsp_late_layer_protect'] = layer_policy['late_protect']
            structural['cfsp_middle_layer_aggressive'] = layer_policy['middle_aggressive']
            structural['cfsp_attn_keep_ratio'] = attention_policy['keep_ratio']
            structural['cfsp_attn_min_heads'] = attention_policy['min_heads']
            structural['cfsp_distill_rerank'] = rerank_policy['distill']
            structural['cfsp_candidate_pool_ratio'] = rerank_policy['candidate_pool_ratio']
            structural['cfsp_calib_chunk_size'] = rerank_policy['calib_chunk_size']
            structural['cfsp_rerank_batch_size'] = rerank_policy['rerank_batch_size']
            structural['cfsp_rerank_max_tokens'] = rerank_policy['rerank_max_tokens']

        def _build_taylor_profile(self):
            taylor = self.taylor

            rerank_policy = {
                'enabled': True,
                'weight': 0.45,
            }
            finescore_policy = {
                'enabled': False,
                'weight': 0.30,
            }
            swap_policy = {
                'enabled': True,
                'topk': 384,
                'margin': 1.05,
                'mode': 'param_mix',
            }
            attention_swap_policy = {
                'enabled': True,
                'topk': 8,
                'margin': 1.03,
            }
            boundary_policy = {
                'primary': False,
                'window_ratio': 0.35,
            }
            late_parammix_policy = {
                'enabled': False,
                'start_ratio': 0.67,
            }

            taylor['cfsp_use_taylor_rerank'] = rerank_policy['enabled']
            taylor['cfsp_taylor_rerank_weight'] = rerank_policy['weight']
            taylor['cfsp_use_taylor_finescore'] = finescore_policy['enabled']
            taylor['cfsp_taylor_finescore_weight'] = finescore_policy['weight']
            taylor['cfsp_post_taylor_swap'] = swap_policy['enabled']
            taylor['cfsp_post_taylor_swap_topk'] = swap_policy['topk']
            taylor['cfsp_post_taylor_swap_margin'] = swap_policy['margin']
            taylor['cfsp_post_taylor_mode'] = swap_policy['mode']
            taylor['cfsp_attention_post_taylor_swap'] = attention_swap_policy['enabled']
            taylor['cfsp_attention_post_taylor_swap_topk'] = attention_swap_policy['topk']
            taylor['cfsp_attention_post_taylor_swap_margin'] = attention_swap_policy['margin']
            taylor['cfsp_boundary_taylor_primary'] = boundary_policy['primary']
            taylor['cfsp_boundary_taylor_window_ratio'] = boundary_policy['window_ratio']
            taylor['cfsp_late_layer_parammix'] = late_parammix_policy['enabled']
            taylor['cfsp_late_layer_parammix_start_ratio'] = late_parammix_policy['start_ratio']

        def _build_reconstruction_profile(self):
            reconstruction = self.reconstruction

            reconstruction_policy = {
                'enabled': True,
                'tokens': 1024,
                'layers': 8,
            }

            reconstruction['cfsp_struct_function_first'] = True
            reconstruction['cfsp_struct_importance_weight'] = 0.22
            reconstruction['cfsp_post_reconstruct_ffn'] = reconstruction_policy['enabled']
            reconstruction['cfsp_post_reconstruct_tokens'] = reconstruction_policy['tokens']
            reconstruction['cfsp_post_reconstruct_layers'] = reconstruction_policy['layers']
            reconstruction['cfsp_log_wanda_struct_scores'] = False
            reconstruction['cfsp_mlp_coarse_mode'] = 'disabled'
            reconstruction['cfsp_attn_score_mode'] = 'disabled'

        def _compose_profile(self):
            profile = self.profile
            for section in (
                self.runtime,
                self.hybrid,
                self.structural,
                self.taylor,
                self.reconstruction,
            ):
                profile.update(section)

        def _resolve_budget_dependencies(self):
            profile = self.profile

            profile['cfsp_candidate_pool_ratio'] = min(
                profile['cfsp_candidate_pool_ratio'],
                1.0,
            )
            profile['cfsp_min_keep_ratio'] = min(
                profile['cfsp_min_keep_ratio'],
                profile['cfsp_candidate_pool_ratio'],
            )

            profile['cfsp_rerank_max_tokens'] = max(
                profile['cfsp_rerank_max_tokens'],
                profile['cfsp_calib_chunk_size'] * 512 + 1024,
            )
            profile['cfsp_post_reconstruct_tokens'] = max(
                profile['cfsp_post_reconstruct_tokens'],
                profile['cfsp_post_reconstruct_layers'] * 128,
            )
            profile['cfsp_post_taylor_swap_topk'] = max(
                profile['cfsp_post_taylor_swap_topk'],
                profile['cfsp_attention_post_taylor_swap_topk'] * 32,
            )

        def _resolve_policy_dependencies(self):
            profile = self.profile

            if profile['cfsp_use_taylor_rerank']:
                profile['cfsp_taylor_rerank_weight'] = max(
                    profile['cfsp_taylor_rerank_weight'],
                    0.45,
                )

            if not profile['cfsp_use_taylor_finescore']:
                profile['cfsp_taylor_finescore_weight'] = min(
                    profile['cfsp_taylor_finescore_weight'],
                    0.30,
                )

            if profile['cfsp_post_taylor_swap']:
                profile['cfsp_post_taylor_swap_topk'] = max(
                    profile['cfsp_post_taylor_swap_topk'],
                    384,
                )

            if profile['cfsp_attention_post_taylor_swap']:
                profile['cfsp_attention_post_taylor_swap_topk'] = max(
                    profile['cfsp_attention_post_taylor_swap_topk'],
                    8,
                )

            if profile['cfsp_struct_function_first']:
                profile['cfsp_struct_importance_weight'] = 0.22

            if profile['cfsp_post_reconstruct_ffn']:
                profile['cfsp_post_reconstruct_tokens'] = max(
                    profile['cfsp_post_reconstruct_tokens'],
                    1024,
                )
                profile['cfsp_post_reconstruct_layers'] = max(
                    profile['cfsp_post_reconstruct_layers'],
                    8,
                )

        def _finalize_numeric_fields(self):
            integer_fields = (
                'iterative_steps',
                'num_examples',
                'calibration_seq_len',
                'extra_eval_batch_size',
                'seed',
                'hybrid_nsamples',
                'hybrid_min_heads',
                'hybrid_min_mlp_neurons',
                'cfsp_attn_min_heads',
                'cfsp_calib_chunk_size',
                'cfsp_rerank_batch_size',
                'cfsp_rerank_max_tokens',
                'cfsp_post_taylor_swap_topk',
                'cfsp_attention_post_taylor_swap_topk',
                'cfsp_post_reconstruct_tokens',
                'cfsp_post_reconstruct_layers',
            )

            for field_name in integer_fields:
                self.profile[field_name] = int(self.profile[field_name])

        def _materialize(self):
            for field_name, field_value in self.profile.items():
                setattr(self.args, field_name, field_value)

    return PruningProfileBuilder(args).build()