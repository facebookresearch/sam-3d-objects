# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Multi-view multidiffusion utilities for SAM 3D Objects
Adapted from TRELLIS implementation, adapted for SAM 3D Objects' two-stage structure
"""
from contextlib import contextmanager
from typing import List, Literal
import torch
from loguru import logger


@contextmanager
def inject_generator_multi_view(
    generator,
    num_views: int,
    num_steps: int,
    mode: Literal['stochastic', 'multidiffusion'] = 'multidiffusion',
):
    """
    Inject multi-view support into generator
    
    Args:
        generator: SAM 3D Objects generator (ss_generator or slat_generator)
        num_views: Number of views
        num_steps: Number of inference steps
        mode: 'stochastic' or 'multidiffusion'
    
    Yields:
        None
    """
    original_dynamics = generator._generate_dynamics
    
    if mode == 'stochastic':
        if num_views > num_steps:
            logger.warning(
                f"Warning: number of views ({num_views}) is greater than number of steps ({num_steps}). "
                "This may lead to performance degradation."
            )
        
        cond_indices = (torch.arange(num_steps) % num_views).tolist()
        cond_idx_counter = [0]
        
        def _new_dynamics_stochastic(x_t, t, *args_conditionals, **kwargs_conditionals):
            """Stochastic mode: select one view per time step"""
            cond_idx = cond_indices[cond_idx_counter[0] % len(cond_indices)]
            cond_idx_counter[0] += 1
            
            if len(args_conditionals) > 0:
                cond_tokens = args_conditionals[0]
                if isinstance(cond_tokens, (list, tuple)):
                    cond_i = cond_tokens[cond_idx:cond_idx+1] if isinstance(cond_tokens[0], torch.Tensor) else [cond_tokens[cond_idx]]
                    new_args = (cond_i,) + args_conditionals[1:]
                elif isinstance(cond_tokens, torch.Tensor) and cond_tokens.shape[0] == num_views:
                    cond_i = cond_tokens[cond_idx:cond_idx+1]
                    new_args = (cond_i,) + args_conditionals[1:]
                else:
                    new_args = args_conditionals
            else:
                new_args = args_conditionals
            
            return original_dynamics(x_t, t, *new_args, **kwargs_conditionals)
        
        generator._generate_dynamics = _new_dynamics_stochastic
        
    elif mode == 'multidiffusion':
        def _new_dynamics_multidiffusion(x_t, t, *args_conditionals, **kwargs_conditionals):
            """Multidiffusion mode: fuse predictions from all views at each time step"""
            cond_idx = 0
            if len(args_conditionals) > 0:
                if isinstance(args_conditionals[0], (int, float)) or (isinstance(args_conditionals[0], torch.Tensor) and args_conditionals[0].numel() == 1):
                    cond_idx = 1
            
            if len(args_conditionals) > cond_idx:
                cond_tokens = args_conditionals[cond_idx]
                
                if not hasattr(_new_dynamics_multidiffusion, '_logged_cond_shape'):
                    logger.info(f"[Multidiffusion] args_conditionals length: {len(args_conditionals)}")
                    logger.info(f"[Multidiffusion] cond_idx: {cond_idx}")
                    if isinstance(cond_tokens, torch.Tensor):
                        logger.info(f"[Multidiffusion] Condition tokens shape: {cond_tokens.shape}")
                    elif isinstance(cond_tokens, (list, tuple)):
                        logger.info(f"[Multidiffusion] Condition tokens type: {type(cond_tokens)}, length: {len(cond_tokens)}")
                        if len(cond_tokens) > 0 and isinstance(cond_tokens[0], torch.Tensor):
                            logger.info(f"[Multidiffusion] First condition token shape: {cond_tokens[0].shape}")
                    else:
                        logger.info(f"[Multidiffusion] Condition tokens type: {type(cond_tokens)}")
                    _new_dynamics_multidiffusion._logged_cond_shape = True
                
                if isinstance(cond_tokens, (list, tuple)):
                    view_conditions = cond_tokens
                elif isinstance(cond_tokens, torch.Tensor) and cond_tokens.shape[0] == num_views:
                    view_conditions = []
                    for i in range(num_views):
                        view_cond = cond_tokens[i]
                        view_conditions.append(view_cond)
                else:
                    logger.warning(f"Condition tokens shape {cond_tokens.shape if isinstance(cond_tokens, torch.Tensor) else type(cond_tokens)} not organized by views, using same condition for all views")
                    view_conditions = [cond_tokens] * num_views
                
                preds = []
                for view_idx in range(num_views):
                    view_cond = view_conditions[view_idx]
                    if cond_idx < len(args_conditionals):
                        new_args = args_conditionals[:cond_idx] + (view_cond,) + args_conditionals[cond_idx+1:]
                    else:
                        new_args = args_conditionals + (view_cond,)
                    pred = original_dynamics(x_t, t, *new_args, **kwargs_conditionals)
                    preds.append(pred)
                
                if not hasattr(_new_dynamics_multidiffusion, '_logged_shape'):
                    if isinstance(x_t, dict):
                        logger.info(f"[Multidiffusion] Latent shape (dict): {[(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in x_t.items()]}")
                    elif isinstance(x_t, (list, tuple)):
                        logger.info(f"[Multidiffusion] Latent shape (tuple/list): {[v.shape if isinstance(v, torch.Tensor) else type(v) for v in x_t]}")
                    else:
                        logger.info(f"[Multidiffusion] Latent shape: {x_t.shape if isinstance(x_t, torch.Tensor) else type(x_t)}")
                    
                    if isinstance(preds[0], dict):
                        logger.info(f"[Multidiffusion] Pred shape (dict): {[(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in preds[0].items()]}")
                    elif isinstance(preds[0], (list, tuple)):
                        logger.info(f"[Multidiffusion] Pred shape (tuple/list): {[v.shape if isinstance(v, torch.Tensor) else type(v) for v in preds[0]]}")
                    else:
                        logger.info(f"[Multidiffusion] Pred shape: {preds[0].shape if isinstance(preds[0], torch.Tensor) else type(preds[0])}")
                    logger.info(f"[Multidiffusion] Number of views: {num_views}, fusing {len(preds)} predictions")
                    _new_dynamics_multidiffusion._logged_shape = True
                
                if isinstance(preds[0], dict):
                    fused_pred = {}
                    for key in preds[0].keys():
                        fused_pred[key] = torch.stack([p[key] for p in preds]).mean(dim=0)
                    return fused_pred
                elif isinstance(preds[0], (list, tuple)):
                    fused_pred = tuple(
                        torch.stack([p[i] for p in preds]).mean(dim=0)
                        for i in range(len(preds[0]))
                    )
                    return fused_pred
                else:
                    fused_pred = torch.stack(preds).mean(dim=0)
                    return fused_pred
            else:
                return original_dynamics(x_t, t, *args_conditionals, **kwargs_conditionals)
        
        generator._generate_dynamics = _new_dynamics_multidiffusion
        
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    try:
        yield
    finally:
        generator._generate_dynamics = original_dynamics

