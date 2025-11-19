# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Callable, Sequence, Union
import torch
import numpy as np
from functools import partial
import optree
import math

from sam3d_objects.model.backbone.generator.base import Base
from sam3d_objects.data.utils import right_broadcasting
from sam3d_objects.data.utils import tree_tensor_map, tree_reduce_unique
from sam3d_objects.model.backbone.generator.flow_matching.solver import (
    ODESolver,
    Euler,
    Midpoint,
    RungeKutta4,
    gradient,
    SDE,
)

# default sampler in flow matching
uniform_sampler = torch.rand


# https://arxiv.org/pdf/2403.03206
def lognorm_sampler(mean=0.0, std=1.0, **kwargs):
    logit = torch.randn(**kwargs) * std + mean
    return torch.nn.functional.sigmoid(logit)


# for backwards compatibility; please do not use this
def rev_lognorm_sampler(mean=0.0, std=1.0, **kwargs):
    logit = torch.randn(**kwargs) * std + mean
    return 1 - torch.nn.functional.sigmoid(logit)


# https://arxiv.org/pdf/2210.02747
class FlowMatching(Base):
    SOLVER_METHODS = {
        "euler": Euler,
        "midpoint": Midpoint,
        "rk4": RungeKutta4,
        "sde": SDE,
    }

    def __init__(
        self,
        reverse_fn: Callable,
        sigma_min: float = 0.0,  # 0. = rectifier flow
        inference_steps: int = 100,
        time_scale: float = 1000.0,  # scale [0,1]-time before passing to `reverse_fn`
        training_time_sampler_fn: Callable = partial(
            lognorm_sampler,
            mean=0,
            std=1,
        ),
        reversed_timestamp=False,
        rescale_t=1.0,
        loss_fn=partial(torch.nn.functional.mse_loss, reduction="mean"),
        loss_weights=1.0,
        solver_method: Union[str, ODESolver] = "euler",
        solver_kwargs: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.reverse_fn = reverse_fn
        self.sigma_min = sigma_min
        self.inference_steps = inference_steps
        self.time_scale = time_scale
        self.training_time_sampler_fn = training_time_sampler_fn
        self.reversed_timestamp = reversed_timestamp
        self.rescale_t = rescale_t
        self.loss_fn = loss_fn
        self.loss_weights = loss_weights
        self._solver_method, self._solver = self._get_solver(
            solver_method, solver_kwargs
        )

    def _get_solver(self, solver_method, solver_kwargs):
        if solver_method in FlowMatching.SOLVER_METHODS:
            solver = FlowMatching.SOLVER_METHODS[solver_method](**solver_kwargs)
        elif isinstance(solver_method, ODESolver):
            solver_method = f"custom[{solver_method.__class__.__name__}]"
            solver = solver_method
        else:
            raise ValueError(
                f"Invalid solver `{solver_method}`, should be in {set(self.SOLVER_METHODS.keys())} or an ODESolver instance"
            )
        return solver_method, solver

    def _generate_noise_tensor(self, x_shape, x_device):
        return torch.randn(
            x_shape,
            # generator=self.random_generator,
            device=x_device,
        )

    def _generate_noise(self, x_shape, x_device):
        def is_shape(maybe_shape):
            return isinstance(maybe_shape, Sequence) and all(
                (isinstance(s, int) and s >= 0) for s in maybe_shape
            )

        return optree.tree_map(
            partial(self._generate_noise_tensor, x_device=x_device),
            x_shape,
            is_leaf=is_shape,
            none_is_leaf=False,
        )

    def _generate_x0_tensor(self, x1: torch.Tensor):
        x0 = self._generate_noise_tensor(x1.shape, x1.device)
        return x0

    def _generate_xt_tensor(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        # equation (22)
        tb = right_broadcasting(t.to(x1.device), x1)
        x_t = (1 - (1 - self.sigma_min) * tb) * x0 + tb * x1

        return x_t

    def _generate_target_tensor(self, x0: torch.Tensor, x1: torch.Tensor):
        # equation (23)
        target = x1 - (1 - self.sigma_min) * x0

        return target

    def _generate_x0(self, x1):
        return tree_tensor_map(self._generate_x0_tensor, x1)

    def _generate_xt(self, x0, x1, t):
        return tree_tensor_map(
            partial(self._generate_xt_tensor, t=t),
            x0,
            x1,
        )

    def _generate_target(self, x0, x1):
        return tree_tensor_map(
            self._generate_target_tensor,
            x0,
            x1,
        )

    def _generate_t(self, x1):
        first_tensor = optree.tree_flatten(x1)[0][0]
        batch_size = first_tensor.shape[0]
        device = first_tensor.device

        t = self.training_time_sampler_fn(
            size=(batch_size,),
            generator=self.random_generator,
        ).to(device)

        return t

    def loss(self, x1: torch.Tensor, *args_conditionals, **kwargs_conditionals):
        t = self._generate_t(x1)
        x0 = self._generate_x0(x1)
        x_t = self._generate_xt(x0, x1, t)
        target = self._generate_target(x0, x1)

        prediction = self.reverse_fn(
            x_t,
            t * self.time_scale,
            *args_conditionals,
            **kwargs_conditionals,
        )

        # broadcast & and compute loss
        loss = optree.tree_broadcast_map(
            lambda fn, weight, pred, targ: weight * fn(pred, targ),
            self.loss_fn,
            self.loss_weights,
            prediction,
            target,
        )

        total_loss = sum(optree.tree_flatten(loss)[0])

        # Create detailed loss breakdown
        detail_losses = {
            "flow_matching_loss": total_loss,
        }
        if isinstance(loss, dict):
            detail_losses.update(loss)
        return total_loss, detail_losses

    def _prepare_t(self, steps=None):
        steps = self.inference_steps if steps is None else steps
        t_seq = torch.linspace(0, 1, steps + 1)

        if self.rescale_t:
            t_seq = t_seq / (1 + (self.rescale_t - 1) * (1 - t_seq))

        if self.reversed_timestamp:
            t_seq = 1 - t_seq

        return t_seq

    def generate_iter(
        self,
        x_shape,
        x_device,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        x_0 = self._generate_noise(x_shape, x_device)
        t_seq = self._prepare_t().to(x_device)

        for x_t, t in self._solver.solve_iter(
            self._generate_dynamics,
            x_0,
            t_seq,
            *args_conditionals,
            **kwargs_conditionals,
        ):
            yield t, x_t, ()

    def _generate_dynamics(
        self,
        x_t,
        t,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        return self.reverse_fn(x_t, t * self.time_scale, *args_conditionals, **kwargs_conditionals)

    def _log_p0(self, x0):
        x0 = self._tree_flatten(x0)
        inside_exp = -(x0**2).sum(dim=1) / 2
        return inside_exp - math.log(2 * math.pi) / 2 * x0.shape[1]

    def log_likelihood(
        self,
        x1,
        solver=None,
        steps=None,
        z_samples=1,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        device = tree_reduce_unique(lambda tensor: tensor.device, x1)
        # device = "cuda"
        t_seq = self._prepare_t(steps).to(device)
        t_seq = 1 - t_seq  # from x1 to x0
        solver = self._solver if solver is None else self._get_solver(solver)[1]

        x_0 = solver.solve(
            partial(self._log_likelihood_dynamics, device=device, z_samples=z_samples),
            {"x": x1, "log_p": 0.0},
            t_seq,
            *args_conditionals,
            **kwargs_conditionals,
        )

        log_p1 = x_0["log_p"] + self._log_p0(x_0["x"])

        return log_p1

    def _log_likelihood_dynamics(
        self,
        state,
        t,
        device,
        z_samples,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        t = torch.tensor([t * self.time_scale], device=device, dtype=torch.float32)
        x_t = state["x"]

        with torch.set_grad_enabled(True):
            tree_tensor_map(lambda x,: x.requires_grad_(True), x_t)
            velocity = self.reverse_fn(
                x_t,
                t,
                *args_conditionals,
                **kwargs_conditionals,
            )

            # compute the divergence estimate
            div = self._compute_hutchinson_divergence(velocity, x_t, z_samples)

        tree_tensor_map(lambda x,: x.requires_grad_(False), x_t)
        velocity = tree_tensor_map(lambda x: x.detach(), velocity)

        return {"x": velocity, "log_p": div.detach()}

    def _tree_flatten(self, tree):
        flat_x = tree_tensor_map(lambda x: x.flatten(start_dim=1), tree)
        flat_x, _ = optree.tree_flatten(
            flat_x,
            is_leaf=lambda x: isinstance(x, torch.Tensor),
        )
        flat_x = torch.cat(flat_x, dim=1)
        return flat_x

    def _compute_hutchinson_divergence(self, velocity, x_t, z_samples):
        flat_velocity = self._tree_flatten(velocity)
        flat_velocity = flat_velocity.unsqueeze(-1)

        z = torch.randn(
            flat_velocity.shape[:-1] + (z_samples,),
            dtype=flat_velocity.dtype,
            device=flat_velocity.device,
        )
        z = z < 0
        z = z * 2.0 - 1.0
        z = z / math.sqrt(z_samples)

        # compute Hutchinson divergence estimator E[z^T D_x(vt) z] = E[D_x(z^T vt) z)]
        vt_dot_z = torch.einsum("ijk,ijk->ik", flat_velocity, z)
        grad_vt_dot_z = [
            gradient(vt_dot_z[..., i], x_t, create_graph=(z_samples > 1))
            for i in range(z_samples)
        ]
        grad_vt_dot_z = [self._tree_flatten(g) for g in grad_vt_dot_z]
        grad_vt_dot_z = torch.stack(grad_vt_dot_z, dim=-1)
        div = torch.einsum("ijk,ijk->i", grad_vt_dot_z, z)
        return div


def _get_device(x):
    device = tree_reduce_unique(lambda tensor: tensor.device, x)
    return device


class ConditionalFlowMatching(FlowMatching):
    def generate_iter(
        self,
        x_shape,
        x_device,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        x_0 = self._generate_noise(x_shape, x_device)
        t_seq = self._prepare_t().to(x_device)

        noise_override = None
        if "noise_override" in kwargs_conditionals:
            noise_override = kwargs_conditionals["noise_override"]
            del kwargs_conditionals["noise_override"]
            if noise_override is not None:
                if type(x_0) == dict:
                    x_0.update(noise_override)
                else:
                    x_0 = noise_override

        for x_t, t in self._solver.solve_iter(
            self._generate_dynamics,
            x_0,
            t_seq,
            *args_conditionals,
            **kwargs_conditionals,
        ):
            if noise_override is not None:
                if type(noise_override) == dict:
                    x_t.update(noise_override)
                else:
                    x_t = noise_override
            yield t, x_t, ()
