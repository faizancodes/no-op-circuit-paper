"""Residual-stream caching and activation-patching for HF decoder LMs.

We target HuggingFace causal-LM models with a `.model.layers` ModuleList and a
final `.model.norm` (matches Qwen2/Qwen2.5, Llama, Mistral, CodeGemma, etc).

What we cache, per forward pass:

  resid_pre[L]  : input to layer L     — shape (B, T, D)
  resid_post[L] : output of layer L    — shape (B, T, D)
  resid_final   : output of final norm — shape (B, T, D)

`last_k` keeps memory bounded — we only retain the last K positions.

For activation patching, `patched_forward` substitutes the residual stream at a
given (layer, hook_point, position) with a supplied tensor before the rest of
the network runs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Literal

import torch
from torch import nn


HookPoint = Literal["resid_pre", "resid_post"]


def _get_layers(model: nn.Module) -> nn.ModuleList:
    """Resolve the decoder-layer ModuleList across common HF causal-LM layouts."""
    for path in (("model", "layers"), ("transformer", "h"), ("gpt_neox", "layers")):
        obj: object = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, nn.ModuleList):
            return obj
    raise AttributeError(
        "Could not locate the decoder layer list on the supplied model. "
        "Expected `.model.layers`, `.transformer.h`, or `.gpt_neox.layers`."
    )


def _get_final_norm(model: nn.Module) -> nn.Module | None:
    for path in (("model", "norm"), ("transformer", "ln_f")):
        obj: object = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, nn.Module):
            return obj
    return None


def _layer_output_hidden(output: object) -> torch.Tensor:
    """HF decoder layers return either a Tensor or a tuple whose first item is one."""
    if isinstance(output, tuple):
        return output[0]  # type: ignore[no-any-return]
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError(f"Unexpected layer output type: {type(output)}")


@dataclass
class ResidualCache:
    """Tensors keyed by hook point. Layer dim is leading."""

    resid_pre: torch.Tensor | None = None  # (L, B, T, D)
    resid_post: torch.Tensor | None = None  # (L, B, T, D)
    resid_final: torch.Tensor | None = None  # (B, T, D)
    last_k: int | None = None
    extras: dict[str, object] = field(default_factory=dict)


@contextmanager
def cache_forward(
    model: nn.Module,
    *,
    last_k: int | None = None,
    hook_points: tuple[HookPoint, ...] = ("resid_pre", "resid_post"),
    cache_final_norm: bool = True,
) -> Iterator[ResidualCache]:
    """Context manager that registers hooks and yields a cache to be populated.

    The yielded `ResidualCache` is filled in-place when the model is run inside
    the `with` block. Tensors are kept on the model device and detached.
    """
    layers = _get_layers(model)
    n_layers = len(layers)
    pre_buf: list[torch.Tensor | None] = [None] * n_layers
    post_buf: list[torch.Tensor | None] = [None] * n_layers
    final_buf: list[torch.Tensor | None] = [None]
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def slice_last(t: torch.Tensor) -> torch.Tensor:
        if last_k is None:
            return t.detach().clone()
        return t[..., -last_k:, :].detach().clone()

    if "resid_pre" in hook_points:
        for i, layer in enumerate(layers):
            def pre_hook(_mod, args, _kwargs=None, _i=i):
                # First positional arg is hidden_states.
                if args and isinstance(args[0], torch.Tensor):
                    pre_buf[_i] = slice_last(args[0])
            # with_kwargs=True keeps us compatible with HF layers that take
            # kwargs (most do under newer transformers versions).
            handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))

    if "resid_post" in hook_points:
        for i, layer in enumerate(layers):
            def post_hook(_mod, _args, output, _i=i):
                hidden = _layer_output_hidden(output)
                post_buf[_i] = slice_last(hidden)
            handles.append(layer.register_forward_hook(post_hook))

    final_norm = _get_final_norm(model) if cache_final_norm else None
    if final_norm is not None:
        def final_hook(_mod, _args, output):
            if isinstance(output, torch.Tensor):
                final_buf[0] = slice_last(output)
        handles.append(final_norm.register_forward_hook(final_hook))

    cache = ResidualCache(last_k=last_k)
    try:
        yield cache
    finally:
        for h in handles:
            h.remove()

    def _stack_or_none(buf: list[torch.Tensor | None]) -> torch.Tensor | None:
        if any(b is None for b in buf):
            return None
        return torch.stack(buf, dim=0)  # type: ignore[arg-type]

    if "resid_pre" in hook_points:
        cache.resid_pre = _stack_or_none(pre_buf)
    if "resid_post" in hook_points:
        cache.resid_post = _stack_or_none(post_buf)
    if final_norm is not None:
        cache.resid_final = final_buf[0]


@dataclass(frozen=True)
class ResidualPatch:
    layer_idx: int
    hook_point: HookPoint
    position: int          # absolute position in the input sequence
    value: torch.Tensor    # shape (D,) or (B, D) — broadcast over batch


@dataclass(frozen=True)
class SteeringInjection:
    """Add `alpha * direction` to the residual at (layer, hook_point, position)."""
    layer_idx: int
    hook_point: HookPoint
    position: int
    direction: torch.Tensor    # shape (D,)
    alpha: float


@contextmanager
def steered_forward(
    model: nn.Module,
    injections: list[SteeringInjection],
) -> Iterator[None]:
    """Run the model with one or more additive residual-stream injections.

    Unlike `patched_forward`, this ADDS `alpha * direction` to the existing
    residual at (layer, hook_point, position) instead of replacing it. Cheap
    and ideal for dose-response sweeps over a steering coefficient.
    """
    layers = _get_layers(model)
    handles: list[torch.utils.hooks.RemovableHandle] = []
    by_layer_pre: dict[int, list[SteeringInjection]] = {}
    by_layer_post: dict[int, list[SteeringInjection]] = {}
    for inj in injections:
        bucket = by_layer_pre if inj.hook_point == "resid_pre" else by_layer_post
        bucket.setdefault(inj.layer_idx, []).append(inj)

    for layer_idx, injs in by_layer_pre.items():
        def pre_hook(_mod, args, kwargs=None, _injs=injs):
            if not args or not isinstance(args[0], torch.Tensor):
                return None
            hidden = args[0].clone()
            for inj in _injs:
                v = inj.direction.to(hidden.dtype).to(hidden.device)
                hidden[:, inj.position, :] = hidden[:, inj.position, :] + inj.alpha * v
            new_args = (hidden,) + args[1:]
            return (new_args, kwargs) if kwargs is not None else new_args
        handles.append(
            layers[layer_idx].register_forward_pre_hook(pre_hook, with_kwargs=True)
        )

    for layer_idx, injs in by_layer_post.items():
        def post_hook(_mod, _args, output, _injs=injs):
            hidden = _layer_output_hidden(output).clone()
            for inj in _injs:
                v = inj.direction.to(hidden.dtype).to(hidden.device)
                hidden[:, inj.position, :] = hidden[:, inj.position, :] + inj.alpha * v
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        handles.append(layers[layer_idx].register_forward_hook(post_hook))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def patched_forward(
    model: nn.Module,
    patches: list[ResidualPatch],
) -> Iterator[None]:
    """Run the model with one or more residual-stream substitutions.

    Each patch replaces the residual stream at a specific (layer, hook_point,
    position) with the supplied value. Stacks naturally — apply multiple to
    intervene at several sites in a single forward pass.
    """
    layers = _get_layers(model)
    handles: list[torch.utils.hooks.RemovableHandle] = []
    by_layer_pre: dict[int, list[ResidualPatch]] = {}
    by_layer_post: dict[int, list[ResidualPatch]] = {}
    for p in patches:
        bucket = by_layer_pre if p.hook_point == "resid_pre" else by_layer_post
        bucket.setdefault(p.layer_idx, []).append(p)

    for layer_idx, ps in by_layer_pre.items():
        def pre_hook(_mod, args, kwargs=None, _ps=ps):
            if not args or not isinstance(args[0], torch.Tensor):
                return None
            hidden = args[0]
            for patch in _ps:
                # broadcast value to (B, D) if needed
                v = patch.value
                if v.dim() == 1:
                    v = v.unsqueeze(0).expand(hidden.shape[0], -1)
                hidden = hidden.clone()
                hidden[:, patch.position, :] = v.to(hidden.dtype).to(hidden.device)
            new_args = (hidden,) + args[1:]
            return (new_args, kwargs) if kwargs is not None else new_args
        handles.append(
            layers[layer_idx].register_forward_pre_hook(pre_hook, with_kwargs=True)
        )

    for layer_idx, ps in by_layer_post.items():
        def post_hook(_mod, _args, output, _ps=ps):
            hidden = _layer_output_hidden(output)
            hidden = hidden.clone()
            for patch in _ps:
                v = patch.value
                if v.dim() == 1:
                    v = v.unsqueeze(0).expand(hidden.shape[0], -1)
                hidden[:, patch.position, :] = v.to(hidden.dtype).to(hidden.device)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        handles.append(layers[layer_idx].register_forward_hook(post_hook))

    try:
        yield
    finally:
        for h in handles:
            h.remove()
