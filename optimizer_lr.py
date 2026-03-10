from __future__ import annotations

from typing import Any, MutableMapping, Sequence


def make_param_group(
    params: Sequence[Any],
    *,
    weight_decay: float,
    lr_scale: float = 1.0,
) -> dict[str, Any]:
    return {
        "params": params,
        "weight_decay": weight_decay,
        "lr_scale": lr_scale,
    }


def apply_param_group_lrs(param_groups: Sequence[MutableMapping[str, Any]], base_lr: float) -> None:
    for param_group in param_groups:
        param_group["lr"] = base_lr * float(param_group.get("lr_scale", 1.0))
