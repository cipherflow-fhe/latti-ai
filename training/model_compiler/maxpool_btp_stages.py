"""Plan Bootstrap boundaries for polynomial MaxPool2D."""

from dataclasses import dataclass
from maxpool_plan import build_maxpool_plan, build_maxpool_stage_plan


@dataclass(frozen=True)
class MaxPoolStage:
    index: int
    operations: tuple[str, ...]
    level_cost: int
    live_tensors: int


def build_maxpool_btp_stages(layer, input_shape, max_level):
    """Plan logical stages; does not mutate the compiler DAG."""

    plan = build_maxpool_plan(layer, input_shape)

    if (
        plan["kernel_shape"] != (3, 3)
        or plan["stride"] != (2, 2)
        or plan["padding"] != (1, 1)
    ):
        raise NotImplementedError(
            "Only MaxPool2D(3, 2, 1) is supported"
        )

    if not isinstance(max_level, int) or max_level <= 0:
        raise ValueError("Invalid Bootstrap max_level")

    operation_plan = build_maxpool_stage_plan(plan)

    stages = []
    current_ops = []
    current_cost = 0
    current_live = 9

    for stage in operation_plan["stages"]:
        name = stage["name"]
        cost = stage["level_cost"]
        live = stage["live_tensors"]
        if cost > max_level:
            raise ValueError(
                f"Operation {name} requires {cost} levels, "
                f"but max_level={max_level}"
            )

        if current_ops and current_cost + cost > max_level:
            stages.append(
                MaxPoolStage(
                    index=len(stages),
                    operations=tuple(current_ops),
                    level_cost=current_cost,
                    live_tensors=current_live,
                )
            )
            current_ops = []
            current_cost = 0

        current_ops.append(name)
        current_cost += cost
        current_live = live

    if current_ops:
        stages.append(
            MaxPoolStage(
                index=len(stages),
                operations=tuple(current_ops),
                level_cost=current_cost,
                live_tensors=current_live,
            )
        )

    return {
        "stages": tuple(stages),
        "bootstrap_count": len(stages) - 1,
        "total_level_cost": sum(s.level_cost for s in stages),
        "input_shape": plan["input_shape"],
        "output_shape": plan["output_shape"],
    }
