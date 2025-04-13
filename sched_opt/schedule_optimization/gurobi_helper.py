"""Hold gurobi things."""

# line too long
import dataclasses
from collections import defaultdict

from gurobipy import GRB  # type: ignore


@dataclasses.dataclass
class GurobiStatus:
    """Status of the Gurubi solver."""

    name: str
    description: str


GUROBI_STATUS: defaultdict[int, GurobiStatus] = defaultdict(
    lambda: GurobiStatus("UNKNOWN", "Unknown reason"),
    {
        GRB.LOADED: GurobiStatus(
            name="LOADED", description="Model is loaded, but no solution information is available."
        ),
        GRB.OPTIMAL: GurobiStatus(
            name="OPTIMAL",
            description="Model was solved to optimality (subject to tolerances), and an optimal solution is available.",
        ),
        GRB.INFEASIBLE: GurobiStatus(name="INFEASIBLE", description="Model was proven to be infeasible."),
        GRB.INF_OR_UNBD: GurobiStatus(
            name="INF_OR_UNBD",
            description="Model was proven to be either infeasible or unbounded.",
        ),
        GRB.UNBOUNDED: GurobiStatus(
            name="UNBOUNDED",
            description="Model was proven to be unbounded. ",
        ),
        GRB.CUTOFF: GurobiStatus(
            name="CUTOFF",
            description="Optimal objective was proven to be worse than the value specified in the Cutoff parameter. ",
        ),
        GRB.ITERATION_LIMIT: GurobiStatus(
            name="ITERATION_LIMIT",
            description="Optimization terminated.",
        ),
        GRB.NODE_LIMIT: GurobiStatus(
            name="NODE_LIMIT",
            description="Optimization terminated.",
        ),
        GRB.TIME_LIMIT: GurobiStatus(
            name="TIME_LIMIT",
            description="Optimization terminated.",
        ),
        GRB.SOLUTION_LIMIT: GurobiStatus(
            name="SOLUTION_LIMIT",
            description="Optimization terminated.",
        ),
        GRB.INTERRUPTED: GurobiStatus(name="INTERRUPTED", description="Optimization was terminated by the user."),
        GRB.NUMERIC: GurobiStatus(name="NUMERIC", description="Optimization was terminated."),
        GRB.SUBOPTIMAL: GurobiStatus(
            name="SUBOPTIMAL",
            description="Unable to satisfy optimality tolerances.",
        ),
        GRB.INPROGRESS: GurobiStatus(
            name="INPROGRESS",
            description="An asynchronous optimization call was made.",
        ),
        GRB.USER_OBJ_LIMIT: GurobiStatus(
            name="USER_OBJ_LIMIT",
            description="User specified an objective limit.",
        ),
        GRB.WORK_LIMIT: GurobiStatus(
            name="WORK_LIMIT",
            description="Optimization terminated.",
        ),
        GRB.MEM_LIMIT: GurobiStatus(
            name="MEM_LIMIT",
            description="Optimization terminated.",
        ),
    },
)
