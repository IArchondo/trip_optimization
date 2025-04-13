"""Implementation for solver."""

import abc
import enum
import pathlib
import typing
from dataclasses import dataclass
from datetime import datetime, timedelta

import gurobipy as gp  # type: ignore
from ortools.linear_solver import pywraplp  # type: ignore

# TODO: Make variable (and possibly LinearExpression) more specific
Variable = typing.Any
Constraint = typing.Any
LinearExpression = typing.Any


@dataclass(frozen=True, eq=True)
class NamedConstraint:
    """A constraint with a name."""

    name: str
    constraint: LinearExpression | bool


class ObjectiveDirection(enum.Enum):
    """Direction of the objective."""

    MINIMIZE = enum.auto()
    MAXIMIZE = enum.auto()


class Objective:
    """Objective function with direction."""

    def __init__(self, expr: LinearExpression, direction: ObjectiveDirection = ObjectiveDirection.MINIMIZE) -> None:
        """Objective function with direction."""
        self.expr = expr
        self.direction = direction


class SolverStatus(enum.Enum):
    """Status of the solver."""

    OPTIMAL = enum.auto()
    FEASIBLE = enum.auto()
    INFEASIBLE = enum.auto()
    UNBOUNDED = enum.auto()
    NOT_STARTED = enum.auto()
    UNKNOWN = enum.auto()


def get_solution_value(variable: Variable) -> float:
    """Get solution value of the variable."""
    try:
        # Gurobi
        return float(variable.X)
    except AttributeError:
        pass
    try:
        # ORtools
        return float(variable.solution_value())
    except AttributeError:
        pass

    raise ValueError(f"Unknown variable type: {type(variable)}")


class BaseSolver(abc.ABC):
    """Base class for solvers."""

    def __init__(self, verbose: bool = False) -> None:
        """Define base class for solvers."""
        self.verbose = verbose
        self.status = SolverStatus.NOT_STARTED

    @abc.abstractmethod
    def add_bool_var(self, name: str) -> Variable:
        """Add boolean variable."""

    @abc.abstractmethod
    def add_num_var(self, lb: float, ub: float, name: str) -> Variable:
        """Add numerical variable."""

    @abc.abstractmethod
    def variable_count(self) -> int:
        """Return number of variables."""

    @abc.abstractmethod
    def constraint_count(self) -> int:
        """Return number of constraints."""

    @abc.abstractmethod
    def add_constraint(self, constraint: NamedConstraint) -> None:
        """Add constraint."""

    @abc.abstractmethod
    def set_objective(self, objective: Objective) -> None:
        """Add objective."""

    @abc.abstractmethod
    def solve(self) -> SolverStatus:
        """Solve the model."""

    @abc.abstractmethod
    def get_objective_value(self) -> float:
        """Return value of the objective."""

    @abc.abstractmethod
    def get_best_bound(self) -> float:
        """Return best bound of the solution."""

    @abc.abstractmethod
    def get_status(self) -> SolverStatus:
        """Return status of the solver."""
        pass

    @abc.abstractmethod
    def set_params(
        self,
        relative_mip_gap: float | None = None,
        time_limit: timedelta | None = None,
        time_limit_no_improvement: timedelta | None = None,
    ) -> None:
        """Set parameters for the solver."""
        pass

    @abc.abstractmethod
    def export_model_as_lp_format(self, path: pathlib.Path) -> None:
        """Export model as LP format."""
        pass

    @abc.abstractmethod
    def get_number_solutions(self) -> int:
        """Return the number of solutions found in the optimization process."""
        pass


class GurobiSolver(BaseSolver):
    """Solver using GurobiPy."""

    def __init__(self, verbose: bool = False) -> None:
        """Solver using GurobiPy."""
        super().__init__(verbose)
        self.model: gp.Model = gp.Model()
        if not self.verbose:
            self.model.setParam("OutputFlag", 0)
        self.callback_func: typing.Callable[[gp.Model, int], None] | None = None

    def add_bool_var(self, name: str) -> Variable:
        """Add boolean variable."""
        return self.model.addVar(vtype="B", name=name)

    def add_num_var(self, lb: float, ub: float, name: str) -> Variable:
        """Add continuous variable."""
        return self.model.addVar(vtype="C", lb=lb, ub=ub, name=name)

    def add_int_var(self, lb: int, ub: int, name: str) -> Variable:
        """Add integer variable."""
        return self.model.addVar(vtype="I", lb=lb, ub=ub, name=name)

    def add_constraint(self, constraint: NamedConstraint) -> Constraint:
        """Add constraint."""
        return self.model.addConstr(constraint.constraint, name=constraint.name)

    def variable_count(self) -> int:
        """Return number of variables."""
        self.model.update()
        return len(self.model.getVars())

    def constraint_count(self) -> int:
        """Return number of constraints."""
        self.model.update()
        return len(self.model.getConstrs())

    def set_objective(self, objective: Objective) -> None:
        """Add objective."""
        gurobi_direction = {
            ObjectiveDirection.MINIMIZE: gp.GRB.MINIMIZE,
            ObjectiveDirection.MAXIMIZE: gp.GRB.MAXIMIZE,
        }[objective.direction]

        self.model.setObjective(objective.expr, gurobi_direction)

    def set_params(
        self,
        relative_mip_gap: float | None = None,
        time_limit: timedelta | None = None,
        time_limit_no_improvement: timedelta | None = None,
    ) -> None:
        """Set parameters for the solver."""
        if relative_mip_gap is not None:
            self.model.setParam(gp.GRB.Param.MIPGap, relative_mip_gap)
        if time_limit is not None:
            self.model.setParam(gp.GRB.Param.TimeLimit, time_limit.total_seconds())
        if time_limit_no_improvement is not None:
            self.callback_func = gurobi_callback(time_limit=time_limit_no_improvement)

    def solve(self) -> SolverStatus:
        """Solve the model."""
        self.model._current_best_objective = 0
        self.model._best_objective_since = datetime.now()
        if self.callback_func is not None:
            self.model.optimize(self.callback_func)  # type: ignore
        else:
            self.model.optimize()

        # stopping_reason = GUROBI_STATUS[self.model.status]

        self.status = {
            gp.GRB.OPTIMAL: SolverStatus.OPTIMAL,
            gp.GRB.SUBOPTIMAL: SolverStatus.FEASIBLE,
            gp.GRB.ITERATION_LIMIT: SolverStatus.FEASIBLE,
            gp.GRB.TIME_LIMIT: SolverStatus.FEASIBLE,
            gp.GRB.USER_OBJ_LIMIT: SolverStatus.FEASIBLE,
            gp.GRB.INTERRUPTED: SolverStatus.FEASIBLE,
            gp.GRB.INFEASIBLE: SolverStatus.INFEASIBLE,
            gp.GRB.UNBOUNDED: SolverStatus.UNBOUNDED,
        }.get(self.model.status, SolverStatus.UNKNOWN)

        return self.status

    def export_model_as_lp_format(self, path: pathlib.Path) -> None:
        """Export model."""
        self.model.write(str(path.absolute()))

    def get_objective_value(self) -> float:
        """Return value of the objective."""
        return float(self.model.objVal)

    def get_best_bound(self) -> float:
        """Return best bound."""
        return float(self.model.objBound)

    def get_status(self) -> SolverStatus:
        """Return status of the solver."""
        return self.status

    def get_number_solutions(self) -> int:
        """Return the number of solutions found in the optimization process."""
        return int(self.model.SolCount)


def gurobi_callback(time_limit: timedelta) -> typing.Callable[[gp.Model, int], None]:
    """Define callback function for Gurobi.

    Stops the model if there has not been a material improvement in the objective in the last `time_limit` seconds.

    See https://www.gurobi.com/documentation/current/refman/py_cb_s.html for more information.
    """

    def actual_callback(model: gp.Model, where: int) -> None:
        """Actual callback function."""
        # Check if the new MIP solution is better than the previous one
        if where == gp.GRB.Callback.MIPSOL:
            model._best_objective_since = datetime.now()
        # Status of callback where you are in the MIP process but didn't find a better solution
        elif where in [gp.GRB.Callback.MIP, gp.GRB.Callback.MIPNODE]:
            sol_count = (
                model.cbGet(gp.GRB.Callback.MIP_SOLCNT)
                if where == gp.GRB.Callback.MIP
                else model.cbGet(gp.GRB.Callback.MIPNODE_SOLCNT)
            )
            time_no_improvement = datetime.now() - model._best_objective_since
            # Terminate if objective has not improved for `time_limit` seconds and there is at least one solution
            if sol_count > 0 and time_no_improvement > time_limit:
                model.terminate()

    return actual_callback


class ORToolsSolver(BaseSolver):
    """Solver using OR-Tools."""

    OPTIMAL = pywraplp.Solver.OPTIMAL
    FEASIBLE = pywraplp.Solver.FEASIBLE
    INFEASIBLE = pywraplp.Solver.INFEASIBLE
    UNBOUNDED = pywraplp.Solver.UNBOUNDED

    def __init__(self, verbose: bool = False) -> None:
        """Solver using OR-Tools."""
        super().__init__(verbose)
        self.solver = pywraplp.Solver("Optimizer", pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)
        self.solver_params = pywraplp.MPSolverParameters()
        self.status = SolverStatus.NOT_STARTED

        if self.verbose:
            self.solver.EnableOutput()

    def add_bool_var(self, name: str) -> Variable:
        """Add boolean variable."""
        return self.solver.BoolVar(name)

    def add_num_var(self, lb: float, ub: float, name: str) -> Variable:
        """Add numerical variable."""
        return self.solver.NumVar(lb, ub, name)

    def add_constraint(self, constraint: NamedConstraint) -> Constraint:
        """Add constraint."""
        return self.solver.Add(constraint.constraint, name=constraint.name)

    def variable_count(self) -> int:
        """Return number of variables."""
        return int(self.solver.NumVariables())

    def constraint_count(self) -> int:
        """Return number of constraints."""
        return int(self.solver.NumConstraints())

    def set_objective(self, objective: Objective) -> None:
        """Add objective."""
        if objective.direction == ObjectiveDirection.MINIMIZE:
            self.solver.Minimize(objective.expr)
        else:
            self.solver.Maximize(objective.expr)

    def set_params(
        self,
        relative_mip_gap: float | None = None,
        time_limit: timedelta | None = None,
        time_limit_no_improvement: timedelta | None = None,
    ) -> None:
        """Set parameters for the solver."""
        if relative_mip_gap is not None:
            self.solver_params.SetDoubleParam(self.solver_params.RELATIVE_MIP_GAP, relative_mip_gap)
        if time_limit is not None:
            self.solver.set_time_limit(int(time_limit.total_seconds() * 1000))
        if time_limit_no_improvement is not None:
            raise NotImplementedError("No improvement time limit not supported for OR-Tools.")

    def solve(self) -> SolverStatus:
        """Solve the model."""
        ort_status = self.solver.Solve(self.solver_params)

        self.status = {
            self.OPTIMAL: SolverStatus.OPTIMAL,
            self.FEASIBLE: SolverStatus.FEASIBLE,
            self.INFEASIBLE: SolverStatus.INFEASIBLE,
            self.UNBOUNDED: SolverStatus.UNBOUNDED,
        }[ort_status]
        return self.status

    def export_model_as_lp_format(self, path: pathlib.Path) -> None:
        """Export model."""
        mps_text = self.solver.ExportModelAsLpFormat(obfuscated=False)

        with path.open("w") as output_file:
            output_file.write(mps_text)

    def get_objective_value(self) -> float:
        """Return value of the objective."""
        return float(self.solver.Objective().Value())

    def get_best_bound(self) -> float:
        """Return best bound."""
        return float(self.solver.Objective().BestBound())

    def get_status(self) -> SolverStatus:
        """Return status of the solver."""
        return self.status

    def get_number_solutions(self) -> int:
        """Return the number of solutions found in the optimization process."""
        # TODO implement this for ORtools
        # logger.warning("This function is not implemented for ORToolsSolver, always will return 1.")
        return 1
