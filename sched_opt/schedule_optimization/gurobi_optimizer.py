"""Implement Gurobi optimizer."""

import logging
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from gurobipy import GRB  # type: ignore

from sched_opt.distance_calculation.distance_matrix_calculator import LocalizationData
from sched_opt.g0_utils.utils import ProblemDefinition
from sched_opt.schedule_optimization.flexible_solver import (
    BaseSolver,
    NamedConstraint,
    Objective,
    ObjectiveDirection,
    ORToolsSolver,
    SolverStatus,
    get_solution_value,
)

TOTAL_TIME_IN_DAY = 10 * 60

logger = logging.getLogger("Hola")


@dataclass
class Coordinates:
    """Determine a place coordinates."""

    lat: float
    lng: float


@dataclass
class Activity:
    """Hold elements for a destination."""

    name: str
    activity_duration: float
    # coords: Coordinates = Coordinates(0.0, 0.0)


@dataclass
class Trip:
    """Hold elements for a trip."""

    origin: str
    destination: str
    duration: float
    duration_incl_activity_length: float


@dataclass
class ModelInputs:
    """Gather all model inputs."""

    activities: dict[str, Activity]
    trips: dict[tuple[str, str], Trip]
    hotel: str


Variable = Any

AssignmentDict = dict[int, dict[tuple[str, str], Variable]]


def initiate_model() -> BaseSolver:
    """Initiate model."""
    solver_arguments = {
        "verbose": True,
    }
    return ORToolsSolver(**solver_arguments)


def create_model_inputs(loc: LocalizationData, model_definition: ProblemDefinition) -> ModelInputs:
    """Preprocess inputs for model."""
    activities = {place: Activity(name=place, activity_duration=loc.duration_dict[place][0]) for place in loc.places}
    trips = {
        (origin, destination): Trip(
            origin=origin,
            destination=destination,
            duration=loc.combination_distance_dict[(origin, destination)],
            duration_incl_activity_length=loc.combination_distance_dict[(origin, destination)]
            + activities[destination].activity_duration,
        )
        for origin, destination in loc.combination_distance_dict
    }
    return ModelInputs(activities, trips, hotel=loc.places[model_definition.hotel_index])


def generate_variables(
    solver: BaseSolver, model_inputs: ModelInputs, problem_definition: ProblemDefinition
) -> AssignmentDict:
    """Generate variables for model."""
    # Nested defaultdict
    assignment_vars: AssignmentDict = defaultdict(lambda: defaultdict(dict))

    num_vars = 0
    for day in range(problem_definition.num_days):
        for task_from, task_to in model_inputs.trips:
            assignment_vars[day][(task_from, task_to)] = solver.add_bool_var(
                name=f"assign[{day}][{task_from}][{task_to}]",
            )
            num_vars += 1
    return assignment_vars


def start_and_return_hotel(model_inputs: ModelInputs, assignments: AssignmentDict) -> Iterable[NamedConstraint]:
    """You start and return to hotel every day."""
    for day, possible_trips in assignments.items():
        yield NamedConstraint(
            name=f"must_leave_hotel[{str(day)}]",
            constraint=sum(
                variable for (from_task, to_task), variable in possible_trips.items() if from_task == model_inputs.hotel
            )
            == 1,
        )
        yield NamedConstraint(
            name=f"must_go_back_hotel[{str(day)}]",
            constraint=sum(
                variable for (from_task, to_task), variable in possible_trips.items() if to_task == model_inputs.hotel
            )
            == 1,
        )


def activity_after_activity(
    assignments: AssignmentDict,
) -> Iterable[NamedConstraint]:
    """After finishing an activity, tourists move to the next one, on the same day."""
    for day, activity_combinations in assignments.items():
        for task in {task_ for combi in activity_combinations for task_ in combi}:
            yield NamedConstraint(
                name=f"task_must_follow_task[{str(day)}][{str(task)}]",
                constraint=(
                    sum(variable for (task_from, task_to), variable in assignments[day].items() if task_to == task)
                    == sum(variable for (task_from, task_to), variable in assignments[day].items() if task_from == task)
                ),  # If (and only if) a mechanic goes to task, it also has to leave that task
            )


def all_activities_assigned(model_inputs: ModelInputs, assign: AssignmentDict) -> Iterable[NamedConstraint]:
    """All activities that are assigned."""
    for activity in model_inputs.activities.values():
        # due_day = (task.due_date - constants.TIME_EPSILON).date()  # Ensure 00:00:00 next day is mapped to due day
        # if due_day <= max(instance.schedule_days):
        yield NamedConstraint(
            name=f"mandatory_task_assigned[{activity}]",
            constraint=sum(
                variable
                for _, mechanic_day_assignments in assign.items()
                for (origin, _), variable in mechanic_day_assignments.items()
                if origin == activity.name
            )
            >= 1,  # It must be assigned
        )


def day_time_limit(model_inputs: ModelInputs, assign: AssignmentDict) -> Iterable[NamedConstraint]:
    """Next activity starts only after the previous one has finished."""
    for day in assign:
        yield NamedConstraint(
            name=f"activities_fit_within_day[{day}]",
            constraint=sum(
                variable * model_inputs.trips[(origin, destination)].duration_incl_activity_length
                for (origin, destination), variable in assign[day].items()
            )
            <= TOTAL_TIME_IN_DAY,
        )


#    # get all variables in a day
#    for day in assign:
#         yield NamedConstraint(
#             name=f"activities_fit_within_day[{day}]",
#             constraint
#         )


def generate_constraints(model: BaseSolver, model_inputs: ModelInputs, assignments: AssignmentDict) -> None:
    """Generate constraints."""
    constraint_generators = {
        "start_return_hotel": start_and_return_hotel(model_inputs, assignments),
        "start_next_where_you_finished": activity_after_activity(assignments),
        "all_activities_assigned": all_activities_assigned(model_inputs, assignments),
        "next_activity_starts_after_duration": day_time_limit(model_inputs, assignments),
    }

    for _, constraint_generator in constraint_generators.items():
        for constraint in constraint_generator:
            model.add_constraint(constraint)


def define_time_travelled(model_input: ModelInputs, assignments: AssignmentDict) -> Any:
    """Define time travelled."""
    return sum(
        model_input.trips[(origin, destination)].duration * assignment
        for _, day_tasks_for_mechanic in assignments.items()
        for (origin, destination), assignment in day_tasks_for_mechanic.items()
    )


def get_objective_function(solver: BaseSolver, model_inputs: ModelInputs, variables: AssignmentDict) -> dict[str, Any]:
    """Define objective function."""
    objective_parts = {"time_travelled": define_time_travelled(model_inputs, variables)}
    obj_vars = {}

    for obj_name, obj_expr in objective_parts.items():
        obj_vars[obj_name] = solver.add_num_var(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="obj_var_" + obj_name)
        solver.add_constraint(NamedConstraint(name="obj_const_" + obj_name, constraint=obj_vars[obj_name] == obj_expr))

    solver.set_objective(Objective(sum(obj_vars.values()), direction=ObjectiveDirection.MAXIMIZE))
    return obj_vars


def generate_objective_function(
    solver: BaseSolver, model_inputs: ModelInputs, variables: AssignmentDict
) -> dict[str, Any]:
    """Generate model objective function."""
    obj_elements = get_objective_function(solver, model_inputs, variables)
    return obj_elements


def define_model(model_inputs: ModelInputs, problem_definition: ProblemDefinition) -> tuple[BaseSolver, AssignmentDict]:
    """Define model."""
    model = initiate_model()
    variables = generate_variables(model, model_inputs, problem_definition)
    generate_constraints(model, model_inputs, variables)
    generate_objective_function(model, model_inputs, variables)
    return model, variables


def solve_model(solver: BaseSolver) -> float:
    """Solve defined model."""
    # Solve
    solver.set_params(
        # relative_mip_gap=config.solver.mip_gap_pct,
        time_limit=timedelta(minutes=1),
        # time_limit_no_improvement=(
        #     timedelta(seconds=config.solver.time_limit_no_improvement_in_seconds)
        # if config.solver.use_gurobi else None
        # ),
    )
    start_time = time.time()
    status = solver.solve()
    runtime = time.time() - start_time
    logger.info(f"Solve completed in {runtime:.2f} seconds")

    status_infeasible = status not in [SolverStatus.OPTIMAL, SolverStatus.FEASIBLE]
    number_solutions = solver.get_number_solutions()
    if status_infeasible or number_solutions == 0:
        logger.warning("Model failed to solve.")

    # if config.solver.lp_file_always or (config.solver.lp_file_when_failed and status_infeasible):
    # solver.export_model_as_lp_format("model.lp")
    # logger.info(f"Model error stored at model.lp.")

    # if config.solver.iis_when_failed and status_infeasible:
    #     _write_iis(lp_file_path)
    #     logger.info("IIS file stored.")

    if status_infeasible:
        logger.info("Optimizer did not complete.")

    logger.info("Model solved successfully!")
    logger.debug(f"Objective value is {solver.get_objective_value()}")

    return runtime


def extract_solution(model_inputs: ModelInputs, assignments: AssignmentDict) -> dict[int, list[str]]:
    """Extract solution from solver."""
    #  mechanic_tasks: dict[Mechanic, set[TaskId]] = {mechanic: set({}) for mechanic in instance.mechanics.values()}
    solution_dict: dict[int, list[str]] = {}

    for day, day_tasks in assignments.items():
        solution_dict[day] = []
        for (from_task, _), assignment in day_tasks.items():
            if get_solution_value(assignment) >= 1:
                solution_dict[day].append(from_task)
        solution_dict[day].append(model_inputs.hotel)

    return solution_dict


def run_new_solver(loc: LocalizationData, problem_definition: ProblemDefinition) -> dict[int, list[str]]:
    """Run new solver."""
    model_inputs = create_model_inputs(loc, problem_definition)
    solver, variables = define_model(model_inputs, problem_definition)
    solve_model(solver)
    return extract_solution(model_inputs, variables)
