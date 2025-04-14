"""Define new visualizer."""

import logging

import matplotlib.pyplot as plt

from sched_opt.schedule_optimization.gurobi_optimizer import ModelInputs

logger = logging.getLogger("ScheduleVisualizer")

map_image = plt.imread("01_map_data/map_input_1.png")
BBox = (-74.05, -73.9209, 40.6106, 40.8)


def generate_day_route_output(
    model_inputs: ModelInputs, solution: dict[int, list[str]], day_to_plot: int, current_run: str
) -> None:
    """Generate graph for one day."""
    route_list = [
        (solution[day_to_plot][i], solution[day_to_plot][i + 1]) for i in range(len(solution[day_to_plot]) - 1)
    ]

    lngs = [model_inputs.activities[act].coords.lng for act in solution[day_to_plot]]
    lats = [model_inputs.activities[act].coords.lat for act in solution[day_to_plot]]

    ## plot limits
    xmin = min(lngs) - 0.01
    xmax = max(lngs) + 0.01
    ymin = min(lats) - 0.01
    ymax = max(lats) + 0.01

    hotel_x = model_inputs.activities[model_inputs.hotel].coords.lng
    hotel_y = model_inputs.activities[model_inputs.hotel].coords.lat

    _, ax = plt.subplots(figsize=(16, 8.3))

    destination_list = []
    for ix, destination in enumerate(solution[day_to_plot]):
        ax.annotate(
            f"{ix}-{destination}",
            (
                model_inputs.activities[destination].coords.lng,
                model_inputs.activities[destination].coords.lat,
            ),
        )
        destination_list.append(f"{ix}-{destination}")

        ax.scatter(
            model_inputs.activities[destination].coords.lng,
            model_inputs.activities[destination].coords.lat,
            zorder=2,
            c="red",
            s=120,
        )
        ax.scatter(
            hotel_x,
            hotel_y,
            zorder=3,
            facecolors="yellow",
            edgecolors="red",
            linewidth=2,
            s=140,
        )
        ax.set_title(f"Schedule for day {day_to_plot}", fontsize=24, fontweight="bold")
        ax.set_xlim(BBox[0], BBox[1])
        ax.set_ylim(BBox[2], BBox[3])
        ax.imshow(map_image, extent=BBox, aspect="equal", zorder=0)

    for origin, destination in route_list:
        point1 = [model_inputs.activities[origin].coords.lng, model_inputs.activities[origin].coords.lat]
        point2 = [model_inputs.activities[destination].coords.lng, model_inputs.activities[destination].coords.lat]

        diff_x = point2[0] - point1[0]
        diff_y = point2[1] - point1[1]

        plt.arrow(
            point1[0],
            point1[1],
            diff_x,
            diff_y,
            length_includes_head=True,
            head_width=0.0008,
            width=0.0003,
            zorder=1,
        )
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.axis("off")
    plt.savefig(f"02_reports/{current_run}/route_day_{day_to_plot}.png", bbox_inches="tight")
    plt.close()

    with open(f"02_reports/{current_run}/route_day_{day_to_plot}.txt", "w") as text_file:
        text_file.write("\n".join(destination_list))


def visualize_output(model_inputs: ModelInputs, solution: dict[int, list[str]], current_run: str) -> None:
    """Visualize output."""
    logger.info("Generating trip report")

    for day in range(model_inputs.no_of_days):
        generate_day_route_output(model_inputs, solution, day, current_run)
