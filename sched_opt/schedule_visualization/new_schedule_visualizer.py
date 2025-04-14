"""Define new visualizer."""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from reportlab.lib.pagesizes import A4  # type: ignore
from reportlab.lib.units import inch  # type: ignore
from reportlab.pdfgen import canvas  # type: ignore

from sched_opt.elements import ModelInputs

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
        destination_list.append(f"{ix}-{destination}")
        if destination != model_inputs.hotel:
            ax.annotate(
                f"{ix}-{destination}",
                (
                    model_inputs.activities[destination].coords.lng,
                    model_inputs.activities[destination].coords.lat,
                ),
            )

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


def generate_trip_report_pdf(model_inputs: ModelInputs, current_run: str) -> None:
    """Generate trip report as pdf."""
    folder_path = Path("02_reports") / f"{current_run}"
    folder = Path(folder_path)

    # Create PDF canvas
    c = canvas.Canvas(str(folder_path / "report.pdf"), pagesize=A4)
    width, height = A4

    for day in range(model_inputs.no_of_days):
        txt_path = folder / f"route_day_{day}.txt"
        img_path = folder / f"route_day_{day}.png"

        if not txt_path.exists() or not img_path.exists():
            break  # Stop when there are no more files

        c.setFont("Helvetica-Bold", 20)
        c.drawString(1 * inch, height - 1 * inch, f"Day {day}")

        # Draw text
        c.setFont("Courier", 12)
        with open(txt_path) as f:
            lines = f.readlines()

        text_obj = c.beginText(1 * inch, height - 1.5 * inch)
        for line in lines:
            text_obj.textLine(line.strip())
        c.drawText(text_obj)

        # Use PIL to get image size and apply scaling
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        # Convert to points (1 pixel = 0.75 points)
        img_width_pt = img_width * 0.75
        img_height_pt = img_height * 0.75

        scale_factor = 0.4
        scaled_width = img_width_pt * scale_factor
        scaled_height = img_height_pt * scale_factor

        # Position image below text
        text_block_height = len(lines) * 14  # approx line height in points
        image_y_position = height - 1.5 * inch - text_block_height - scaled_height - 0.5 * inch
        image_y_position = max(image_y_position, 1 * inch)  # avoid running off page

        x_pos = (width - scaled_width) / 2

        c.drawImage(
            str(img_path),
            x_pos,
            image_y_position,
            width=scaled_width,
            height=scaled_height,
            preserveAspectRatio=True,
            mask="auto",
        )

        c.showPage()
    c.save()


def generate_schedules(model_inputs: ModelInputs, solution: dict[int, list[str]], current_run: str) -> None:
    """Generate schedules for each day."""
    folder_path = Path("02_reports") / f"{current_run}"

    for day in solution:
        route = solution[day]
        trips = list(zip(route, route[1:], strict=False))
        start_time = datetime.strptime("08:00", "%H:%M")
        current_time = start_time
        timeline = []

        for origin, destination in trips:
            timeline.append(
                {
                    "activity": origin,
                    "time": current_time.strftime("%H:%M"),
                }
            )
            current_time += timedelta(minutes=model_inputs.activities[origin].activity_duration)
            timeline.append({"activity": "travel", "time": current_time.strftime("%H:%M")})
            current_time += timedelta(minutes=model_inputs.trips[(origin, destination)].duration)
        timeline.append({"activity": model_inputs.hotel, "time": current_time.strftime("%H:%M")})
        pd.DataFrame(timeline).to_excel(folder_path / f"schedule_day_{day}.xlsx")


def visualize_output(model_inputs: ModelInputs, solution: dict[int, list[str]], current_run: str) -> None:
    """Visualize output."""
    logger.info("Generating trip report")

    generate_schedules(model_inputs, solution, current_run)

    for day in range(model_inputs.no_of_days):
        generate_day_route_output(model_inputs, solution, day, current_run)

    generate_trip_report_pdf(model_inputs, current_run)
