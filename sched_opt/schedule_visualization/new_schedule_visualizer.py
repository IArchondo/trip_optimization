"""Define new visualizer."""

import logging
import os
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


def excel_to_image(excel_path: Path, img_output_path: Path) -> None:
    """Convert Excel file to an image with readable text."""
    df = pd.read_excel(excel_path)

    font_size = 7
    row_height = 0.5  # inches per row
    fig_height = max(1.5, row_height * (len(df) + 1))  # +1 for column header

    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,  # type: ignore
        colLabels=list(df.columns),
        loc="center",
        cellLoc="left",
        colLoc="left",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(font_size + 1)
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    fig.savefig(img_output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


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

    # destination_list = []
    for ix, destination in enumerate(solution[day_to_plot]):
        # destination_list.append(f"{ix}-{destination}")
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


def generate_trip_report_pdf(model_inputs: ModelInputs, current_run: str) -> None:
    """Generate trip report as PDF."""
    folder_path = Path("02_reports") / f"{current_run}"
    folder = Path(folder_path)

    c = canvas.Canvas(str(folder_path / "report.pdf"), pagesize=A4)
    width, height = A4

    for day in range(model_inputs.no_of_days):
        img_path = folder / f"route_day_{day}.png"
        table_path = folder / f"schedule_image_day_{day}.jpg"

        # Title
        c.setFont("Helvetica-Bold", 20)
        c.drawString(1 * inch, height - 1 * inch, f"Day {day}")

        # --- Table Image ---
        with Image.open(table_path) as img:
            table_width, table_height = img.size

        scale_factor = 0.4
        table_width_pt = table_width * 0.75 * scale_factor
        table_height_pt = table_height * 0.75 * scale_factor

        table_x = (width - table_width_pt) / 2
        table_y = height - 1.5 * inch - table_height_pt

        c.drawImage(
            str(table_path),
            table_x,
            table_y,
            width=table_width_pt,
            height=table_height_pt,
            preserveAspectRatio=True,
            mask="auto",
        )

        # --- Map Image ---
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        map_width_pt = img_width * 0.75 * scale_factor
        map_height_pt = img_height * 0.75 * scale_factor

        map_x = (width - map_width_pt) / 2
        map_y = table_y - map_height_pt - 0.5 * inch
        map_y = max(map_y, 1 * inch)  # Ensure it stays on page

        c.drawImage(
            str(img_path),
            map_x,
            map_y,
            width=map_width_pt,
            height=map_height_pt,
            preserveAspectRatio=True,
            mask="auto",
        )

        c.showPage()

    c.save()

    for day in range(model_inputs.no_of_days):
        os.remove(folder_path / f"schedule_image_day_{day}.jpg")
        os.remove(folder_path / f"route_day_{day}.png")


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
            end_time = current_time + timedelta(minutes=model_inputs.activities[origin].activity_duration)
            timeline.append(
                {
                    "activity": origin,
                    "start_time": current_time.strftime("%H:%M"),
                    "end_time": end_time.strftime("%H:%M"),
                }
            )
            current_time = end_time
            end_time = current_time + timedelta(minutes=model_inputs.trips[(origin, destination)].duration)
            timeline.append(
                {
                    "activity": "travel",
                    "start_time": current_time.strftime("%H:%M"),
                    "end_time": end_time.strftime("%H:%M"),
                }
            )
            current_time = end_time
        timeline.append(
            {
                "activity": model_inputs.hotel,
                "start_time": current_time.strftime("%H:%M"),
                "end_time": current_time.strftime("%H:%M"),
            }
        )
        pd.DataFrame(timeline).to_excel(folder_path / f"schedule_day_{day}.xlsx", index=False)
        excel_to_image(folder_path / f"schedule_day_{day}.xlsx", folder_path / f"schedule_image_day_{day}.jpg")


def visualize_output(model_inputs: ModelInputs, solution: dict[int, list[str]], current_run: str) -> None:
    """Visualize output."""
    logger.info("Generating trip report")

    generate_schedules(model_inputs, solution, current_run)

    for day in range(model_inputs.no_of_days):
        generate_day_route_output(model_inputs, solution, day, current_run)

    generate_trip_report_pdf(model_inputs, current_run)
