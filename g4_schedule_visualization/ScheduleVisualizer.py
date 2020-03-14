import matplotlib.pyplot as plt
import pandas as pd


class ScheduleVisualizer:
    def __init__(self, proc_distances_dict, solution_dict):
        self.proc_distances_dict = proc_distances_dict
        self.solution_dict = solution_dict
        self.coord_dict = self.__extract_coords(self.proc_distances_dict)
        self.coord_table = self.__create_coord_table(self.coord_dict)

        ## load map image for plotting
        self.map_image = plt.imread("01_map_data/map_input_1.png")
        self.BBox = (-74.05, -73.9209, 40.6106, 40.8)

    def __extract_coords(self, proc_distances_dict):
        """Extract coordinates from processed distances dictionary
        
        Args:
            proc_distances_dict (dict): processed distances dictionary
        
        Returns:
            dict: dictionary with coordinates for each place in dictionary
        """
        coord_dict = {}
        for place in list(proc_distances_dict["places"]):
            coords = proc_distances_dict["placed_geocoding_dict"][place][0]["geometry"][
                "location"
            ]
            coord_dict[place] = {"lng": coords["lng"], "lat": coords["lat"]}

        return coord_dict

    def __create_coord_table(self, coord_dict):
        """Create a table with coords based on a coordinates dict
        
        Args:
            coord_dict (dict): dictionary with coordinated for each place
        
        Returns:
            pd.DataFrame: dataframe with places, longitude and latitude
        """
        lng_list = [coord_dict[place]["lng"] for place in coord_dict.keys()]

        lat_list = [coord_dict[place]["lat"] for place in coord_dict.keys()]

        coord_table = pd.DataFrame(
            {"place": list(coord_dict.keys()), "lng": lng_list, "lat": lat_list}
        )

        return coord_table

    def generate_day_route_output(
        self, solution_dict, coord_table, day_to_plot, current_run
    ):
        """Save plots for routes in each day and txt files detailing each route
        
        Args:
            solution_dict (dict): dict with routes solution
            coord_table (pd.DataFrame): table with places, longitude and latitude
            day_to_plot (int): index from the day to plot
            current_run (str): string of the folder into which to save results
        """
        route_list = [
            (solution_dict[day_to_plot][i], solution_dict[day_to_plot][i + 1])
            for i in range(len(solution_dict[day_to_plot]) - 1)
        ]

        ## plot limits
        filt_coors = coord_table[coord_table["place"].isin(solution_dict[day_to_plot])]
        xmin = filt_coors["lng"].min() - 0.01
        xmax = filt_coors["lng"].max() + 0.01
        ymin = filt_coors["lat"].min() - 0.01
        ymax = filt_coors["lat"].max() + 0.01

        hotel_x = coord_table[coord_table["place"] == "Pod Times Square"]["lng"].iloc[0]
        hotel_y = coord_table[coord_table["place"] == "Pod Times Square"]["lat"].iloc[0]

        _, ax = plt.subplots(figsize=(16, 8.3))

        destination_list = []
        for ix, destination in enumerate(solution_dict[day_to_plot]):
            ax.annotate(
                f"{ix}-{destination}",
                (
                    coord_table[coord_table["place"] == destination]["lng"].iloc[0],
                    coord_table[coord_table["place"] == destination]["lat"].iloc[0],
                ),
            )
            destination_list.append(f"{ix}-{destination}")

        ax.scatter(coord_table.lng, coord_table.lat, zorder=2, c="red", s=120)
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
        ax.set_xlim(self.BBox[0], self.BBox[1])
        ax.set_ylim(self.BBox[2], self.BBox[3])
        ax.imshow(self.map_image, extent=self.BBox, aspect="equal", zorder=0)

        for route in route_list:
            place_from = coord_table[coord_table["place"] == route[0]]
            place_to = coord_table[coord_table["place"] == route[1]]

            point1 = [place_from["lng"].iloc[0], place_from["lat"].iloc[0]]
            point2 = [place_to["lng"].iloc[0], place_to["lat"].iloc[0]]

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
        plt.savefig(
            f"02_reports/{current_run}/route_day_{day_to_plot}.png", bbox_inches="tight"
        )
        plt.close()

        text_file = open(f"02_reports/{current_run}/route_day_{day_to_plot}.txt", "w")
        text_file.write("\n".join(destination_list))
        text_file.close()

    def execute_visualizer_pipeline(self, current_run):
        """Generate output for all days in solution
        
        Args:
            current_run (str): string detailing name of the folder to save output in
        """

        for day in list(self.solution_dict.keys()):
            self.generate_day_route_output(
                self.solution_dict, self.coord_table, day, current_run
            )
