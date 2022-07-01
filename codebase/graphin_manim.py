from manim import *
import codebase.analysis_functions as af
import codebase.web_scrape_functions as wsf
import pandas as pd
config.quality = 'low_quality'

class CareerGraph(Scene):
    def __init__(self, player_id, renderer=None, camera_class=Camera, always_update_mobjects=False, random_seed=None, skip_animations=False,
                _format = 'test', _type='bat', player_age=None, dates=None, window_size=10):
        super().__init__(renderer, camera_class, always_update_mobjects, random_seed, skip_animations)
        
        if player_age:
            self.dates = af.dates_from_age(player_id, player_age)
        else:
            self.dates = dates
        
        self.player_id = player_id
        self.player_name = wsf.get_player_json(self.player_id)["name"]
        self.type = _type
        self.format = _format
        self.matchlist = wsf.player_match_list(player_id, dates=self.dates, _format=self.format)
        self.innings = af.get_cricket_totals(self.player_id, self.matchlist, _type=self.type, by_innings=True, is_object_id=True)
        self.running_ave = af.get_running_average(self.player_id, self.innings)
        self.recent_form_ave = af.get_recent_form_average(self.player_id, self.innings, window_size=window_size)
    
    def construct(self):
        x = np.arange(len(self.running_ave))
        innings_df = pd.DataFrame(self.innings)
        r_a = self.running_ave
        r_f_a = self.recent_form_ave
        _x_range=[0, x[-1]+1, (lambda x:x//4 if (x < 50) else 10)(x[-1])]
        _y_range=[0, max(innings_df.runs)+20, max((max(innings_df.runs)//8), 10)]
        
        # axes = Axes(
        #     x_range=(_x_range[0], _x_range[1], _x_range[2]),
        #     y_range=(_y_range[0], _y_range[1], _y_range[2]),
        #     tips=False,
        #     axis_config={"include_numbers": True},
        #     #x_axis_config={
        #     #    "numbers_to_include": np.arange(_x_range[0],_x_range[1], _x_range[2])
        #     #},
        #     #y_axis_config={
        #     #    "numbers_to_include": np.arange(_y_range[0],_y_range[1], _y_range[2])
        #     #}
        # )
        
        axes = BarChart(
            values=innings_df.runs,
            y_range=(_y_range[0], _y_range[1], _y_range[2]),
            axis_config={"include_numbers": True},
            x_axis_config={
               "numbers_to_include": np.arange(_x_range[0],_x_range[1], _x_range[2])
            },
            bar_colors=['#003f5c']*len(x)
        )

        grid= VGroup()
        for val in np.arange(_x_range[0]+_x_range[2],_x_range[1]+(1*_x_range[2]), _x_range[2]):
            grid += axes.get_vertical_line(axes.c2p(val, _y_range[1], 0), color=GRAY)
        for val in np.arange(_y_range[0]+_y_range[2],_y_range[1], _y_range[2]):
            grid += axes.get_horizontal_line(axes.c2p(_x_range[1], val, 0), color=GRAY)

        running_ave = axes.plot_line_graph(x, r_a, add_vertex_dots=False, line_color=BLUE)
        recent_form_ave = axes.plot_line_graph(x, r_f_a, add_vertex_dots=False, line_color=RED)
        top_line = axes.plot(lambda x: _y_range[1], color=WHITE)
        area = axes.get_area(top_line, x_range=[23,47], color=YELLOW, opacity=0.2)
        title = Text(f"{self.player_name} Career")
        title.next_to(axes, UP)
        self.play(Create(axes), Create(grid), Write(title))
        self.play(Create(running_ave), Create(recent_form_ave), run_time=1.5)
        self.wait(2)
        self.play(Create(area))
        self.wait(2)

        #self.add(axes, running_ave, recent_form_ave, title, grid)

class LineGraphExample(Scene):
    def construct(self):
        plane = NumberPlane(
            x_range = (0, 300),
            y_range = (0, 200),
            axis_config={"include_numbers": True},
        )
        plane.scale_to_fit_width(10)
        line_graph = plane.plot_line_graph(
            x_values = [0, 1.5, 2, 2.8, 4, 6.25],
            y_values = [1, 3, 2.25, 4, 2.5, 1.75],
            line_color=GOLD_E,
            vertex_dot_style=dict(stroke_width=3,  fill_color=PURPLE),
            stroke_width = 4,
        )
        self.add(plane, line_graph)

if __name__ == "__main__":
    scene = CareerGraph('253802', player_age='30:')
    scene.render()

    # scene = LineGraphExample()
    # scene.render()

