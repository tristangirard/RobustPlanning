import os
os.environ["KIVY_NO_ARGS"] = "1"
os.environ["KIVY_NO_CONSOLELOG"] = "1"
from algorithm import compare_objectives
from grid_world import GridWorld

if __name__ == '__main__':
    grid_length_x = 7
    grid_length_y = 7
    initial_state = (3, 3)
    action_noise = {}
    walls = [(0, 3), (1, 3), (5, 3), (6, 3), (3, 0), (3, 1), (3, 5), (3, 6)]
    total_steps = 6
    environments = [GridWorld(grid_length_x=grid_length_x, grid_length_y=grid_length_y, initial_state=initial_state,
                              action_noise=action_noise, walls=walls,
                              terminal_state_rewards={(0, 0): 11, (0, 6): 1, (6, 0): 22}, total_steps=total_steps,
                              grid_world_name='Grid world 0'),
                    GridWorld(grid_length_x=grid_length_x, grid_length_y=grid_length_y, initial_state=initial_state,
                              action_noise=action_noise, walls=walls,
                              terminal_state_rewards={(0, 0): 11, (0, 6): 22, (6, 0): 1}, total_steps=total_steps,
                              grid_world_name='Grid world 1')]
    policy = compare_objectives(environments, record=True)
