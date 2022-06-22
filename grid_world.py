import random
import numpy
from gym import spaces
from utils import create_directory
import kivy
kivy.config.Config.set('graphics', 'resizable', False)
from kivy.app import App
from kivy.uix.widget import Widget


class GridWorld:
    # action_noise : dict from states to (p_0, p_1, p_2, p_3, p_4), whose sum is <= 1
    # walls : set of states representing walls
    # terminal_state_rewards : dict from states representing terminal states to rewards
    def __init__(self, grid_length_x, grid_length_y, initial_state, action_noise, walls, terminal_state_rewards, total_steps, grid_world_name):
        super(GridWorld, self).__init__()
        self.grid_length_x = grid_length_x
        self.grid_length_y = grid_length_y
        self.n_action = 5
        self.action_space = spaces.Discrete(self.n_action)
        self.n_observation = grid_length_x * grid_length_y
        self.observation_space = spaces.Discrete(self.n_observation)
        self.current_observation = None
        self.current_step = None
        self.initial_state = initial_state
        self.reset()
        self.action_noise = action_noise
        for observation in range(self.n_observation):
            state = self.observation_to_state(observation)
            if state not in action_noise:
                self.action_noise[state] = [0.0] * 5
        self.walls = walls
        self.wall_reward = -3
        self.terminal_state_rewards = terminal_state_rewards  # Dict that goes from states to rewards
        self.transition_reward = -1
        self.transitions = dict()
        self.populate_transitions()
        self.total_steps = total_steps
        self.reward_range = [op(self.transition_reward, self.wall_reward, op(self.terminal_state_rewards.values()), 0) for op in [min, max]]
        self.grid_world_name = grid_world_name

    def reset(self):
        self.current_observation = self.state_to_observation(self.initial_state)
        self.current_step = 0
        return self.current_observation

    def step(self, action):
        self.current_step += 1
        if self.current_step > self.total_steps:
            return self.current_observation, 0.0, True, {}
        next_observation = random.choices(population=range(self.n_observation), weights=self.transitions[(self.current_observation, action)][0], k=1)[0]
        past_observation = self.current_observation
        self.current_observation = next_observation
        return next_observation, self.transitions[(past_observation, action)][1][next_observation], self.current_step == self.total_steps, {}

    def populate_transitions(self):
        for action in range(self.n_action):
            for observation in range(self.n_observation):
                state = self.observation_to_state(observation)
                action_dist = self.action_noise[state][:]
                action_dist[action] += 1 - sum(action_dist)
                next_observations, rewards = [0.0] * self.n_observation, [0.0] * self.n_observation
                if state in self.terminal_state_rewards:
                    next_observations[observation] = 1.0
                else:
                    for noisy_action in range(5):
                        next_unconstrained_state = self.state_after_action(state, noisy_action)
                        if not self.valid_state(next_unconstrained_state) or next_unconstrained_state in self.walls:
                            next_observations[observation] += action_dist[noisy_action]
                            rewards[observation] += action_dist[noisy_action] * self.wall_reward
                        else:
                            next_observation = self.state_to_observation(next_unconstrained_state)
                            next_observations[next_observation] += action_dist[noisy_action]
                            rewards[next_observation] += action_dist[noisy_action] * ((self.transition_reward if noisy_action > 0 else 0) + (self.terminal_state_rewards[next_unconstrained_state] if next_unconstrained_state in self.terminal_state_rewards else 0))
                    for o in range(self.n_observation):
                        if next_observations[o] > 0.0:
                            rewards[o] /= next_observations[o]
                self.transitions[(observation, action)] = (next_observations, rewards)

    def get_transitions(self):
        return self.transitions

    def valid_state(self, state):
        return isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], int) and 0 <= state[0] < self.grid_length_x and isinstance(state[1], int) and 0 <= state[1] < self.grid_length_y

    def observation_to_state(self, observation):
        assert isinstance(observation, int) and 0 <= observation < self.n_observation, 'Observation is not valid'
        return observation // self.grid_length_y, observation % self.grid_length_y

    def state_to_observation(self, state):
        return state[0] * self.grid_length_y + state[1] if self.valid_state(state) else None

    def state_after_action(self, state, action):
        assert self.valid_state(state), 'State is not valid'
        grid_action = self.action_to_grid_action(action)
        return state[0] + grid_action[0], state[1] + grid_action[1]

    def observation_after_action(self, observation, action):
        return self.state_to_observation(self.state_after_action(self.observation_to_state(observation), action))

    @staticmethod
    def action_to_grid_action(action):
        assert isinstance(action, int) and 0 <= action < 5, 'Action is not valid'
        if action == 0:
            return 0, 0  # No move
        if action == 1:
            return 0, -1  # Up
        if action == 2:
            return 1, 0  # Right
        if action == 3:
            return 0, 1  # Down
        return -1, 0  # Left

    @staticmethod
    def action_name(action):
        assert (isinstance(action, int) or isinstance(action, numpy.int64)) and 0 <= action < 5, 'Action is not valid'  # (isinstance(action, int) or isinstance(action, numpy.int64))
        return ['No move', 'Up', 'Right', 'Down', 'Left'][action]


class GridWorldAnimationWidget(Widget):
    def __init__(self, grid_world, policy, title):
        from kivy.core.text import Label as CoreLabel
        from kivy.graphics import Color, Ellipse, Rectangle
        from kivy.core.window import Window
        super().__init__()
        assert isinstance(grid_world, GridWorld)
        self.grid_world = grid_world
        self.grid_world.reset()
        self.policy = policy
        self.title = title
        self.full_grid_side_margin = 0.1
        self.full_grid_top_margin = 0.1
        self.full_grid_bottom_margin = 0.1
        self.full_grid_side = 0.01
        self.thin_line_fraction = 0.3
        self.pos_dot_size = 0.4
        self.circle_diameter = 0.8
        self.cell_size = (Window.size[0] * (1 - 2 * self.full_grid_side_margin) / self.grid_world.grid_length_x, Window.size[1] * (1 - self.full_grid_top_margin - self.full_grid_bottom_margin) / self.grid_world.grid_length_y)
        self.past_states = [self.grid_world.observation_to_state(self.grid_world.current_observation)]
        self.draw_grid()
        self.total_reward = 0.0
        self.done = False
        with self.canvas:
            Color(0, 0, 1)
            pos, size = self.get_circle_pos_size(self.grid_world.observation_to_state(self.grid_world.current_observation)[0], self.grid_world.observation_to_state(self.grid_world.current_observation)[1], self.pos_dot_size)
            Ellipse(pos=pos, size=size)
        state = 'Time step : {}/{}, initial state : {}, next action : {}'.format(self.grid_world.current_step, self.grid_world.total_steps, self.grid_world.observation_to_state(self.grid_world.current_observation), self.grid_world.action_name(self.policy.get_action(self.grid_world.current_step, self.grid_world.current_observation)))
        self.draw_grid()
        with self.canvas:
            Color(0, 0, 1)
            pos, size = self.get_circle_pos_size(self.grid_world.observation_to_state(self.grid_world.current_observation)[0], self.grid_world.observation_to_state(self.grid_world.current_observation)[1], self.pos_dot_size)
            Ellipse(pos=pos, size=size)
            label = CoreLabel(text=state, font_size=Window.size[1] * 0.025, color=(0, 0, 0, 1))
            label.refresh()
            texture = label.texture
            texture_size = list(texture.size)
            pos = (Window.size[0] / 2 - texture_size[0] / 2, Window.size[1] * (self.full_grid_bottom_margin * 0.2))
            Widget().canvas.add(Rectangle(texture=texture, size=texture_size, pos=pos))

    def draw_grid(self):
        from kivy.graphics import Color, Ellipse, Rectangle, Line
        from kivy.core.window import Window
        from kivy.core.text import Label as CoreLabel
        with self.canvas:
            Color(1, 1, 1)
            Rectangle(pos=(0, 0), size=Window.size)
            Color(0, 0, 0)
            Rectangle(pos=(self.full_grid_side_margin * Window.size[0], self.full_grid_bottom_margin * Window.size[1]), size=((1 - 2 * self.full_grid_side_margin) * Window.size[0], (1 - self.full_grid_top_margin - self.full_grid_bottom_margin) * Window.size[1]))
            Color(1, 1, 1)
            Rectangle(pos=(self.full_grid_side_margin * Window.size[0] + 0.5 * self.full_grid_side * min(Window.size), self.full_grid_bottom_margin * Window.size[1] + 0.5 * self.full_grid_side * min(Window.size)), size=((1 - 2 * self.full_grid_side_margin) * Window.size[0] - self.full_grid_side * min(Window.size), (1 - self.full_grid_top_margin - self.full_grid_bottom_margin) * Window.size[1] - self.full_grid_side * min(Window.size)))
            Color(0.5, 0.5, 0.5)
            for x in range(self.grid_world.grid_length_x - 1):
                Rectangle(pos=(self.full_grid_side_margin * Window.size[0] + Window.size[0] * (1 - 2 * self.full_grid_side_margin) * (x + 1) / self.grid_world.grid_length_x, self.full_grid_bottom_margin * Window.size[1] + 0.5 * self.full_grid_side * min(Window.size)), size=(self.thin_line_fraction * self.full_grid_side * min(Window.size), (1 - self.full_grid_top_margin - self.full_grid_bottom_margin) * Window.size[1] - self.full_grid_side * min(Window.size)))
            for y in range(self.grid_world.grid_length_y - 1):
                Rectangle(pos=(self.full_grid_side_margin * Window.size[0] + 0.5 * self.full_grid_side * min(Window.size), self.full_grid_bottom_margin * Window.size[1] + Window.size[1] * (1 - self.full_grid_top_margin - self.full_grid_bottom_margin) * (y + 1) / self.grid_world.grid_length_y), size=((1 - 2 * self.full_grid_side_margin) * Window.size[0] - self.full_grid_side * min(Window.size), self.thin_line_fraction * self.full_grid_side * min(Window.size)))
            Color(0, 0, 0)
            for wall_x, wall_y in self.grid_world.walls:  # Maybe swap these
                pos, size = self.get_rectangle_pos_size(wall_x, wall_y)
                Rectangle(pos=pos, size=size)
            Color(1, 1, 0)
            pos, size = self.get_circle_pos_size(self.grid_world.initial_state[0], self.grid_world.initial_state[1], self.circle_diameter)
            Ellipse(pos=pos, size=size)
            label = CoreLabel(text='S', font_size=0.03*min(Window.size), color=(0, 0, 0, 1))
            label.refresh()
            texture = label.texture
            texture_size = list(texture.size)
            Widget().canvas.add(Rectangle(texture=texture, size=texture_size, pos=(pos[0] + size[0] / 2 - texture_size[0] / 2, pos[1] + size[1] / 2 - texture_size[1] / 2)))
            for x in range(self.grid_world.grid_length_x):
                for y in range(self.grid_world.grid_length_y):
                    for a in range(5):
                        if self.grid_world.action_noise[(x, y)][a] > 0:
                            pos, size = self.get_circle_pos_size(x, y, self.circle_diameter)
                            label = CoreLabel(text=str(self.grid_world.action_noise[(x, y)][a]), font_size=0.015*min(Window.size), color=(0, 0, 0, 1))
                            label.refresh()
                            texture = label.texture
                            texture_size = list(texture.size)
                            action_pos = self.grid_world.action_to_grid_action(a)
                            f = 0.4
                            pos = (pos[0] + size[0] / 2 - texture_size[0] / 2 + action_pos[0] * self.cell_size[0] * f, pos[1] + size[1] / 2 - texture_size[1] / 2 + action_pos[1] * self.cell_size[1] * f)
                            Widget().canvas.add(Rectangle(texture=texture, size=texture_size, pos=pos))
            Color(0, 1, 0)
            for terminal_x, terminal_y in self.grid_world.terminal_state_rewards.keys():
                if self.grid_world.terminal_state_rewards[(terminal_x, terminal_y)] >= 0:
                    Color(0, 1, 0)
                else:
                    Color(1, 0, 0)
                pos, size = self.get_circle_pos_size(terminal_x, terminal_y, self.circle_diameter)
                Ellipse(pos=pos, size=size)
                label = CoreLabel(text=str(self.grid_world.terminal_state_rewards[(terminal_x, terminal_y)]), font_size=0.03*min(Window.size), color=(0, 0, 0, 1))
                label.refresh()
                texture = label.texture
                texture_size = list(texture.size)
                Widget().canvas.add(Rectangle(texture=texture, size=texture_size, pos=(pos[0] + size[0] / 2 - texture_size[0] / 2, pos[1] + size[1] / 2 - texture_size[1] / 2)))
            label = CoreLabel(text='{} - {}'.format(self.grid_world.grid_world_name, self.policy.policy_name), font_size=Window.size[1] * 0.04, color=(0, 0, 0, 1))
            label.refresh()
            texture = label.texture
            texture_size = list(texture.size)
            pos = (Window.size[0] / 2 - texture_size[0] / 2, Window.size[1] * (1 - self.full_grid_top_margin * 0.6))
            Widget().canvas.add(Rectangle(texture=texture, size=texture_size, pos=pos))
            self.past_states.append(self.grid_world.observation_to_state(self.grid_world.current_observation))
            Color(0, 0, 1)
            points = []
            for i in range(len(self.past_states)):
                start_pos, start_size = self.get_rectangle_pos_size(self.past_states[i][0], self.past_states[i][1])
                points += [start_pos[0] + start_size[0] / 2, start_pos[1] + start_size[1] / 2]
            Line(points=points, width=0.001 * min(Window.size))

    def get_rectangle_pos_size(self, x, y):
        from kivy.core.window import Window
        pos = self.full_grid_side_margin * Window.size[0] + self.thin_line_fraction * self.full_grid_side * min(Window.size) + x * (1 - 2 * self.full_grid_side_margin) * Window.size[0] / self.grid_world.grid_length_x, self.full_grid_bottom_margin * Window.size[1] + self.thin_line_fraction * self.full_grid_side * min(Window.size) + (self.grid_world.grid_length_y - y - 1) * (1 - self.full_grid_top_margin - self.full_grid_bottom_margin) * Window.size[1] / self.grid_world.grid_length_y
        size = (1 - 2 * self.full_grid_side_margin) * Window.size[0] / self.grid_world.grid_length_x - self.thin_line_fraction * self.full_grid_side * min(Window.size), (1 - self.full_grid_top_margin - self.full_grid_bottom_margin) * Window.size[1] / self.grid_world.grid_length_y - self.thin_line_fraction * self.full_grid_side * min(Window.size)
        return (pos[0], pos[1]), size

    def get_circle_pos_size(self, x, y, diameter):
        rect_pos, rect_size = self.get_rectangle_pos_size(x, y)
        pos = rect_pos[0] + rect_size[0] / 2, rect_pos[1] + rect_size[1] / 2
        size = tuple([min(rect_size[0], rect_size[1]) * diameter] * 2)
        pos = pos[0] - size[0] / 2, pos[1] - size[1] / 2
        return pos, size

    def next_action(self, args):
        from kivy.uix.widget import Widget
        from kivy.graphics import Color, Ellipse, Rectangle
        from kivy.core.text import Label as CoreLabel
        from kivy.core.window import Window
        last_observation = self.grid_world.current_observation
        if self.grid_world.current_step == 0:
            self.export_to_png('recordings/{}-0.png'.format(self.title))
        if self.done:
            App.get_running_app().stop()
            return
        else:
            action = self.policy.get_action(self.grid_world.current_step, self.grid_world.current_observation)
        next_observation, reward, done, _ = self.grid_world.step(action)
        self.done = done
        self.total_reward += reward
        if self.grid_world.current_step >= self.grid_world.total_steps:
            state = 'Episode ended with total reward : {}'.format(self.total_reward)
        elif self.grid_world.observation_to_state(self.grid_world.current_observation) in self.grid_world.terminal_state_rewards:
            state = 'Time step : {}/{}, reached terminal state : {}, {}total reward : {:.2f}'.format(self.grid_world.current_step, self.grid_world.total_steps, self.grid_world.observation_to_state(self.grid_world.current_observation), '' if last_observation == next_observation else 'last reward : {:.2f}, '.format(reward), self.total_reward)
        else:
            state = 'Time step : {}/{}, current state : {}, last reward : {:.2f}, total reward : {:.2f}, next action : {}'.format(self.grid_world.current_step, self.grid_world.total_steps, self.grid_world.observation_to_state(self.grid_world.current_observation), reward, self.total_reward, self.grid_world.action_name(self.policy.get_action(self.grid_world.current_step, self.grid_world.current_observation)))
        self.draw_grid()
        with self.canvas:
            Color(0, 0, 1)
            pos, size = self.get_circle_pos_size(self.grid_world.observation_to_state(self.grid_world.current_observation)[0], self.grid_world.observation_to_state(self.grid_world.current_observation)[1], self.pos_dot_size)
            Ellipse(pos=pos, size=size)
            label = CoreLabel(text=state, font_size=Window.size[1] * 0.025, color=(0, 0, 0, 1))
            label.refresh()
            texture = label.texture
            texture_size = list(texture.size)
            pos = (Window.size[0] / 2 - texture_size[0] / 2, Window.size[1] * (self.full_grid_bottom_margin * 0.2))
            Widget().canvas.add(Rectangle(texture=texture, size=texture_size, pos=pos))
        self.export_to_png('recordings/{}-{}.png'.format(self.title, self.grid_world.current_step))


class GridWorldAnimationApp(App):
    def __init__(self, grid_world, policy, title):
        super().__init__()
        self.grid_world = grid_world
        self.policy = policy
        self.title = title
        self.grid_world.reset()
        self.event = None

    def build(self):
        from kivy.clock import Clock
        from kivy.core.window import Window
        create_directory('recordings')
        Window.size = (900, 700)
        widget = GridWorldAnimationWidget(self.grid_world, self.policy, self.title)
        self.event = Clock.schedule_interval(widget.next_action, 0.3)
        return widget

    def on_stop(self):
        from kivy.clock import Clock
        Clock.unschedule(self.event)
