import numpy as np
from grid_world import GridWorldAnimationApp
from utils import images_to_video


class Policy:
    def __init__(self, total_steps, n_observation, policy_name='No name'):
        self.policy = []
        for i in range(total_steps):
            self.policy.append([0] * n_observation)
        self.policy_name = policy_name

    def set_action(self, time_step, observation, action):
        self.policy[time_step][observation] = action

    def get_action(self, time_step, observation):
        return self.policy[time_step][observation]

    def evaluate(self, env):
        transitions = env.get_transitions()
        total_steps = env.total_steps
        n_observation = env.n_observation
        dp_table = np.zeros((total_steps + 1, n_observation))
        for i in range(total_steps - 1, -1, -1):
            for o in range(n_observation):
                observations = np.asarray(transitions[(o, self.get_action(i, o))][0])
                rewards = np.asarray(transitions[(o, self.get_action(i, o))][1])
                dp_table[i, o] = observations.dot(rewards + dp_table[i+1, :])
        return dp_table[0, env.state_to_observation(env.initial_state)]


def solve_single_env(env):
    return solve_multiple_env([env])


def solve_multiple_env(envs):
    environments_transitions = [env.get_transitions() for env in envs]
    total_steps = envs[0].total_steps
    n_action = envs[0].n_action
    n_observation = envs[0].n_observation
    policy = Policy(total_steps, n_observation)
    dp_table = np.zeros((total_steps + 1, n_observation))
    for i in range(total_steps - 1, -1, -1):
        for o in range(n_observation):
            action_values = np.zeros(n_action)
            for a in range(n_action):
                for transitions in environments_transitions:
                    observations = np.asarray(transitions[(o, a)][0])
                    rewards = np.asarray(transitions[(o, a)][1])
                    action_values[a] += observations.dot(rewards + dp_table[i+1, :])
            best_action = np.argmax(action_values)
            policy.set_action(i, o, best_action)
            dp_table[i, o] = action_values[best_action] / len(envs)
    return policy, dp_table[0, envs[0].state_to_observation(envs[0].initial_state)]


def solve_objective(envs, optimal_rewards, threshold_ratio):
    n_env = len(envs)
    total_steps = envs[0].total_steps
    min_reward_range, max_reward_range = float('inf'), -float('inf')
    for env in envs:
        min_reward_range, max_reward_range = min(min_reward_range, env.reward_range[0]), max(max_reward_range, env.reward_range[1])
    reward_corrections = []
    for env_index in range(n_env):
        reward_corrections.append((max(optimal_rewards) - optimal_rewards[env_index]) / total_steps - min_reward_range)
    threshold = threshold_ratio * (total_steps * (max_reward_range - min_reward_range) + max(optimal_rewards))
    environments_transitions = [env.get_transitions() for env in envs]
    n_action = envs[0].n_action
    n_observation = envs[0].n_observation
    policy = Policy(total_steps, n_observation)
    dp_table = np.zeros((total_steps + 1, n_observation, n_env))
    for i in range(total_steps - 1, -1, -1):
        for o in range(n_observation):
            action_values = np.zeros((n_action, n_env))
            for a in range(n_action):
                for env_index in range(n_env):
                    observations = np.asarray(environments_transitions[env_index][(o, a)][0])
                    rewards = np.asarray(environments_transitions[env_index][(o, a)][1])
                    action_values[a, env_index] = min(observations.dot(rewards + dp_table[i+1, :, env_index]) + reward_corrections[env_index], threshold)
            best_action = np.argmax(action_values.sum(axis=1))
            policy.set_action(i, o, best_action)
            dp_table[i, o, :] = action_values[best_action, :]
    reached = dp_table[0, envs[0].state_to_observation(envs[0].initial_state)].sum() >= n_env * threshold - 1e-6
    return policy, reached


def compare_objectives(envs, record=False):
    n_recordings = 1
    optimal_policies, optimal_rewards = [], []
    for env_index, env in enumerate(envs):
        print('Computing the optimal policy for environment No.{} ...'.format(env_index))
        optimal_policy, optimal_reward = solve_single_env(env)
        optimal_policy.policy_name = 'Optimal policy for {}'.format(env.grid_world_name)
        optimal_policies.append(optimal_policy)
        optimal_rewards.append(optimal_reward)
    delta_span = [0.0, 1.0]
    iterations = 0
    print('Computing the policy minimizing the gap to optimality ...', end='')
    while delta_span[1] - delta_span[0] > 1e-5:
        delta = sum(delta_span) / 2
        _, reached = solve_objective(envs, optimal_rewards, delta)
        if reached:
            delta_span[0] = delta
        else:
            delta_span[1] = delta
        iterations += 1
    print(' (total iterations : {})'.format(iterations))
    min_gap_policy, reached = solve_objective(envs, optimal_rewards, delta_span[0])
    min_gap_policy.policy_name = 'Policy with the minimal gap to optimality'
    average_policy, _ = solve_multiple_env(envs)
    average_policy.policy_name = 'Policy with the best average reward'
    min_gap_policy_reward_sum, average_policy_reward_sum = 0, 0
    min_gap_policy_max_gap, average_policy_max_gap = -float('inf'), -float('inf')
    for env_index, env in enumerate(envs):
        if record:
            for title, policy in [('env_{}-opt'.format(env_index), optimal_policies[env_index]), ('env_{}-min_gap'.format(env_index), min_gap_policy), ('env_{}-avg'.format(env_index), average_policy)]:
                for i in range(n_recordings):
                    num_title = '{}_{}'.format(title, i)
                    GridWorldAnimationApp(env, policy, num_title).run()
                    images_to_video(num_title)
        min_gap_policy_reward = min_gap_policy.evaluate(env)
        average_policy_reward = average_policy.evaluate(env)
        min_gap_policy_reward_sum += min_gap_policy_reward
        average_policy_reward_sum += average_policy_reward
        min_gap_policy_max_gap = max(min_gap_policy_max_gap, optimal_rewards[env_index] - min_gap_policy_reward)
        average_policy_max_gap = max(average_policy_max_gap, optimal_rewards[env_index] - average_policy_reward)
        print('Environment No.{} - optimal reward : {}, minimal gap policy reward : {} (gap : {}), average policy reward : {} (gap : {})'.format(env_index, optimal_rewards[env_index], min_gap_policy_reward, optimal_rewards[env_index] - min_gap_policy_reward, average_policy_reward, optimal_rewards[env_index] - average_policy_reward))
    print('Average rewards over the environments - minimal gap policy : {}, average policy : {}'.format(min_gap_policy_reward_sum / len(envs), average_policy_reward_sum / len(envs)))
    print('Maximal gap over the environments - minimal gap policy : {}, average policy : {}'.format(min_gap_policy_max_gap, average_policy_max_gap))
    return min_gap_policy
