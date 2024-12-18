import random


def immediate_reward(state_node):
    """
    Estimate reward with the immediate return of that state.
    :param state_node:
    :return:
    """
    return state_node.state.reward(state_node.parent.parent.state,
                                   state_node.parent.action)


class RandomKStepRollOut(object):
    """
    Estimate reward with the sum of returns of a k step rollout
    """
    def __init__(self, k):
        self.k = k

    def __call__(self, state_node):
        self.current_k = 0

        def stop_k_step(state):
            self.current_k += 1
            return self.current_k > self.k or state.is_terminal()

        return _roll_out(state_node, stop_k_step)


def _roll_out(state_node, stopping_criterion):
    reward = 0
    state = state_node.state
    parent = state_node.parent.parent.state
    action = state_node.parent.action
    while not stopping_criterion(state):
        reward += state.reward(parent, action)

        action = random.choice(state_node.state.actions)
        parent = state
        state = parent.perform(action)

    return reward
