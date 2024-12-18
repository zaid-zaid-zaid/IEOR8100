from __future__ import print_function

import random
from . import utils


class MCTS(object):
    def __init__(self, tree_policy, default_policy, backup):
        self.tree_policy = tree_policy
        self.default_policy = default_policy
        self.backup = backup

    def __call__(self, root, n=1500):
        """
        Run tree search
        """
        if root.parent is not None:
            raise ValueError("Root's parent must be None.")

        for _ in range(n):
            node = _get_next_node(root, self.tree_policy)
            node.reward = self.default_policy(node)
            self.backup(node)

        return utils.rand_max(root.children.values(), key=lambda x: x.q).action


def _expand(state_node):
    action = random.choice(state_node.untried_actions)
    return state_node.children[action].sample_state()


def _best_child(state_node, tree_policy):
    best_action_node = utils.rand_max(state_node.children.values(),
                                      key=tree_policy)
    return best_action_node.sample_state()


def _get_next_node(state_node, tree_policy):
    while not state_node.state.is_terminal():
        if state_node.untried_actions:
            return _expand(state_node)
        else:
            state_node = _best_child(state_node, tree_policy)
    return state_node
