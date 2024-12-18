from __future__ import division
from .graph import StateNode, ActionNode


class Bellman(object):
    """
    For future testing...Dynamical programming update similar to the Bellman equation
    of value iteration (Feldman and Domshlak, 2014).
    """
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, node):
        """
        :param node: The node to start backups from
        """
        while node is not None:
            node.n += 1
            if isinstance(node, StateNode):
                node.q = max([x.q for x in node.children.values()])
            elif isinstance(node, ActionNode):
                n = sum([x.n for x in node.children.values()])
                node.q = sum([(self.gamma * x.q + x.reward) * x.n
                              for x in node.children.values()]) / n
            node = node.parent


def monte_carlo(node):
    """
    UCT monte carlo update.
    :param node: The node to start backups from
    """
    r = node.reward
    while node is not None:
        node.n += 1
        node.q = ((node.n - 1)/node.n) * node.q + 1/node.n * r
        node = node.parent
