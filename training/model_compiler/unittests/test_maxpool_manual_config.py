"""Manual configuration propagation and rejection tests."""
import unittest
from types import SimpleNamespace
import networkx as nx
from training.nn_tools.maxpool_config import configure_maxpool


class ManualConfigTest(unittest.TestCase):
    def graph(self):
        node = type('Pool', (), {})()
        node.layer_id, node.layer_type = 'pool', 'maxpool2d'
        node.kernel_shape, node.stride, node.padding = [3, 3], [2, 2], [1, 1]
        node.running_max = 2.5
        graph = SimpleNamespace(dag=nx.DiGraph())
        graph.dag.add_node(node)
        return graph, node

    def test_profile_and_calibration(self):
        graph, node = self.graph()
        configure_maxpool(graph, {'pool': {'alpha': 14, 'upper_bound': .1, 'B': 1}})
        self.assertEqual((node.alpha, node.upper_bound, node.B, node.running_max), (14, .1, 1., 2.5))

    def test_invalid_request_is_atomic(self):
        for values in ({'alpha': 15, 'upper_bound': 1, 'B': 1},
                       {'alpha': 7, 'upper_bound': float('nan'), 'B': 1},
                       {'alpha': 7, 'upper_bound': 1, 'B': 0},
                       {'alpha': 7, 'upper_bound': 1, 'B': 1, 'degree': 7}):
            graph, node = self.graph()
            with self.assertRaises((ValueError, TypeError)):
                configure_maxpool(graph, {'pool': values})
            self.assertFalse(hasattr(node, 'alpha'))
        graph, node = self.graph()
        with self.assertRaises(ValueError):
            configure_maxpool(graph, {'missing': {'alpha': 7, 'upper_bound': 1, 'B': 1}})
