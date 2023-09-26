import unittest
import numpy as np
from simplex.simplex import Simplex, InfeasibleSolution


class Test(unittest.TestCase):
    def test_0(self):
        A = np.array(
            [
                [6, 4],
                [1, 2],
                [-1, 1],
                [0, 1],
            ],
            dtype=np.double,
        )
        b = np.array([24, 6, 1, 2], dtype=np.double)
        C = np.array([5, 4], dtype=np.double)
        simplex = Simplex(A, b.T, C)
        solution = simplex.solve()
        self.assertEqual(solution.decision_variables.tolist(), [3, 1.5])
        self.assertEqual(21, solution.value)

    def test_1(self):
        A = np.array(
            [
                [1, 2],
                [1, 1],
                [3, 2],
            ],
            dtype=np.double,
        )
        b = np.array([16, 9, 24], dtype=np.double)
        C = np.array([40, 30], dtype=np.double)
        simplex = Simplex(A, b.T, C)
        solution = simplex.solve()
        self.assertEqual(solution.decision_variables.tolist(), [6, 3])
        self.assertAlmostEqual(330, solution.value)

    def test_2(self):
        A = np.array(
            [
                [2, 1, 1],
                [1, 2, 3],
                [2, 2, 1],
            ],
            dtype=np.double,
        )
        b = np.array([2, 5, 6], dtype=np.double)
        C = np.array([3, 1, 3], dtype=np.double)
        simplex = Simplex(A, b.T, C)
        solution = simplex.solve()
        self.assertEqual(solution.decision_variables.tolist(), [0.2, 0, 1.6])
        self.assertAlmostEqual(27 / 5, solution.value)

    def test_3(self):
        A = np.array(
            [
                [2, 1, 1, 3],
                [1, 3, 1, 2],
            ],
            dtype=np.double,
        )
        b = np.array([5, 3], dtype=np.double)
        C = np.array([6, 8, 5, 9], dtype=np.double)
        simplex = Simplex(A, b.T, C)
        solution = simplex.solve()
        self.assertEqual(solution.decision_variables.tolist(), [2, 0, 1, 0])
        self.assertAlmostEqual(17, solution.value)

    def test_4(self):
        A = np.array(
            [
                [0, 2, 3],
                [1, 1, 2],
                [1, 2, 3],
            ],
            dtype=np.double,
        )
        b = np.array([5, 4, 7], dtype=np.double)
        C = np.array([2, 3, 4], dtype=np.double)
        simplex = Simplex(A, b.T, C)
        solution = simplex.solve()
        self.assertEqual(solution.decision_variables.tolist(), [1.5, 2.5, 0])
        self.assertAlmostEqual(21 / 2, solution.value)

    def test_5(self):
        A = np.array(
            [
                [0, 0],
            ],
            dtype=np.double,
        )
        b = np.array([1], dtype=np.double)
        C = np.array([2, 3], dtype=np.double)
        with self.assertRaises(InfeasibleSolution):
            simplex = Simplex(A, b.T, C)
            simplex.solve()
