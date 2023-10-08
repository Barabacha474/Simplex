import unittest
from simplex import Matrix, Vector
from simplex import Simplex, InfeasibleSolution


class Test(unittest.TestCase):
    def test_0(self):
        A = Matrix(
            [
                [6, 4],
                [1, 2],
                [-1, 1],
                [0, 1],
            ]
        )
        b = Vector([24, 6, 1, 2])
        C = Vector([5, 4])
        simplex = Simplex(A, b, C)
        solution = simplex.solve()
        self.assertEqual(solution.decision_variables.getVector(), [3.0, 1.5])
        self.assertEqual(21, solution.value)

    def test_1(self):
        A = Matrix(
            [
                [1, 2],
                [1, 1],
                [3, 2],
            ]
        )
        b = Vector([16, 9, 24])
        C = Vector([40, 30])
        simplex = Simplex(A, b, C)
        solution = simplex.solve()
        self.assertEqual(solution.decision_variables.getVector(), [6.0, 3.0])
        self.assertAlmostEqual(330, solution.value)

    def test_2(self):
        A = Matrix(
            [
                [2, 1, 1],
                [1, 2, 3],
                [2, 2, 1],
            ]
        )
        b = Vector([2, 5, 6])
        C = Vector([3, 1, 3])
        simplex = Simplex(A, b, C)
        solution = simplex.solve()
        self.assertEqual(solution.decision_variables.getVector(), [0.2, 0.0, 1.6])
        self.assertAlmostEqual(27 / 5, solution.value)

    def test_3(self):
        A = Matrix(
            [
                [2, 1, 1, 3],
                [1, 3, 1, 2],
            ]
        )
        b = Vector([5, 3])
        C = Vector([6, 8, 5, 9])
        simplex = Simplex(A, b, C)
        solution = simplex.solve()
        self.assertEqual(solution.decision_variables.getVector(), [2, 0, 1, 0])
        self.assertAlmostEqual(17, solution.value)

    def test_4(self):
        A = Matrix(
            [
                [0, 2, 3],
                [1, 1, 2],
                [1, 2, 3],
            ]
        )
        b = Vector([5, 4, 7])
        C = Vector([2, 3, 4])
        simplex = Simplex(A, b, C)
        solution = simplex.solve()
        self.assertEqual(solution.decision_variables.getVector(), [1.5, 2.5, 0])
        self.assertAlmostEqual(21 / 2, solution.value)

    def test_5(self):
        A = Matrix(
            [
                [0, 0],
            ]
        )
        b = Vector([1])
        C = Vector([2, 3])
        with self.assertRaises(InfeasibleSolution):
            simplex = Simplex(A, b, C)
            simplex.solve()
