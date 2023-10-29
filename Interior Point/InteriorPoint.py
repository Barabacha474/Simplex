from math import log10
from dataclasses import dataclass
from simplex.models import Matrix, IdentityMatrix, Vector
from simplex.exceptions import InfeasibleSolution


@dataclass
class InteriorPointSolution:
    """Stores a solution to simplex method.

    Args:
        decision_variables (Vector): Vector of decision variables
        value (float): Solution to maximization problem
    """

    decision_variables: Vector
    value: float


class InteriorPoint:
    def __init__(self,
                 A: Matrix,
                 b: Vector,
                 C: Vector,
                 X: Vector,
                 accuracy: float = 1e-4):
                   


    def solve(self, debug: bool = False) -> InteriorPointSolution:
