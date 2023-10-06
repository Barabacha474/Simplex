from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from exceptions import *


@dataclass
class SimplexSolution:
    """Stores a solution to simplex method.

    Args:
        decision_variables (npt.NDArray[np.double]): Vector of decision variables
        value (float): Solution to maximization problem
    """

    decision_variables: npt.NDArray[np.double]
    value: float


class Simplex:
    def __init__(self,
                 A: npt.NDArray,
                 b: npt.NDArray,
                 C: npt.NDArray,
                 accuracy: float = 1e-4):
        """Initialization of simplex method solver.

        Args:
            A (npt.NDArray): (m x n) matrix representing the constraint coefficients
            b (npt.NDArray): m-vector representing the constraint right-hand side
            C (npt.NDArray): n-vector representing the objective-function coefficients
            accuracy (float, optional): Approximation accuracy. Defaults to 1e-4.
        """

        # Sanity checks for correct input
        assert A.ndim == 2, "A is not a matrix"
        assert b.ndim == 1, "b is not a vector"
        assert C.ndim == 1, "C is not a vector"
        assert A.shape[0] == b.size, "Length of vector b does not correspond to # of rows of matrix A"
        assert A.shape[1] == C.size, "Length of vector C does not correspond to # of cols of matrix A"

        # Add slack variables
        self.A = np.hstack((A, np.identity(A.shape[0], dtype=np.double)))
        self.C = np.hstack((C, np.zeros(A.shape[0], dtype=np.double)))
        self.b = b

        self.m, self.n = self.A.shape
        self.eps = accuracy

    def solve(self, debug: bool = False) -> SimplexSolution:
        """Solve maximization linear programming problem using simplex method in matrices.

        Maximize z = c_1*x_1 + ... + c_n*x_n    
            subject to
                a_1_1*x_1 + ... + a_1_n*x_n <= b_1
                            ...
                a_n_1*x_1 + ... + a_n_n*x_n <= b_n
                all x_i >= 0

        Args:
            debug (bool): Enable printing debug info

        Returns:
            SimplexSolution: Solution to optimization problem
        """  # noqa: E501

        # [Step 0]
        # Constructing a starting basic feasible solution
        B = np.identity(self.m, dtype=np.double)
        C_B = np.zeros(self.m, dtype=np.double)
        basic = list(range(self.n - self.m, self.n))  # keeping track of basic variables

        while True:
            # [Step 1]
            # Finding the inverse of B
            B_inv = np.linalg.inv(B)

            # Compute some matrices that will be used later
            X_B = B_inv @ self.b
            z = C_B @ X_B
            C_B_times_B_inv = C_B @ B_inv

            if debug:
                print("-" * 10)
                print("BASIC:", basic)
                print("B:", B)
                print("X_B:", X_B)
                print("Z:", z)
                print("C_B:", C_B)
                print("C_B*B_inv", C_B_times_B_inv)

            # [Step 2]
            # Finding the entering variable
            entering_j, min_delta, cnt = 0, float("inf"), 0
            for j in range(self.n):
                # We are searching among nonbasic variables
                if j in basic:
                    continue

                # Calculate the delta
                P_j = self.A[:, [j]]
                z_j = C_B_times_B_inv @ P_j
                delta = z_j.item() - self.C[j]

                if debug:
                    print(j, z_j, self.C[j], delta)

                # We count how many deltas are greater than machine epsilon
                # to later see if we can exit the algorithm
                if delta >= self.eps:  # maximization
                    cnt += 1
                elif delta < min_delta - self.eps:
                    # Update current best candidate (most negative)
                    # for entering vector
                    min_delta, entering_j = delta, j

            if debug:
                print("CNT:", cnt)
                print("MIN DELTA:", min_delta)

            # We found the optimal solution
            # (z_j - c_j >= 0 for all nonbasic vectors)
            if cnt == self.n - self.m:
                # Example: entering epsilon of 0.001 means
                # rounding to 3 digits after the decimal point
                round_decimals = round(-np.log10(self.eps))
                X_decision = np.zeros(self.n - self.m)

                # Returning only the original variables
                # without slack variables
                for i, j in enumerate(basic):
                    if j < self.n - self.m:
                        X_decision[j] = round(X_B[i], round_decimals)

                return SimplexSolution(X_decision, round(z.item(), round_decimals))

            # [Step 3]
            # Again compute some matrices that will be used later
            P_j = self.A[:, [entering_j]]
            B_inv_times_P_j = B_inv @ P_j

            if debug:
                print("B_inv*P_j:", B_inv_times_P_j)

            # Condition for unbounded solution
            if np.all(B_inv_times_P_j <= self.eps):
                raise InfeasibleSolution
            else:
                # If everything is ok, then we do the ratio test
                # to find the leaving vector
                i, leaving_i, min_ratio = 0, 0, float("inf")
                for j in basic:
                    # According to ratio test, we need to find the minimal
                    # positive ratio
                    if B_inv_times_P_j[i] <= self.eps:
                        i += 1
                        continue

                    ratio = X_B[i] / B_inv_times_P_j[i].item()

                    if debug:
                        print(X_B[i], B_inv_times_P_j[i].item(), ratio)

                    # Update current best candidate (most minimal positive)
                    # for leaving vector
                    if ratio < min_ratio - self.eps:
                        min_ratio, leaving_i = ratio, j

                    i += 1

            if debug:
                print("MIN RATIO:", min_ratio)
                print("LEAVING, ENTERING:", leaving_i, entering_j)

            # [Step 4]
            # Forming the next basis by finding the leaving vector
            # replacing it with the entering vector.
            # Also update the objective vector coefficient of B
            for i in range(self.m):
                if basic[i] == leaving_i:
                    B[:, [i]] = P_j
                    basic[i] = entering_j
                    C_B[i] = self.C[entering_j]
                    break


def main():
    A, b, C = np.array(
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
    eps = 1e-9

    print(Simplex(A, b, C, eps).solve())


if __name__ == "__main__":
    main()
