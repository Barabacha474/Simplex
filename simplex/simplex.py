from math import log10
from dataclasses import dataclass
from simplex.models import Matrix, IdentityMatrix, Vector
from simplex.exceptions import InfeasibleSolution


@dataclass
class SimplexSolution:
    """Stores a solution to simplex method.

    Args:
        decision_variables (Vector): Vector of decision variables
        value (float): Solution to maximization problem
    """

    decision_variables: Vector
    value: float


class Simplex:
    def __init__(self,
                 A: Matrix,
                 b: Vector,
                 C: Vector,
                 accuracy: float = 1e-4):
        """Initialization of simplex method solver.

        Args:
            A (Matrix): (m x n) matrix representing the constraint coefficients
            b (Vector): m-column vector representing the constraint right-hand side
            C (Vector): n-row vector representing the objective-function coefficients
            accuracy (float, optional): Approximation accuracy. Defaults to 1e-4.
        """

        # Sanity checks for correct input
        assert isinstance(A, Matrix), "A is not a matrix"
        assert isinstance(b, Vector), "b is not a vector"
        assert isinstance(C, Vector), "C is not a vector"
        assert A.getHeight() == b.getWidth(
        ), "Length of vector b does not correspond to # of rows of matrix A"
        assert A.getWidth() == C.getWidth(
        ), "Length of vector C does not correspond to # of cols of matrix A"
        assert accuracy > 0, "Accuracy should be greater than zero"

        # Add slack variables
        self.A: Matrix = A.hconcat(IdentityMatrix(A.getHeight()))
        self.C: Vector = C.hconcat(Vector([0]*A.getHeight()))
        self.b: Vector = b

        self.m, self.n = self.A.getHeight(), self.A.getWidth()
        self.eps = accuracy

    def solve(self, debug: bool = False) -> SimplexSolution:
        """Solve maximization linear programming problem using simplex method in matrices.

        Input is in standard form.

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
        B: Matrix = IdentityMatrix(self.m)
        C_B: Vector = Vector([0]*self.m)
        # keeping track of basic variables
        basic: list[int] = list(range(self.n - self.m, self.n))

        prev_z = None

        while True:
            # [Step 1]
            # Finding the inverse of B
            try:
                B_inv: Matrix = B.inverseMatrix()
            except Exception as e:
                # If we can't find the inverse, then something wrong with input
                raise InfeasibleSolution from e

            # Compute some matrices that will be used later
            X_B: Vector = (B_inv * self.b.vTranspose()).m2vTranspose()
            z: float = C_B * X_B
            C_B_times_B_inv: Vector = C_B ^ B_inv

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
                P_j: Vector = self.A.getColumn(j)
                z_j: float = C_B_times_B_inv * P_j
                delta = z_j - self.C[j]

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
            round_decimals = round(-log10(self.eps))
            if cnt == self.n - self.m or prev_z is not None and round(z, round_decimals) == round(prev_z, round_decimals):
                # Example: entering epsilon of 0.001 means
                # rounding to 3 digits after the decimal point
                X_decision: Vector = Vector([0]*(self.n - self.m))
                # Returning only the original variables
                # without slack variables
                for i, j in enumerate(basic):
                    if j < self.n - self.m:
                        X_decision[j] = round(X_B[i], round_decimals)

                if prev_z is not None and round(z, round_decimals) == round(prev_z, round_decimals):
                    print("Alternative optima has been detected\nInfinite number of solutions, there is one of them:\n")
                return SimplexSolution(X_decision, round(z, round_decimals))

            prev_z = z



            # [Step 3]
            # Again compute some matrices that will be used later
            P_j: Vector = self.A.getColumn(entering_j)
            B_inv_times_P_j: Vector = (B_inv * P_j.vTranspose()).m2vTranspose()

            if debug:
                print("B_inv*P_j:", B_inv_times_P_j)

            # Condition for unbounded solution
            if all([x <= self.eps for x in B_inv_times_P_j.getVector()]):
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

                    ratio = X_B[i] / B_inv_times_P_j[i]

                    if debug:
                        print(X_B[i], B_inv_times_P_j[i], ratio)

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
                    B.setColumn(i, P_j)
                    basic[i] = entering_j
                    C_B[i] = self.C[entering_j]
                    break