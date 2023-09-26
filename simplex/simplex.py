from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class SimplexSolution:
    """Stores a solution to simplex method.

    Args:
        decision_variables (npt.NDArray[np.double]): Vector of decision variables
        value (float): Solution to maximization problem
    """

    decision_variables: npt.NDArray[np.double]
    value: float


class InfeasibleSolution(Exception):
    """Raised when no feasible solution is found."""

    def __init__(self, message="No feasible solution is found"):
        self.message = message
        super().__init__(self.message)


class Simplex:
    def __init__(
        self,
        A: npt.NDArray[np.double],
        b: npt.NDArray[np.double],
        C: npt.NDArray[np.double],
        eps: float = 1e-4,
    ):
        """Initialization of simplex method solver.

        Args:
            A (npt.NDArray[np.double]): (self.m x self.n) matrix representing the constraint coefficients
            b (npt.NDArray[np.double]): self.m-column vector representing the constraint right-hand side
            C (npt.NDArray[np.double]): self.n-vector representing the objective-function coefficients
            eps (float, optional): Approximation accuracy. Defaults to 1e-4.
        """  # noqa: E501
        self.A = np.hstack((A, np.identity(A.shape[0])))
        self.C = np.hstack((C, np.zeros(A.shape[0])))
        self.b = b
        self.eps = eps
        self.m, self.n = self.A.shape

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

        # Step 0
        B = np.identity(self.m)
        C_B = np.zeros(self.m)
        basic = list(range(self.n - self.m, self.n))
        while True:
            # Step 1
            B_inv = np.linalg.inv(B)
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

            # Step 2
            entering_j = 0
            min_delta = float("inf")
            cnt = 0
            for j in range(self.n):
                if j in basic:
                    continue
                P_j = self.A[:, [j]]
                z_j = C_B_times_B_inv @ P_j
                delta = z_j.item() - self.C[j]

                if debug:
                    print(j, z_j, self.C[j], delta)

                if delta >= self.eps:  # maximization
                    cnt += 1
                else:
                    if delta < min_delta - self.eps:
                        min_delta = delta
                        entering_j = j

            if debug:
                print("CNT:", cnt)
                print("MIN DELTA:", min_delta)

            if cnt == self.n - self.m:
                round_decimals = round(-np.log10(self.eps))
                X_decision = np.zeros(self.n - self.m)
                for i, j in enumerate(basic):
                    if j < self.n - self.m:
                        X_decision[j] = round(X_B[i], round_decimals)
                return SimplexSolution(X_decision, round(z.item(), round_decimals))
            # Step 3
            P_j = self.A[:, [entering_j]]
            B_inv_times_P_j = B_inv @ P_j

            if debug:
                print("B_inv*P_j:", B_inv_times_P_j)

            if np.all(B_inv_times_P_j <= self.eps):
                raise InfeasibleSolution
            else:
                i = 0
                leaving_i = 0
                min_ratio = float("inf")
                for j in basic:
                    if B_inv_times_P_j[i] <= self.eps:
                        i += 1
                        continue
                    ratio = X_B[i] / B_inv_times_P_j[i].item()

                    if debug:
                        print(X_B[i], B_inv_times_P_j[i].item(), ratio)

                    if ratio < min_ratio - self.eps:
                        min_ratio = ratio
                        leaving_i = j
                    i += 1

            if debug:
                print("MIN RATIO:", min_ratio)
                print("LEAVING, ENTERING:", leaving_i, entering_j)

            # Step 4
            for i in range(self.m):
                if basic[i] == leaving_i:
                    B[:, [i]] = P_j
                    basic[i] = entering_j
                    C_B[i] = self.C[entering_j]
                    break

        raise InfeasibleSolution


def main():
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
    eps = 1e-9
    simplex = Simplex(A, b.T, C, eps)
    solution = simplex.solve()
    print(solution)


if __name__ == "__main__":
    main()
