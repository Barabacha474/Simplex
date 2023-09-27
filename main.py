import numpy as np
from simplex import Simplex, InfeasibleSolution


def main():
    m = int(input("Enter number of constraints: "))

    C = list(
        map(float, input("Enter objective function coefficients vector: ").split())
    )

    print("Enter constraints coefficients matrix:")
    A = [list(map(float, input().split())) for _ in range(m)]

    b = list(map(float, input("Enter constraints right hand side vector: ").split()))

    eps = float(input("Enter approximation accuracy: "))

    solver = Simplex(
        np.array(A, dtype=np.double),
        np.array(b, dtype=np.double),
        np.array(C, dtype=np.double),
        eps,
    )

    try:
        solution = solver.solve()
    except InfeasibleSolution:
        print("Infeasible solution")
    else:
        print("Decision variables x*:", *solution.decision_variables)
        print("Value:", solution.value)


if __name__ == "__main__":
    main()
