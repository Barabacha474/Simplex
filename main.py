from numpy import array, double
from simplex import Simplex, InfeasibleSolution


def main():
    m = int(input("Enter number of constraints: "))

    C: list[float] = [float(x) for x in input("Enter objective function coefficients vector: ").split()]

    print("Enter constraints coefficients matrix:")
    A: list[list[float]] = [[float(x) for x in input().split()] for _ in range(m)]

    b: list[float] = [float(x) for x in input("Enter constraints right hand side vector: ").split()]

    eps = float(input("Enter approximation accuracy: "))

    solver = Simplex(array(A, dtype=double), array(b, dtype=double), array(C, dtype=double), eps)

    try:
        solution = solver.solve()
    except InfeasibleSolution:
        print("Infeasible solution")
    else:
        print("Decision variables x*:", *solution.decision_variables)
        print("Value:", solution.value)


if __name__ == "__main__":
    main()
