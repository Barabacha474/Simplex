from simplex import Matrix, Vector
from simplex import Simplex, InfeasibleSolution


def main():
    print("Enter objective function coefficients vector: ")
    C: Vector = Vector()
    C.vInput()

    print("Enter constraints coefficients matrix:")
    A: Matrix = Matrix()
    A.mInput()

    print("Enter constraints right hand side vector: ")
    b: Vector = Vector()
    b.vInput()

    eps = float(input("Enter approximation accuracy: "))

    solver = Simplex(A, b, C, eps)

    try:
        solution = solver.solve()
    except InfeasibleSolution:
        print("Infeasible solution")
    else:
        print("Decision variables x*:", *solution.decision_variables.getVector())
        print("Value:", solution.value)


if __name__ == "__main__":
    main()
