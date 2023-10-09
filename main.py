from simplex import Matrix, Vector
from simplex import Simplex, InfeasibleSolution


def main():
    print("Enter a vector of coefficients of objective function - C:")
    C: Vector = Vector()
    C.vInput()

    print("Enter a matrix of coefficients of constraint function - A:")
    A: Matrix = Matrix()
    A.mInput()

    print("Enter a vector of right-hand side numbers - b:")
    b: Vector = Vector()
    b.vInput()

    eps = float(input("Enter the approximation accuracy ε:\n"))

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
