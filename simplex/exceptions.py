class InfeasibleSolution(Exception):
    """Raised when no feasible solution is found."""

    def __init__(self, message="No feasible solution is found"):
        self.message = message
        super().__init__(self.message)
