from typing import List
from copy import deepcopy


class Matrix:
    _mat: List[List[float]] = []

    def __init__(self, arr: List[List[float]] = []):
        self._mat = deepcopy(arr)

    def mInput(self):
        while (s := input()) != "":
            row = list(map(float, s.split()))
            self._mat.append(row)

    def __mul__(self, matrix: 'Matrix') -> 'Matrix':
        if self.getWidth() != matrix.getHeight():
            raise ArithmeticError(
                "Cannot multiply matrices of these dimensions")

        result: list[list[float]] = [[sum(a * b for a, b in zip(self_row, matrix_col))
                                      for matrix_col in zip(*matrix.getMatrix())] for self_row in self._mat]

        return Matrix(result)

    def inverseMatrix(self) -> 'Matrix':
        if self.getHeight() != self.getWidth():
            raise ValueError("Cannot calculate inverse of nonsquare matrix")

        det = self.det()

        if self.getHeight() == 2:
            return Matrix([[self._mat[1][1] / det, -1 * self._mat[0][1] / det],
                           [-1 * self._mat[1][0] / det, self._mat[0][0] / det]])

        inv = Matrix()
        for i in range(self.getHeight()):
            row = []
            for j in range(self.getWidth()):
                row.append(((-1) ** (i + j)) * self.minorMatrix(i, j).det())
            inv._mat.append(row)
        inv = inv.mTranspose()
        matrix = inv.getMatrix()

        matrix = list(list(map(lambda x: x/det, row)) for row in matrix)

        return Matrix(matrix)

    def mTranspose(self):
        return Matrix(list(map(list, zip(*self._mat))))

    def m2vTranspose(self):
        if self.getWidth() != 1:
            raise ValueError("Cannot transpose to a column vector")
        return Vector([x[0] for x in self._mat])

    def minorMatrix(self, i: int, j: int) -> 'Matrix':
        return Matrix([row[:j] + row[j + 1:] for row in (self._mat[:i] + self._mat[i + 1:])])

    def det(self) -> float:
        if self.getHeight() != self.getWidth():
            raise ValueError("Cannot calculate inverse of nonsquare matrix")

        if self.getHeight() == 2:
            return self._mat[0][0] * self._mat[1][1] - self._mat[0][1] * self._mat[1][0]

        minor = Matrix()
        det = 0
        for i in range(self.getHeight()):
            minor = self.minorMatrix(0, i)
            det += ((-1) ** i) * self._mat[0][i] * minor.det()
        return det

    def getMatrix(self) -> List[List[float]]:
        return self._mat

    def hconcat(self, matrix: 'Matrix') -> 'Matrix':
        if self.getHeight() != matrix.getHeight():
            raise ArithmeticError(
                "Cannot hconcat matrices of different heights")

        result = Matrix(deepcopy(self._mat))
        for i in range(self.getHeight()):
            for j in matrix.getMatrix()[i]:
                result.getMatrix()[i].append(j)

        return result

    def getColumn(self, j: int) -> 'Vector':
        return Vector([x[j] for x in self._mat])

    def setColumn(self, j: int, vector: 'Vector'):
        for i in range(self.getHeight()):
            self._mat[i][j] = vector[i]

    def getHeight(self) -> int:
        return len(self._mat)

    def getWidth(self) -> int:
        return len(self._mat[0])

    def __getitem__(self, index):
        return self._mat[index]


class IdentityMatrix(Matrix):
    def __init__(self, length: int):
        self._mat = [[0]*length for _ in range(length)]
        for i in range(length):
            for j in range(length):
                if i == j:
                    self._mat[i][j] = 1


class ZeroMatrix(Matrix):
    def __init__(self, length: int):
        self._mat = [[0]*length for _ in range(length)]


class Vector(Matrix):
    def __init__(self, arr: List[float] = []):
        self._mat = [deepcopy(arr)]

    def __mul__(self, v2: 'Vector') -> float:
        dot = 0
        for i in range(len(v2.getVector())):
            dot += self._mat[0][i] * v2._mat[0][i]
        return dot

    def __xor__(self, matrix: 'Matrix') -> 'Vector':
        if self.getWidth() != matrix.getHeight():
            raise ArithmeticError(
                "Cannot multiply matrices of these dimensions")

        result: list[list[float]] = [[sum(a * b for a, b in zip(self_row, matrix_col))
                                      for matrix_col in zip(*matrix.getMatrix())] for self_row in self._mat]

        return Vector(result[0])

    def vInput(self):
        s = input()
        self._mat[0] = [float(x) for x in s.split()]
        # For elimination of null string after vector
        input()

    def vTranspose(self) -> 'Matrix':
        return Matrix([[x] for x in self._mat[0]])

    def getVector(self) -> List[float]:
        return self._mat[0]

    def hconcat(self, vector: 'Vector') -> 'Vector':
        result = Vector(deepcopy(self._mat[0]))
        for j in vector.getMatrix()[0]:
            result.getMatrix()[0].append(j)

        return result

    def __getitem__(self, index):
        return self._mat[0][index]

    def __setitem__(self, index, value):
        self._mat[0][index] = value
