from copy import deepcopy


class Matrix:
    _mat = list([])

    def __init__(self, arr=list([])):
        self._mat = arr

    def mInput(self):
        while (s := input()) != "":
            row = list(map(float, s.split()))
            self._mat.append(row)

    def __mul__(self, matrix: 'Matrix') -> 'Matrix':
        result = [[sum(a * b for a, b in zip(self_row, matrix_col))
                   for matrix_col in zip(*matrix.getMatrix())] for self_row in self._mat]
        return Matrix(result)

    def inverseMatrix(self) -> 'Matrix':
        det = self.det()

        if len(self._mat) == 2:
            return Matrix([[self._mat[1][1] / det, -1 * self._mat[0][1] / det],
                           [-1 * self._mat[1][0] / det, self._mat[0][0] / det]])

        inv = Matrix()
        for i in range(len(self._mat)):
            row = []
            for j in range(len(self._mat)):
                row.append(((-1) ** (i + j)) * self.minorMatrix(i, j).det())
            inv._mat.append(row)
        inv = inv.mTranspose()
        matrix = inv.getMatrix()

        matrix = list(list(map(lambda x: x/det, row)) for row in matrix)

        return Matrix(matrix)

    def mTranspose(self):
        return Matrix(list(map(list, zip(*self._mat))))

    def minorMatrix(self, i: int, j: int) -> 'Matrix':
        return Matrix([row[:j] + row[j + 1:] for row in (self._mat[:i] + self._mat[i + 1:])])

    def det(self) -> float:
        if len(self._mat) == 2:
            return self._mat[0][0] * self._mat[1][1] - self._mat[0][1] * self._mat[1][0]

        minor = Matrix()
        det = 0
        for i in range(len(self._mat)):
            minor = self.minorMatrix(0, i)
            det += ((-1) ** i) * self._mat[0][i] * minor.det()
        return det

    def getMatrix(self) -> list:
        return self._mat

    def concat(self, matrix: 'Matrix') -> 'Matrix':
        result = deepcopy(self)
        for i in range(len(self._mat)):
            for j in matrix.getMatrix()[i]:
                result.getMatrix()[i].append(j)

        return result


class IdentityMatrix(Matrix):
    def __init__(self, length):
        self._mat = [[0]*length]*length
        for i in range(length):
            for j in range(length):
                if i == j:
                    self._mat[i][j] = 1


class ZeroMatrix(Matrix):
    def __init__(self, length):
        self._mat = [[0]*length]*length


class Vector:
    _v = []

    def __init__(self, arr=list()):
        self._v = arr

    def vInput(self):
        s = input()
        self._v = [float(x) for x in s.split()]
        # For elimination of null string after vector
        input()

    def __mul__(self, v2: 'Vector') -> float:
        dot = 0
        for i in range(len(v2.getVector())):
            dot += self._v[i] * v2._v[i]
        return dot

    def vTranspose(self) -> 'Vector':
        return Vector(list(zip(*self._v)))

    def getVector(self):
        return self._v
