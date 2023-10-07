class Matrix:
    mat = []

    def mInput(self):
        s = input()
        while s != "":
            row = [int(x) for x in s.split()]
            self.mat.append(row)
            s = input()
        return self

    def multiplpyM(self, matrix):
        result = [[sum(a * b for a, b in zip(self_row, matrix_col))
                   for matrix_col in zip(*matrix)] for self_row in self.mat]
        return result

    def inverseMatrix(self):
        det = self.det()

        if len(self.mat) == 2:
            return [[self.mat[1][1] / det, -1 * self.mat[0][1] / det],
                    [-1 * self.mat[1][0] / det, self.mat[0][0] / det]]

        inv = Matrix()
        for i in range(len(self.mat)):
            row = []
            for j in range(len(self.mat)):
                row.append(((-1) ** (i + j)) * self.det())
            inv.mat.append(row)
        inv.mat = inv.mTranspose()
        matrix = inv.gerMatrix()
        det = inv.det()
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                matrix[i][j] = matrix[i][j] / det

        return matrix

    def mTranspose(self):
        return map(list, zip(*self.mat))

    def minorMatrix(self, i, j):
        return [row[:j] + row[j + 1:] for row in (self.mat[:i] + self.mat[i + 1:])]

    def det(self):
        if len(self.mat) == 2:
            return self.mat[0][0] * self.mat[1][1] - self.mat[0][1] * self.mat[1][0]

        minor = Matrix()
        det = 0
        for i in range(len(self.mat)):
            minor.mat = self.minorMatrix(0, i)
            det += ((-1) ** i) * self.mat[0][i] * minor.det()
        return det

    def gerMatrix(self):
        return self.mat


class IdentityMatrix(Matrix):
    def __init__(self, len):
        for i in range(len):
            for j in range(len):
                if i == j:
                    self.mat[i][j] = 1
                else:
                    self.mat[i][j] = 0


class ZeroMatrix(Matrix):
    def __init__(self, len):
        for i in range(len):
            for j in range(len):
                self.mat[i][j] = 0


class Vector:
    v = []

    def vInput(self):
        s = input()
        self.v = [int(x) for x in s.split()]
        """For elimination of null string after vector"""
        s = input()
        return self

    def dot(self, v2):
        dot = 0
        for i in range(len(v2)):
            dot += self.v[i] * v2.v[i]
        return dot

    def vTranspose(self):
        return list(zip(*self.v))

    def getVector(self):
        return self.v
