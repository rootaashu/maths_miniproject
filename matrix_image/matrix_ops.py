import numpy as np 

def identity(n):

    I = np.zeros((n,n))
    for i in range(n):
        I[i,i] = 1.0 
        return I 

def transpose(A):
    A = np.asarray(A, dtype=float)
    m, n = A.shape 
    T = np.zeros((n,n))
    for i in range(m):
        for j in range(n):
            T[j, i] = A[i,j]
            return T 

def mutmul(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    m, n = A.shape 
    n2, p = B.shape 
    if n != n2:
        raise ValueError(f"Shape mismatch: {A.shape} x {B.shape}")
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


def determinant(A):
    M = np.array(A, dtype=float)
    n, m = M.shape
    if n != m:
        raise ValueError("Determinant needs a square matrix")
    det = 1.0
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(M[col:, col])))
        if abs(M[pivot, col]) < 1e-12:
            return 0.0
        if pivot != col:
            M[[col, pivot]] = M[[pivot, col]]
            det = -det  # a row swap flips the sign
        det *= M[col, col]
        for r in range(col + 1, n):
            M[r, col:] -= (M[r, col] / M[col, col]) * M[col, col:]
    return det
def inverse(A):
    A = np.asarray(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("Inverse needs a square matrix")
    aug = np.hstack([A, identity(n)])
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(aug[col:, col])))
        if abs(aug[pivot, col]) < 1e-12:
            raise ValueError("Matrix is singular (not invertible)")
        if pivot != col:
            aug[[col, pivot]] = aug[[pivot, col]]
        aug[col] /= aug[col, col]  # make pivot 1
        for r in range(n):
            if r != col:
                aug[r] -= aug[r, col] * aug[col]  # clear the column
    return aug[:, n:]
