import numpy as np
import random

def generate_function(n, k):
    A = np.diag([1, k] + [random.uniform(1, k) for i in range(n - 2)])
    Q  = np.linalg.qr(np.random.rand(n, n))[0]
    return np.matmul(Q, np.matmul(A, np.transpose(Q)))

def get_representation(matrix):
    ans = []
    for i in range(matrix.shape[0]):
        for g in range(i, matrix.shape[1]):
            val = matrix[i][g] + matrix[g][i]
            x = 'x' + str(i + 1) + 'x' + str(g + 1)
            ans.append(str(val) + x)
    return ' + '.join(ans)

def get_f(matrix):
    return (lambda x: sum([sum([matrix[i][g] * x[i] * x[g] for g in range(matrix.shape[1])]) for i in range(matrix.shape[0])]))

#print(get_representation(generate_function(3, 5)))