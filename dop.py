import matplotlib.pyplot as plt
import numpy as np
import random



def generate_function(n, k):
    A = np.diag([1, k] + [random.uniform(1, k) for i in range(n - 2)])
    Q  = np.linalg.qr(np.random.rand(n, n))[0]
    return np.matmul(Q, np.matmul(A, np.transpose(Q)))
    
def random_func():
    ans = [random.randint(-5 ** (i + 1), 5 ** (i + 1)) for i in range(5)][::-1]
    for i in range(len(ans) - 1, -1, -1):
        if(ans[i] != 0):
            ans[i] = abs(ans[i])
            break
    return ans


def get_func(f):
    return (lambda x: sum([f[i] * (x**i) for i in range(len(f))])) 

def get_df(f):
    ans = []
    for i in range(1, len(f)):
        ans.append(i * f[i])
    return get_func(ans)

def search(start, f, grad, c1, c2):
    x = start
    ans = [x]
    for g in range(1000):
        a = find_a(x, f, grad, c1, c2)
        x = x - a * np.array(grad(x))
        ans.append(x)
        if np.linalg.norm(ans[-1] - ans[-2]) < 1e-9:
            break
    return ans

def find_a(x, f, grad, c1, c2):
    a = 0
    t = 1
    b = "inf"
    d = -np.array(grad(x))
    if np.linalg.norm(d) == 0:
        return 0
    while True:
        if(f(x + t * d) > f(x) + c1 * t * np.matmul(grad(x), d)):
            b = t
            t = (a + b) / 2
        elif(np.matmul(grad(x + t * d), np.transpose(d)) < c2 * np.matmul(grad(x), d)):
            a = t
            t = (2 * a) if b == "inf" else ((a + b) / 2)
        else:
            break
    #print(t)
    return t

def test():
    #function = random_func()
    # function = [0, 2, 1]
    f = (lambda x : (1 - x[0])**2 + (x[1] - x[0])**2)
    df = lambda x : [-2 * (1 - x[0]) - 2 * (x[1] - x[0]), 2 * (x[1] - x[0])]
        
    axis = np.linspace(-10, 10)
    #f = get_func(function)
    #plt.plot(axis, f(axis))
    ans = search([4, -4], f, df, 0.0001,  0.9)
    #print([[i, f(i)] for i in ans])
    axis = np.linspace(-20, 20)
    x, y = np.meshgrid(axis, axis)
    print(len(ans))
    plt.plot([i[0] for i in ans], [i[1] for i in ans], c="red")
    plt.contour(x, y, f([x, y]), levels=sorted([f(i) for i in ans]))
    plt.show()
    #plt.plot(ans, f(np.array(ans)), "red")
    #plt.show()