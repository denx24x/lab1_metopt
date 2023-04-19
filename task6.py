from task5 import generate_function, get_f
import numpy as np
import random
import matplotlib.pyplot as plt

def dichotomy(l, r, f, eps):
    directional = (r - l) / np.linalg.norm(r - l)
    while abs(np.linalg.norm(r - l)) >= eps:
        mid = (l + r) / 2 
        m1 = mid - directional * (eps / 4)
        m2 = mid + directional * (eps / 4)
        if f(m1) < f(m2):
            r = m2
        else:
            l = m1
    return l

def withDich(f, grad, n, eps):
    x = np.array([10] * n)
    points = [x]
    inf = 10000
    cnt = 0
    for i in range(inf):
        cnt += 1
        g = np.array(grad(x))
        old = x
        cur = eps
        while f(x) > f(x - g * cur):
            cur *= 2
        x = dichotomy(x, x - g * cur, f, eps)
        if(np.linalg.norm(x - old) < eps):
            break
        points.append(x)
    return points, cnt

def get_df(matrix):
    n = np.copy(matrix)
    for i in range(matrix.shape[0]):
        for g in range(matrix.shape[1]):
            if g > i:
                n[i][g] += n[g][i]
                n[g][i] = n[i][g]
    return (lambda x : [sum([((n[i][g] * x[g]) if i != g else (2 * n[i][g] * x[g])) for g in range(matrix.shape[1])]) for i in range(matrix.shape[0])])

def T(n, k):
    matrix = generate_function(n, k)
    #f = (lambda x : (1 - x[0])**2 + (x[1] - x[0])**2)
    #df = lambda x : [-2 * (1 - x[0]) - 2 * (x[1] - x[0]), 2 * (x[1] - x[0])]#f = (lambda x : 6 * x[0]**2 + 2 * x[1]**2)
    #df = (lambda x : [12 * x[0], 4 * x[1]])
    ans, cnt = withDich(get_f(matrix), get_df(matrix), n, 0.0001)
    #ans, cnt = withDich(f, df, n, 0.0001)
    
    #axis = np.linspace(-20, 20)
    #x, y = np.meshgrid(axis, axis)

    #ax = plt.figure().add_subplot(projection="3d")
    #ax.plot_surface(x, y, get_f(matrix)([x, y]), cmap='viridis', edgecolor='none')
    #plt.show()

    #plt.plot([i[0] for i in ans], [i[1] for i in ans], c="red")
    #plt.contour(x, y, get_f(matrix)([x, y]), levels=sorted([get_f(matrix)(i) for i in ans]))
    #plt.contour(x, y, f([x, y]), levels=sorted([f(i) for i in ans]))
    #plt.show()
    #print(cnt)
    #print(ans)
    return cnt

def get_data():
    NS =[2, 3, 4, 5, 6, 7, 8, 9, 10]
    KS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    import sys
    sys.stdout = open("out.csv", "w")
    print('k/n', end=';')
    print(';'.join(map(str, NS)))
    for k in KS:
        data = [sum([T(n, k)  for g in range(10)]) / 10 for n in NS]
        print(k, end=';')
        print(';'.join(map(str, data)))


#for i in range(10000):
#    print(T(2, 8))
#x1x1 + x1x2 + 2x2x2
# x1 + x2
# x1 + 4x2
# []