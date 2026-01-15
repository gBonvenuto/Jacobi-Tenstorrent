import numpy as np
import matplotlib.pyplot as plt

import sys
sys.setrecursionlimit(2000)

# Esse script serve para eu aprender a fazer um algoritmo simples de
# Jacobi

max_it = 1000
it = 0


def save_to_file(matrix, filename):
    # Flatten the matrix (row-major order, C-style) and save as raw float32 data.
    # This format is easy to read linearly in C++.
    matrix.astype(np.float32).tofile(filename)


def read_from_file(filename, shape):
    # Read raw float32 data from file and reshape it.
    return np.fromfile(filename, dtype=np.float32).reshape(shape)


def jacobi(p0):
    global max_it
    global it
    pnew = p0.copy()
    if it > max_it:
        print("Chegamos na iteração máxima, retornando...")
        return p0
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            pnew[i, j] = 0.25*(
                p0[i-1, j]  # esquerda
                + p0[i+1, j]  # direita
                + p0[i, j-1]  # baixo
                + p0[i, j+1]  # cima
            )
    it = it + 1
    return jacobi(pnew)


# Grid parameters.
nx = 32                  # number of points in the x direction
ny = 32                  # number of points in the y direction
xmin, xmax = 0.0, 1.0     # limits in the x direction
ymin, ymax = -0.5, 0.5    # limits in the y direction
lx = xmax - xmin          # domain length in the x direction
ly = ymax - ymin          # domain length in the y direction
dx = lx / (nx-1)          # grid spacing in the x direction
dy = ly / (ny-1)          # grid spacing in the y direction

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y, indexing='ij')


# p0 = (np.sin(np.pi*X)*np.cos(np.pi*Y)
#       + np.sin(5.0*np.pi*X)*np.cos(5.0*np.pi*Y))
p0 = np.ones((nx, ny), dtype=np.float32)
save_to_file(p0, "input.bin")
print(p0[-1])

plt.figure()
# Usa pcolormesh para criar um mapa de cores 2D de b sobre a grade (X, Y)
plt.imshow(p0, origin="upper", interpolation="nearest")
plt.title('RHS (b) da equação de Jacobi/Poisson')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Valor de b')
plt.show()

# Compute the exact solution
# p_e = jacobi(p0)
p_e = read_from_file("output.bin", p0.shape)
plt.figure()
# Usa pcolormesh para criar um mapa de cores 2D de b sobre a grade (X, Y)
# plt.pcolormesh(X, Y, p_e, shading='auto')
plt.imshow(p_e, origin="upper", interpolation="nearest")
plt.title('RHS (b) da equação de Jacobi/Poisson')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Valor de b')
plt.show()
