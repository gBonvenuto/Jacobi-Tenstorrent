import sys
import numpy as np
import matplotlib.pyplot as plt
import subprocess

exec_path = "./run_script.sh"

# MODES:
# - python
# - tenstorrent
MODE = "tenstorrent"
print("MODE:", MODE)

sys.setrecursionlimit(2000)

# Esse script serve para eu aprender a fazer um algoritmo simples de
# Jacobi

max_it = 1
it = 0


def save_to_file(matrix, filename):
    # Flatten the matrix (row-major order, C-style) and save as raw float32 data.
    # This format is easy to read linearly in C++.
    matrix.astype(np.float32).tofile(filename)


def read_from_file(filename, shape):
    # Read raw float32 data from file and reshape it.
    return np.fromfile(filename, dtype=np.float32).reshape(shape)


def jacobi(p0):
    global MODE
    if MODE == "tenstorrent":
        print("running tenstorrent")
        cmd = ["bash", exec_path]

        with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              bufsize=1) as processo:
            for linha in processo.stdout:
                print(linha, end='', flush=True)

        return read_from_file("output.bin", p0.shape)
    if MODE == "python":
        global max_it
        global it
        pnew = p0.copy()
        if it >= max_it:
            print("Chegamos na iteração máxima, retornando...")
            return p0
        pnew[1:-1, 1:-1] = 0.25 * (p0[0:-2, 1:-1] + p0[2:, 1:-1] +
                                   p0[1:-1, 0:-2] + p0[1:-1, 2:])
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

tamanho_matriz = 32
tamanho_quadrado = 10  # Você pode mudar esse valor

# 2. Crie a matriz de zeros (fundo preto)
p0 = np.zeros((tamanho_matriz, tamanho_matriz), dtype=np.float32)

# 3. Calcule onde o quadrado começa e termina para ficar centralizado
inicio = (tamanho_matriz - tamanho_quadrado) // 2
fim = inicio + tamanho_quadrado

# 4. Aplique o valor 1 na região central (fatiamento)
p0[inicio:fim, inicio:fim] = 1

# --- Visualização Simples no Terminal ---
# Como 32x32 é grande, vamos imprimir de um jeito que dê para ver o desenho
for linha in p0:
    print(' '.join(str(x) for x in linha).replace('0', '.').replace('1', '#'))

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
p_e = jacobi(p0)
plt.figure()
# Usa pcolormesh para criar um mapa de cores 2D de b sobre a grade (X, Y)
# plt.pcolormesh(X, Y, p_e, shading='auto')
plt.imshow(p_e, origin="upper", interpolation="nearest")
plt.title('RHS (b) da equação de Jacobi/Poisson')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Valor de b')
plt.show()
