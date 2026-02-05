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

max_it = 2
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
    global max_it
    if MODE == "tenstorrent":
        print("running tenstorrent")
        cmd = ["bash", exec_path]
        subprocess.run(cmd, check=True)

        return read_from_file("output.bin", p0.shape)
    if MODE == "python":
        global it
        pnew = p0.copy()
        if it >= max_it:
            print("Chegamos na iteração máxima, retornando...")
            return p0
        pnew[1:-1, 1:-1] = 0.25 * (p0[0:-2, 1:-1] + p0[2:, 1:-1] +
                                   p0[1:-1, 0:-2] + p0[1:-1, 2:])
        it = it + 1
        return jacobi(pnew)


# Grid parameters
nx = 32                  # largura (colunas)
ny = 32                  # altura (linhas)
xmin, xmax = 0.0, nx    # Aumentei o domínio X para manter a proporção da malha
ymin, ymax = 0.0, ny    # Ajustei Y para começar do 0 (mais intuitivo)
lx = xmax - xmin
ly = ymax - ymin
dx = lx / (nx-1)
dy = ly / (ny-1)

x = np.linspace(xmin, xmax)
y = np.linspace(ymin, ymax)
X, Y = np.meshgrid(x, y, indexing='xy')

# --- Inicialização da Matriz ---
# Numpy usa formato (linhas, colunas) -> (ny, nx)
p0 = np.zeros((ny, nx), dtype=np.float32)

# Definindo o quadrado centralizado dinamicamente
tamanho_quadrado_x = 10
tamanho_quadrado_y = 10

centro_y, centro_x = ny // 2, nx // 2

# Fatiamento (Slicing) para desenhar o quadrado
inicio_y = centro_y - (tamanho_quadrado_y // 2)
fim_y = centro_y + (tamanho_quadrado_y // 2)
inicio_x = centro_x - (tamanho_quadrado_x // 2)
fim_x = centro_x + (tamanho_quadrado_x // 2)

p0[inicio_y:fim_y, inicio_x:fim_x] = 1

# --- Visualização Simples no Terminal ---
# Como 32x32 é grande, vamos imprimir de um jeito que dê para ver o desenho
for linha in p0:
    print(' '.join(str(x) for x in linha).replace('0', '.').replace('1', '#'))

save_to_file(p0, "input.bin")
print(p0[-1])

plt.figure()
# Usa pcolormesh para criar um mapa de cores 2D de b sobre a grade (X, Y)
plt.imshow(p0, origin="upper", interpolation="nearest")
plt.title("Input")
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
plt.title("Output")
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Valor de b')
plt.show()
