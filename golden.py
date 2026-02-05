import sys
import numpy as np
import matplotlib.pyplot as plt
import subprocess

exec_path = "./run_script.sh"

# MODES:
# - python
# - tenstorrent
MODE = "tenstorrent"

max_it = 200

print(f"Mode: {MODE}, max_iterations: {max_it}")

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
        cmd = ["bash", exec_path, f"{max_it}"]
        subprocess.run(cmd, check=True)

        return read_from_file("output.bin", p0.shape)
    elif MODE == "python":
        it = 0
        while (it < max_it):
            print(f"iteração: {it}")
            pnew = p0.copy()
            pnew[1:-1, 1:-1] = 0.25 * (p0[0:-2, 1:-1] + p0[2:, 1:-1] +
                                       p0[1:-1, 0:-2] + p0[1:-1, 2:])
            it = it + 1
            p0 = pnew.copy()
        return p0
    else:
        print(f"UNKOWN MODE: {MODE}")


def criar_quadrado(tamanho_quadrado_x, tamanho_quadrado_y):
    p0 = np.zeros((ny, nx), dtype=np.float32)

    centro_y, centro_x = ny // 2, nx // 2

    # Fatiamento (Slicing) para desenhar o quadrado
    inicio_y = centro_y - (tamanho_quadrado_y // 2)
    fim_y = centro_y + (tamanho_quadrado_y // 2)
    inicio_x = centro_x - (tamanho_quadrado_x // 2)
    fim_x = centro_x + (tamanho_quadrado_x // 2)

    p0[inicio_y:fim_y, inicio_x:fim_x] = 1
    return p0


# Grid parameters
nx = 64                  # largura (colunas)
ny = 64                  # altura (linhas)
xmin, xmax = 0.0, nx
ymin, ymax = 0.0, ny
lx = xmax - xmin
ly = ymax - ymin
dx = lx / (nx-1)
dy = ly / (ny-1)

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y, indexing='xy')

p0 = criar_quadrado(32, 32)

# p0 = (np.sin(2 * np.pi * X / lx) * np.cos(2 * np.pi * Y / ly) +
#       0.5 * np.sin(4 * np.pi * X / lx)).astype(np.float32)

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
