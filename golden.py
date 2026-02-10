import sys
import numpy as np
import torch
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
    matrix.detach().cpu().to(torch.float32).numpy().tofile(filename)


def read_from_file(filename, shape):
    # Read raw float32 data from file and reshape it.
    data = np.fromfile(filename, dtype=np.float32).reshape(shape)
    return torch.from_numpy(data).to(torch.bfloat16)


def jacobi(p0, mode=MODE, compare_with_torch=True):
    global max_it

    # p0 deve ser bfloat16
    if not isinstance(p0, torch.Tensor):
        p0 = torch.tensor(p0, dtype=torch.bfloat16)

    if mode == "tenstorrent":
        print("running tenstorrent")
        cmd = ["bash", exec_path, f"{max_it}"]
        subprocess.run(cmd, check=True)

        result = read_from_file("output.bin", p0.shape)

        plt.figure()
        plt.imshow(result.to(torch.float32).cpu().numpy(),
                   origin="upper", interpolation="nearest")
        plt.title(f"Output (mode: {mode})")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()

        if compare_with_torch:

            torch_result = jacobi(p0, mode="python")

            # Comparação com tolerância
            iguais = torch.allclose(
                result, torch_result, rtol=1e-02, atol=1e-03)

            if iguais:
                print("O Tenstorrent é igual ao golden model em torch ✅")
            else:
                print("O resultado do Tenstorrent é diferente do golden model "
                      "em torch ❌")
                diff = torch.norm(result - torch_result) / \
                    torch.norm(torch_result)
                print(f"Erro Relativo L2: {diff.item():.4f}")

        return result
    elif mode == "python":
        it = 0
        while (it < max_it):
            print(f"iteração: {it}")
            pnew = p0.clone()

            # Não sei se isso é necessário, mas to fazendo pra garantir
            um_quarto = torch.tensor(0.25, dtype=torch.bfloat16)

            pnew[1:-1, 1:-1] = um_quarto * (p0[0:-2, 1:-1] + p0[2:, 1:-1] +
                                            p0[1:-1, 0:-2] + p0[1:-1, 2:])
            it = it + 1
            p0 = pnew.clone()

        plt.figure()
        plt.imshow(p0.to(torch.float32).cpu().numpy(),
                   origin="upper", interpolation="nearest")
        plt.title(f"Output (mode: {mode})")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()

        return p0
    else:
        print(f"UNKOWN MODE: {MODE}")


def criar_quadrado(tamanho_quadrado_x, tamanho_quadrado_y):
    p0 = torch.zeros((ny, nx), dtype=torch.bfloat16)
    centro_y, centro_x = ny // 2, nx // 2

    inicio_y = centro_y - (tamanho_quadrado_y // 2)
    fim_y = centro_y + (tamanho_quadrado_y // 2)
    inicio_x = centro_x - (tamanho_quadrado_x // 2)
    fim_x = centro_x + (tamanho_quadrado_x // 2)

    p0[inicio_y:fim_y, inicio_x:fim_x] = 1.0
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
plt.imshow(p0.to(torch.float32).cpu().numpy(),
           origin="upper", interpolation="nearest")
plt.title("Input")
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()

# Compute the exact solution
p_e = jacobi(p0)

plt.show()
