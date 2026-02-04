Vou tentar fazer um algoritmo de Stencil na placa da Tenstorrent Blackhole.

No momento não tenho acesso à placa física, então estarei utilizando o 
simulador [ttsim](https://github.com/tenstorrent/ttsim).

# O que é o algoritmo de Jacobi?

Algumas coisas que dei uma lida:
- [Higher-dimentional discretizations](https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/05_IterativeMethods/05_01_Iteration_and_2D.html)
- [Jacobi method](https://en.wikipedia.org/wiki/Jacobi_method)
- [Poisson's equation](https://en.wikipedia.org/wiki/Poisson's_equation)
- [Iterative Stencil Loops](https://en.wikipedia.org/wiki/Iterative_Stencil_Loops)

Fiquei interessado [nisso aqui também para ler depois](https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/04_PartialDifferentialEquations/04_01_Advection.html)

Um algoritmo é um Stencil quando, para calcular um valor, o valor de seus vizinhos
é necessário.

O método de Jacobi é um Stencil simples, simplificadamente, ele possui o
objetivo de suavizar a matriz (e isso pode ser utilizado para simulações
físicas de distribuição de fluidos, por exemplo).

Cada célula é calculada com base na média dos valores das celulas à direita,
à esquerda, acima e abaixo.

# Objetivos

1. Criar o algoritmo em Python para validar se entendi corretamente o que este algoritmo faz
2. Criar o algoritmo em C++ (para CPU) que irá servir de golden rule para validar o output da Tenstorrent
3. Criar uma versão single-core
    1. Criar uma versão com loops
    2. Criar uma versão que resolve utilizando convolução
4. Criar uma versão multi-core

# Como executar

```bash
cmake -S . -B build_Debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build_Debug --target jacobi
```

O binário `jacobi` aceita os parâmetros:

```
jacobi <iterações> <input.bin> <output.bin>
```

- `iterações`: número de aplicações do método de Jacobi (padrão = 1)
- `input.bin`: caminho para a matriz quadrada já flatten em `float32`
- `output.bin`: arquivo de saída em `float32`

O script `golden.py` gera um `input.bin` sintético e chama `run_script.sh`.
O host agora usa caminhos relativos, então basta rodar o script no diretório do exemplo
para utilizar os arquivos padrão `input.bin`/`output.bin`.
