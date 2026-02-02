Vou tentar fazer um algoritmo de Stencil na placa da Tenstorrent Blackhole.

No momento não tenho acesso à placa física, então estarei utilizando o 
simulador [ttsim](https://github.com/tenstorrent/ttsim).

O Stencil de jacobi é o seguinte:

$$
p_{new}[i,j] = \frac{1}{4} (p_{old}[i-1,j]+p_{old}[i+1,j]+p_{old}[i,j-1]+p_{old}[i,j+1)
$$

Isto é, a cada iteração, os valores de cada célula se torna a média de seus vizinhos de cima, baixo, esquerda e direita

# Objetivos

1. Criar o algoritmo em Python para validar se entendi corretamente o que este algoritmo faz (✅)
2. Criar uma versão single-core
    1. Criar uma versão com loops (✅)
    2. Criar uma versão que resolve utilizando convolução (❌)
3. Criar uma versão multi-core
    1. Criar uma versão naive em que todos acessam a DRAM (✅)
    2. Criar uma versão com acesso a memória otimizado (❓)
