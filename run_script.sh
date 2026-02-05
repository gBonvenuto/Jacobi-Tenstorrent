#!/bin/sh

export TT_METAL_HOME="$HOME/Documentos/Unicamp/GeoBench/Tenstorrent/tt-metal"
export TT_METAL_SIMULATOR="$HOME/Documentos/Unicamp/GeoBench/Tenstorrent/wormhole/libttsim_wh.so"
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"

# Possíveis valores para DPRINT_CORES
# export TT_METAL_DPRINT_CORES="all"
export TT_METAL_DPRINT_CORES="0,0"
# export TT_METAL_DPRINT_CORES="(0,0)-(1,1)"
# export TT_METAL_DPRINT_CORES="(0,0),(0,1)"

# Valid values are BR,NC,TR0,TR1,TR2,TR*,ER0,ER1,ER*.
# export TT_METAL_DPRINT_RISCVS=TR*

distrobox enter ubuntu -- /home/gian/Documentos/Unicamp/GeoBench/Tenstorrent/tt-metal/build_Debug/programming_examples/jacobi 20
