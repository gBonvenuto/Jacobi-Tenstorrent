#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT

// **Runtime arguments**
//
// * in_addr
// * tile_offset
// * n_tiles
// * n_iterations
// * sem_computed
// * control x,
// * control y,
//
// **Compiletime arguments**
//
// * inA TensorAccessorArgs
void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_offset = get_arg_val<uint32_t>(2);
    const uint32_t n_tiles = get_arg_val<uint32_t>(3);
    const uint32_t n_iterations = get_arg_val<uint32_t>(4);
    const uint32_t sem_computed = get_semaphore(get_arg_val<uint32_t>(5));
    const uint32_t control_x = get_arg_val<uint32_t>(6);
    const uint32_t control_y = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out);

    constexpr auto in_args = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(in_args, in_addr, tile_size_bytes);

    // Tudo que precisamos fazer é esperar ter um dado no cb_out
    // e quando tiver enviamos para a DRAM

    const auto sem_computed_noc = get_noc_addr(control_x, control_y, sem_computed);

    for (uint32_t i = 0; i < n_iterations; i++) {
        cb_wait_front(cb_out, 1);
        DPRINT << "comecei a enviar o dado da L1 para a DRAM" << ENDL();

        noc_async_write_tile(tile_offset, in, get_read_ptr(cb_out));
        DPRINT << "escrevendo no inA" << ENDL();

        noc_async_write_barrier();

        cb_pop_front(cb_out, 1);
        noc_semaphore_inc(sem_computed_noc, 1);
    }
    // NOTE: o Host precisa fazer o cálculo de qual saída deve ler (inA ou inB)
}
