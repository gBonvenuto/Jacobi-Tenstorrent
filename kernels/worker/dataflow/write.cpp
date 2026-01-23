#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT

// **Runtime arguments**
//
// * inA_addr
// * inB_addr
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
// * inB TensorAccessorArgs
void kernel_main() {
    const uint32_t inA_addr = get_arg_val<uint32_t>(0);
    const uint32_t inB_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile_offset = get_arg_val<uint32_t>(2);
    const uint32_t n_tiles = get_arg_val<uint32_t>(3);
    const uint32_t n_iterations = get_arg_val<uint32_t>(4);
    const uint32_t sem_computed = get_semaphore(get_arg_val<uint32_t>(5));
    const uint32_t control_x = get_arg_val<uint32_t>(6);
    const uint32_t control_y = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out);

    constexpr auto inA_args = TensorAccessorArgs<0>();
    const auto inA = TensorAccessor(inA_args, inA_addr, tile_size_bytes);

    constexpr auto inB_args = TensorAccessorArgs<inA_args.next_compile_time_args_offset()>();
    const auto inB = TensorAccessor(inB_args, inB_addr, tile_size_bytes);

    // Tudo que precisamos fazer é esperar ter um dado no cb_out
    // e quando tiver enviamos para a DRAM

    const auto sem_computed_noc = get_noc_addr(control_x, control_y, sem_computed);

    // NOTE: A escrita é o oposto da leitura
    bool current_out = false;  // true: inA
                               // false: inB

    for (uint32_t i = 0; i < n_iterations; i++) {
        cb_wait_front(cb_out, 1);
        DPRINT << "comecei a enviar o dado da L1 para a DRAM" << ENDL();

        if (current_out) {
            noc_async_write_tile(tile_offset, inA, get_read_ptr(cb_out));
            DPRINT << "escrevendo no inA" << ENDL();
        } else {
            noc_async_write_tile(tile_offset, inB, get_read_ptr(cb_out));
            DPRINT << "escrevendo no inB" << ENDL();
        }
        current_out = !current_out;

        noc_async_write_barrier();

        cb_pop_front(cb_out, 1);
        noc_semaphore_inc(sem_computed_noc, 1);
    }
    // NOTE: o Host precisa fazer o cálculo de qual saída deve ler (inA ou inB)
}
