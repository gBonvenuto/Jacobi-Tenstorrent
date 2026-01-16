#include <cstdint>
#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

// **Runtime arguments**
//
// * in_addr
// * n_tiles
// * L_dram_addr
// * U_dram_addr
//
// **Compiletime arguments**
//
// * in TensorAccessorArgs
// * L TensorAccessorArgs
// * U TensorAccessorArgs
//
void kernel_main() {
    DPRINT << "inicializando Reader" << ENDL();
    uint32_t in_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
    uint32_t L_dram_addr = get_arg_val<uint32_t>(2);
    uint32_t U_dram_addr = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_L = tt::CBIndex::c_1;
    constexpr uint32_t cb_U = tt::CBIndex::c_2;

    const uint32_t tile_size_bytes = get_tile_size(cb_in);

    constexpr auto in_args = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(in_args, in_addr, tile_size_bytes);

    constexpr auto L_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto L_dram = TensorAccessor(L_args, L_dram_addr, tile_size_bytes);

    constexpr auto U_args = TensorAccessorArgs<L_args.next_compile_time_args_offset()>();
    const auto U_dram = TensorAccessor(U_args, U_dram_addr, tile_size_bytes);

    // Primeiro vamos copiar o L e o U para a memória na L1
    cb_reserve_back(cb_in, 1);
    cb_reserve_back(cb_L, 1);
    cb_reserve_back(cb_U, 1);
    noc_async_read_tile(0, in, get_write_ptr(cb_in));
    noc_async_read_tile(0, L_dram, get_write_ptr(cb_L));
    noc_async_read_tile(0, U_dram, get_write_ptr(cb_U));

    DPRINT << "terminei de passar tudo da dram para o L1" << ENDL();

    noc_async_read_barrier();  // Esperamos terminar de ler

    // Avisamos que tem valor novo
    cb_push_back(cb_in, 1);
    cb_push_back(cb_L, 1);
    cb_push_back(cb_U, 1);
    DPRINT << "Avisei que tem dado novo nos circular buffers" << ENDL();
}
