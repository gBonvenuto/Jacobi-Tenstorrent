#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT

// **Runtime arguments**
//
// * out_addr
// * n_tiles
//
void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = tt::CBIndex::c_20;
    const uint32_t tile_size_bytes = get_tile_size(cb_out);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr, tile_size_bytes);

    // Tudo que precisamos fazer é esperar ter um dado no cb_out
    // e quando tiver enviamos para a DRAM

    cb_wait_front(cb_out, 1);
    DPRINT << "comecei a enviar o dado da L1 para a DRAM" << ENDL();

    noc_async_write_tile(0, out, get_read_ptr(cb_out));

    noc_async_write_barrier();

    cb_pop_front(cb_out, 1);
}
