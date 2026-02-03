#include <cstdint>
#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

// **Runtime arguments**
//
// * in_addr
// * tile_offset
// * n_tiles
// * n_iterations
// * L_dram_addr
// * U_dram_addr
// * LL_dram_addr
// * UU_dram_addr
// * sem_start
// * my_x           // relativo ao start_coord
// * my_y           // relativo ao start_coord
// * width
// * height
//
// **Compiletime arguments**
//
// * inA TensorAccessorArgs
// * inB TensorAccessorArgs
// * L TensorAccessorArgs
// * U TensorAccessorArgs
//
void kernel_main() {
    DPRINT << "inicializando Reader" << ENDL();
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_offset = get_arg_val<uint32_t>(2);
    // TODO: estamos sempre assumindo que será uma tile por core.
    // Se isso mudar, precisamos mudar o código
    const uint32_t n_tiles = get_arg_val<uint32_t>(3);
    const uint32_t n_iterations = get_arg_val<uint32_t>(4);
    const uint32_t L_dram_addr = get_arg_val<uint32_t>(5);
    const uint32_t U_dram_addr = get_arg_val<uint32_t>(6);
    const uint32_t LL_dram_addr = get_arg_val<uint32_t>(7);
    const uint32_t UU_dram_addr = get_arg_val<uint32_t>(8);

    const auto sem_start_id = get_semaphore(get_arg_val<uint32_t>(9));
    const auto sem_start_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_start_id);

    uint32_t my_x = get_arg_val<uint32_t>(10);
    const uint32_t my_y = get_arg_val<uint32_t>(11);
    const uint32_t width = get_arg_val<uint32_t>(12);
    const uint32_t height = get_arg_val<uint32_t>(13);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_LU = tt::CBIndex::c_1;
    constexpr uint32_t cb_LLUU = tt::CBIndex::c_2;

    constexpr uint32_t cb_top = tt::CBIndex::c_3;
    constexpr uint32_t cb_bottom = tt::CBIndex::c_4;
    constexpr uint32_t cb_left = tt::CBIndex::c_5;
    constexpr uint32_t cb_right = tt::CBIndex::c_6;

    const bool has_left = (my_x > 0);
    const bool has_right = (my_x < width - 1);
    const bool has_top = (my_y > 0);
    const bool has_bottom = (my_y < height - 1);

    DPRINT << "my_x: " << my_x << ", my_y: " << my_y << ENDL();
    DPRINT << "width: " << width << ", height: " << height << ENDL();
    DPRINT << "has_left: " << (int)has_left << ", has_right: " << (int)has_right << ", has_top: " << (int)has_top
           << ", has_bottom: " << (int)has_bottom << ENDL();

    const uint32_t tile_size_bytes = get_tile_size(cb_in);

    // NOTE: Precisamos de dois vetores de entrada para não correr o risco
    // de sobrescrever
    constexpr auto in_args = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(in_args, in_addr, tile_size_bytes);

    constexpr auto L_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto L_dram = TensorAccessor(L_args, L_dram_addr, tile_size_bytes);

    constexpr auto U_args = TensorAccessorArgs<L_args.next_compile_time_args_offset()>();
    const auto U_dram = TensorAccessor(U_args, U_dram_addr, tile_size_bytes);

    constexpr auto LL_args = TensorAccessorArgs<U_args.next_compile_time_args_offset()>();
    const auto LL_dram = TensorAccessor(LL_args, LL_dram_addr, tile_size_bytes);

    constexpr auto UU_args = TensorAccessorArgs<LL_args.next_compile_time_args_offset()>();
    const auto UU_dram = TensorAccessor(UU_args, UU_dram_addr, tile_size_bytes);

    // Primeiro vamos copiar o L e o U para a memória na L1
    DPRINT << "lendo as matrizes L e LL" << ENDL();
    cb_reserve_back(cb_LU, 1);
    cb_reserve_back(cb_LLUU, 1);
    noc_async_read_tile(0, L_dram, get_write_ptr(cb_LU));
    noc_async_read_tile(0, LL_dram, get_write_ptr(cb_LLUU));
    cb_push_back(cb_LU, 1);
    cb_push_back(cb_LLUU, 1);
    noc_async_read_barrier();

    DPRINT << "lendo as matrizes U e UU" << ENDL();
    cb_reserve_back(cb_LU, 1);
    cb_reserve_back(cb_LLUU, 1);
    noc_async_read_tile(0, U_dram, get_write_ptr(cb_LU));
    noc_async_read_tile(0, UU_dram, get_write_ptr(cb_LLUU));
    cb_push_back(cb_LU, 1);
    cb_push_back(cb_LLUU, 1);
    noc_async_read_barrier();

    DPRINT << "começando uma iteração" << ENDL();

    const uint32_t left_offset = tile_offset - 1;
    const uint32_t right_offset = tile_offset + 1;
    const uint32_t top_offset = tile_offset - width;
    const uint32_t bottom_offset = tile_offset + width;

    for (uint32_t i = 0; i < n_iterations; i++) {
        DPRINT << "Esperando sem_start" << ENDL();
        noc_semaphore_wait(sem_start_ptr, i + 1);  // esperamos o start virar 1
        DPRINT << "start!!" << ENDL();

        cb_reserve_back(cb_in, 1);
        cb_reserve_back(cb_left, 1);
        cb_reserve_back(cb_right, 1);
        cb_reserve_back(cb_top, 1);
        cb_reserve_back(cb_bottom, 1);

        noc_async_read_tile(tile_offset, in, get_write_ptr(cb_in));
        if (has_left) {
            noc_async_read_tile(left_offset, in, get_write_ptr(cb_left));
        }
        if (has_right) {
            noc_async_read_tile(right_offset, in, get_write_ptr(cb_right));
        }
        if (has_top) {
            noc_async_read_tile(top_offset, in, get_write_ptr(cb_top));
        }
        if (has_bottom) {
            noc_async_read_tile(bottom_offset, in, get_write_ptr(cb_bottom));
        }
        DPRINT << "lendo do in" << ENDL();

        noc_async_read_barrier();  // Esperamos terminar de ler

        cb_push_back(cb_in, 1);  // Avisamos que tem valor novo
        cb_push_back(cb_left, 1);
        cb_push_back(cb_right, 1);
        cb_push_back(cb_top, 1);
        cb_push_back(cb_bottom, 1);

        // NOTE: e incrementaremos o computed no writer quando ele terminar de
        // escrever na DRAM
    }

    DPRINT << "todas as iterações foram finalizadas" << ENDL();
}
