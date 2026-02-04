#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api.h"

// **Runtime arguments**
//
// * n_tiles
// * n_iterations
// * my_x
// * my_y
// * width
// * height
//
// ** Compiletime arguments**
//
// * cb_in index
// * cb_out index
// * cb_LU index
// * cb_LLUU index
// * cb_top index
// * cb_bottom index
// * cb_left index
// * cb_right index
namespace NAMESPACE {
void MAIN {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    const uint32_t n_iterations = get_arg_val<uint32_t>(1);
    const uint32_t my_x = get_arg_val<uint32_t>(2);
    const uint32_t my_y = get_arg_val<uint32_t>(3);
    const uint32_t width = get_arg_val<uint32_t>(4);
    const uint32_t height = get_arg_val<uint32_t>(5);

    const bool has_left = (my_x > 0);
    const bool has_right = (my_x < width - 1);
    const bool has_top = (my_y > 0);
    const bool has_bottom = (my_y < height - 1);

    // Compiletime args
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    // L primeiro, U depois
    constexpr uint32_t cb_LU = get_compile_time_arg_val(2);
    // LL primeiro, UU depois
    constexpr uint32_t cb_LLUU = get_compile_time_arg_val(3);

    constexpr uint32_t cb_top = get_compile_time_arg_val(4);
    constexpr uint32_t cb_bottom = get_compile_time_arg_val(5);
    constexpr uint32_t cb_left = get_compile_time_arg_val(6);
    constexpr uint32_t cb_right = get_compile_time_arg_val(7);

    constexpr uint32_t dst_reg = 0;

    // Esperamos todas as tiles estarem disponíveis
    cb_wait_front(cb_LU, 2);
    cb_wait_front(cb_LLUU, 2);
    cb_wait_front(cb_in, 1);
    DPRINT_UNPACK(DPRINT << "Recebi as tiles auxiliares" << ENDL());

    for (uint32_t i = 0; i < n_iterations; i++) {
        DPRINT_UNPACK(DPRINT << "Esperando as vizinhas" << ENDL());
        cb_wait_front(cb_left, 1);   // NOTE: esperamos eles mesmo que não
        cb_wait_front(cb_right, 1);  // precisemos deles porque o reader ainda
        cb_wait_front(cb_top, 1);    // os envia vazios.
        cb_wait_front(cb_bottom, 1);
        tile_regs_acquire();

        // Limpar registrador antes de usar
        {
            fill_tile_init();
            fill_tile_int(dst_reg, 0);
        }

        // Utilizando a matriz L
        {
            if (i == 0) {
                mm_init(cb_in, cb_LU, cb_out);
            } else {
                mm_init_short(cb_in, cb_LU);
            }

            matmul_tiles(cb_in, cb_LU, 0, 0, dst_reg);  // Matriz esquerda
            // NOTE: quando é executada uma operação com
            // o mesmo registrador de destino, ele acumula,
            // não sobrescreve
            matmul_tiles(cb_LU, cb_in, 0, 0, dst_reg);  // Matriz abaixo
        }

        // Utilizando a matriz U
        {
            mm_init_short(cb_LU, cb_in);

            matmul_tiles(cb_in, cb_LU, 0, 1, dst_reg);  // Matriz direita
            matmul_tiles(cb_LU, cb_in, 1, 0, dst_reg);  // Matriz acima
        }

        // // filtrar o de cima e acumular no dst_reg
        // if (has_top) {
        //     mm_init_short(cb_LLUU, cb_top);
        //
        //     matmul_tiles(cb_LLUU, cb_top, 1, 0, dst_reg);
        // }

        // // filtrar o de baixo e acumular no dst_reg
        // if (has_bottom) {
        //     mm_init_short(cb_LLUU, cb_bottom);
        //
        //     matmul_tiles(cb_LLUU, cb_bottom, 0, 0, dst_reg);
        // }

        // // filtrar o da esquerda e acumular no dst_reg
        // if (has_left) {
        //     mm_init_short(cb_left, cb_LLUU);
        //
        //     matmul_tiles(cb_left, cb_LLUU, 0, 0, dst_reg);
        // }

        // // filtrar o da direita e acumular no dst_reg
        // if (has_right) {
        //     mm_init_short(cb_right, cb_LLUU);
        //
        //     matmul_tiles(cb_right, cb_LLUU, 0, 1, dst_reg);
        // }

        // // Dividir o resultado por 4
        // {
        //     DPRINT_MATH(DPRINT << "dividindo o acumulado por 4" << ENDL());
        //     binop_with_scalar_tile_init();
        //     // utilizei esse site para converter float para bfloat16
        //     // https://flop.evanau.dev/brainfloat-converter
        //     mul_unary_tile(dst_reg, 0x3e800000);  // multiplicar por 0.25
        // }

        cb_pop_front(cb_in, 1);
        cb_pop_front(cb_top, 1);
        cb_pop_front(cb_bottom, 1);
        cb_pop_front(cb_left, 1);
        cb_pop_front(cb_right, 1);

        tile_regs_commit();
        tile_regs_wait();

        // Então mandamos o resultado para o cb_out
        cb_reserve_back(cb_out, 1);
        cb_reserve_back(cb_in, 1);
        pack_tile(dst_reg, cb_out);
        pack_tile(dst_reg, cb_in);  // Volta para o cb_in
        cb_push_back(cb_out, 1);
        cb_push_back(cb_in, 1);

        tile_regs_release();
        DPRINT_PACK(DPRINT << "Iteração " << i << " finalizada" << ENDL());
    }
}
}  // namespace NAMESPACE
