#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api.h"

// **Runtime arguments**
//
// * n_tiles
//
namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_L = tt::CBIndex::c_1;
    constexpr auto cb_U = tt::CBIndex::c_2;
    // NOTE:Não vamos mais armazená-las. Vou acumular elas no packer
    //
    // constexpr auto cb_cima = tt::CBIndex::c_16;
    // constexpr auto cb_baixo = tt::CBIndex::c_17;
    // constexpr auto cb_esquerda = tt::CBIndex::c_18;
    // constexpr auto cb_direita = tt::CBIndex::c_19;
    constexpr auto cb_out = tt::CBIndex::c_20;

    constexpr uint32_t dst_reg = 0;

    // Esperamos todas as tiles estarem disponíveis
    cb_wait_front(cb_L, 1);
    cb_wait_front(cb_U, 1);
    cb_wait_front(cb_in, 1);
    DPRINT_UNPACK(DPRINT << "Recebi as tiles in, L e U" << ENDL());

    for (int i = 0; i < 100; i++) {
        tile_regs_acquire();

        // Utilizando a matriz L
        {
            mm_init(cb_in, cb_L, cb_in);

            // Matriz esquerda
            matmul_tiles(cb_in, cb_L, 0, 0, dst_reg);
            DPRINT_MATH(DPRINT << "Matriz deslocada para a esquerda pronta" << ENDL());
            // Matriz abaixo
            matmul_tiles(cb_L, cb_in, 0, 0, dst_reg);
            DPRINT_MATH(DPRINT << "Matriz deslocada para baixo pronta" << ENDL());
        }

        // Utilizando a matriz U
        {
            mm_init(cb_in, cb_U, cb_in);

            // Matriz direita
            matmul_tiles(cb_in, cb_U, 0, 0, dst_reg);
            DPRINT_MATH(DPRINT << "Matriz deslocada para a direita pronta" << ENDL());
            // Matriz acima
            matmul_tiles(cb_U, cb_in, 0, 0, dst_reg);
            DPRINT_MATH(DPRINT << "Matriz deslocada para cima pronta" << ENDL());
        }

        // Dividir o resultado por 4
        {
            DPRINT_MATH(DPRINT << "dividindo o acumulado por 4" << ENDL());
            binop_with_scalar_tile_init();
            // utilizei esse site para converter float para bfloat16
            // https://flop.evanau.dev/brainfloat-converter
            mul_unary_tile(dst_reg, 0x3e800000);  // multiplicar por 2
        }

        cb_pop_front(cb_in, 1);
        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_in, 1);
        pack_tile(dst_reg, cb_in);
        cb_push_back(cb_in, 1);

        tile_regs_release();
    }

    cb_wait_front(cb_in, 1);
    tile_regs_acquire();
    copy_tile_init(cb_in);
    copy_tile(cb_in, 0, dst_reg);
    tile_regs_commit();
    tile_regs_wait();

    // Então mandamos o resultado para o cb_out
    cb_reserve_back(cb_out, 1);
    pack_tile(dst_reg, cb_out);
    cb_push_back(cb_out, 1);

    tile_regs_release();
}
}  // namespace NAMESPACE
