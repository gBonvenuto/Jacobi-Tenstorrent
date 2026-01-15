#include "debug/dprint.h"  // required in all kernels using DPRINT
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
    constexpr auto cb_cima = tt::CBIndex::c_16;
    constexpr auto cb_baixo = tt::CBIndex::c_17;
    constexpr auto cb_esquerda = tt::CBIndex::c_18;
    constexpr auto cb_direita = tt::CBIndex::c_19;
    constexpr auto cb_out = tt::CBIndex::c_20;

    constexpr uint32_t dst_reg = 0;

    // Esperamos todas as tiles estarem disponíveis

    // NOTE: Vou começar fazendo apenas um left shift e ver esse valor
    mm_init(cb_in, cb_U, cb_out);

    DPRINT_MATH(DPRINT << "comecei a operação de shift up" << ENDL());
    tile_regs_acquire();
    DPRINT_MATH(DPRINT << "tile_regs_acquire concluído" << ENDL());
    cb_wait_front(cb_in, 1);
    cb_wait_front(cb_L, 1);
    cb_wait_front(cb_U, 1);
    DPRINT_MATH(DPRINT << "CB wait front concluído" << ENDL());

    matmul_tiles(cb_in, cb_U, 0, 0, 0);
    DPRINT_MATH(DPRINT << "matmul tiles concluído" << ENDL());
    cb_pop_front(cb_in, 1);
    cb_pop_front(cb_L, 1);
    cb_pop_front(cb_U, 1);
    DPRINT_MATH(DPRINT << "cb pop concluído" << ENDL());
    tile_regs_commit();
    DPRINT_MATH(DPRINT << "tile regs commit concluído" << ENDL());

    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    DPRINT_MATH(DPRINT << "tile regs wait concluído" << ENDL());

    // Aqui, se fosse uma operação comum,
    // faríamos pop. Mas não é necessário

    // Então mandamos o resultado para o cb_out
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);

    DPRINT_MATH(DPRINT << "terminei a operação de shift up e coloquei na L1" << ENDL();)

    tile_regs_release();
}
}  // namespace NAMESPACE
