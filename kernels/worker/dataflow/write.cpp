#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT

/**
 * Descrição do Writer:
 *
 * O Writer Kernel espera os outros Tensix incrementarem o seu
 * Sem_reader.
 *
 * Após seu próprio Sem_reader atingir um valor, ele irá enviar as
 * tiles para seus vizinhos e então incrementar o Sem_writer deles
 * avisando que tem tiles novas
 */

// **Runtime arguments**
//
// * in_addr
// * tile_offset
// * n_tiles
// * n_iterations
// * semaphore_reader
// * semaphore_writer
// * my_x
// * my_y
// * width
// * height
// * phys_x
// * phys_y
//
// **Compiletime arguments**
//
// * cb_in // TODO: remove
// * cb_out
// * cb_LU // TODO: remove
// * cb_LLUU // TODO: remove
// * cb_top
// * cb_bottom
// * cb_left
// * cb_right
// * inA TensorAccessorArgs
void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_offset = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(2);
    const uint32_t n_iterations = get_arg_val<uint32_t>(3);

    const auto semaphore_reader = get_semaphore(get_arg_val<uint32_t>(4));
    const auto semaphore_reader_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_reader);

    const auto semaphore_writer = get_semaphore(get_arg_val<uint32_t>(5));
    const auto semaphore_writer_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_writer);

    const uint32_t my_x = get_arg_val<uint32_t>(6);
    const uint32_t my_y = get_arg_val<uint32_t>(7);

    const uint32_t width = get_arg_val<uint32_t>(8);
    const uint32_t height = get_arg_val<uint32_t>(9);
    const uint32_t phys_x = get_arg_val<uint32_t>(10);
    const uint32_t phys_y = get_arg_val<uint32_t>(11);

    // Compiletime args
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_LU = get_compile_time_arg_val(2);
    constexpr uint32_t cb_LLUU = get_compile_time_arg_val(3);

    constexpr uint32_t cb_top = get_compile_time_arg_val(4);
    constexpr uint32_t cb_bottom = get_compile_time_arg_val(5);
    constexpr uint32_t cb_left = get_compile_time_arg_val(6);
    constexpr uint32_t cb_right = get_compile_time_arg_val(7);

    const uint32_t tile_size_bytes = get_tile_size(cb_out);

    constexpr auto in_args = TensorAccessorArgs<8>();
    const auto in = TensorAccessor(in_args, in_addr, tile_size_bytes);

    const bool has_left = (my_x > 0);
    const bool has_right = (my_x < width - 1);
    const bool has_top = (my_y > 0);
    const bool has_bottom = (my_y < height - 1);

    const int quantidade_vizinhos = has_left + has_right + has_top + has_bottom;

    const auto semaforo_cima_noc = (has_top) ? get_noc_addr(phys_x, phys_y - 1, semaphore_writer) : 0;
    const auto semaforo_baixo_noc = (has_bottom) ? get_noc_addr(phys_x, phys_y + 1, semaphore_writer) : 0;
    const auto semaforo_esquerda_noc = (has_left) ? get_noc_addr(phys_y - 1, phys_y, semaphore_writer) : 0;
    const auto semaforo_direita_noc = (has_right) ? get_noc_addr(phys_x + 1, phys_y, semaphore_writer) : 0;

    const auto cb_top_ptr = get_read_ptr(cb_top);
    const auto cb_bottom_ptr = get_read_ptr(cb_bottom);
    const auto cb_left_ptr = get_read_ptr(cb_left);
    const auto cb_right_ptr = get_read_ptr(cb_right);

    const uint64_t top_core_noc_addr = has_top ? get_noc_addr(phys_x, phys_y - 1, cb_bottom_ptr) : 0;
    const uint64_t bottom_core_noc_addr = has_bottom ? get_noc_addr(phys_x, phys_y + 1, cb_top_ptr) : 0;
    const uint64_t left_core_noc_addr = has_left ? get_noc_addr(phys_x - 1, phys_y, cb_right_ptr) : 0;
    const uint64_t right_core_noc_addr = has_right ? get_noc_addr(phys_x + 1, phys_y, cb_left_ptr) : 0;

    // WARNING: como temos que enviar a nossa tile para os vizinhos antes da primeira
    // iteração, talvez precisemos fazer n_iterations+1. Não acho que seja o caso, mas se houver
    // um hang na última iteração, talvez seja isso
    for (uint32_t i = 0; i < n_iterations; i++) {
        DPRINT << "Iniciando nova iteração" << ENDL();
        cb_wait_front(cb_out, 1);

        DPRINT << "Esperando os vizinhos estarem prontos para receber" << ENDL();
        noc_semaphore_wait(semaphore_reader_ptr, quantidade_vizinhos * (i + 1));

        DPRINT << "Enviando a tile para os vizinhos" << ENDL();

        if (has_top) {
            noc_async_write(cb_out, top_core_noc_addr, tile_size_bytes);
        }
        if (has_bottom) {
            noc_async_write(cb_out, bottom_core_noc_addr, tile_size_bytes);
        }
        if (has_left) {
            noc_async_write(cb_out, left_core_noc_addr, tile_size_bytes);
        }
        if (has_right) {
            noc_async_write(cb_out, right_core_noc_addr, tile_size_bytes);
        }

        noc_async_write_barrier();

        DPRINT << "Avisando que as tiles foram enviadas" << ENDL();

        if (has_top) {
            noc_semaphore_inc(semaforo_cima_noc, 1);
        }
        if (has_bottom) {
            noc_semaphore_inc(semaforo_baixo_noc, 1);
        }
        if (has_left) {
            noc_semaphore_inc(semaforo_esquerda_noc, 1);
        }
        if (has_right) {
            noc_semaphore_inc(semaforo_direita_noc, 1);
        }

        noc_async_full_barrier();  // BUG: talvez essa não seja a barreira correta.
                                   // Se o código não funcionar, tentar fazer um full_barrier

        DPRINT << "Finalizada uma iteração do writer" << ENDL();
    }

    // Envia o último valor de volta para a DRAM
    cb_wait_front(cb_out, 1);
    noc_async_write_tile(tile_offset, in, get_read_ptr(cb_out));
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
