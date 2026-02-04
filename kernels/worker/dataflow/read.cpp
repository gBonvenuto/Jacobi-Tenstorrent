#include <cstdint>
#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

/**
 * Descrição do Reader:
 *
 * O Reader incrementa o Sem_reader dos vizinhos
 * avisando que está pronto para receber.
 *
 * Após isso, espera o seu próprio Sem_writer ser
 * incrementado para então dar push no circular buffer *
 *
 * Para evitar um hang inicial. Antes da primeira iteração
 * o Reader deve copiar o CB_in para o CB_out
 * para que o Writer possa escrevê-lo nos vizinhos
 */

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
// * semaphore_reader
// * semaphore_writer
// * my_x           // relativo ao start_coord
// * my_y           // relativo ao start_coord
// * width
// * height
//
// **Compiletime arguments**
//
// * cb_in index
// * cb_out index
// * cb_LU index
// * cb_LLUU index
// * cb_top index
// * cb_bottom index
// * cb_left index
// * cb_right index
// * in TensorAccessorArgs
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

    const auto semaphore_reader = get_semaphore(get_arg_val<uint32_t>(9));
    const auto semaphore_reader_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_reader);

    const auto semaphore_writer = get_semaphore(get_arg_val<uint32_t>(10));
    const auto semaphore_writer_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_writer);

    uint32_t my_x = get_arg_val<uint32_t>(11);
    const uint32_t my_y = get_arg_val<uint32_t>(12);
    const uint32_t width = get_arg_val<uint32_t>(13);
    const uint32_t height = get_arg_val<uint32_t>(14);

    const bool has_left = (my_x > 0);
    const bool has_right = (my_x < width - 1);
    const bool has_top = (my_y > 0);
    const bool has_bottom = (my_y < height - 1);

    const int quantidade_vizinhos = has_left + has_right + has_top + has_bottom;

    DPRINT << "my_x: " << my_x << ", my_y: " << my_y << ENDL();
    DPRINT << "width: " << width << ", height: " << height << ENDL();
    DPRINT << "has_left: " << (int)has_left << ", has_right: " << (int)has_right << ", has_top: " << (int)has_top
           << ", has_bottom: " << (int)has_bottom << ENDL();


    // Compiletime args
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_LU = get_compile_time_arg_val(2);
    constexpr uint32_t cb_LLUU = get_compile_time_arg_val(3);

    constexpr uint32_t cb_top = get_compile_time_arg_val(4);
    constexpr uint32_t cb_bottom = get_compile_time_arg_val(5);
    constexpr uint32_t cb_left = get_compile_time_arg_val(6);
    constexpr uint32_t cb_right = get_compile_time_arg_val(7);

    const uint32_t tile_size_bytes = get_tile_size(cb_in);

    constexpr auto in_args = TensorAccessorArgs<8>();
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

    // Lemos a nossa tile da DRAM
    cb_reserve_back(cb_in, 1);
    cb_reserve_back(cb_out, 1);

    // TODO:
    // para simplificar estou lendo a mesma informação duas vezes na DRAM.
    // Mas talvez seria melhor ler apenas uma única vez da DRAM no cb_in e copiar
    // localmente do CB_in para o CB_out.
    // Esse seria um experimento interessante para testar o overhead de leitura
    // da DRAM.
    noc_async_read_tile(tile_offset, in, get_write_ptr(cb_in));
    noc_async_read_tile(tile_offset, in, get_write_ptr(cb_out));

    cb_push_back(cb_in, 1);
    cb_push_back(cb_out, 1);  // Eviamos pro cb_out para podermos enviá-la para
                              // os vizinhos antes da primeira iteração.
                              // Nas iterações seguintes é o Compute quem
                              // atualiza o cb_out

    noc_async_read_barrier();

    const auto semaforo_cima_noc = (has_top) ? get_noc_addr(my_x, my_y - 1, semaphore_reader) : 0;
    const auto semaforo_baixo_noc = (has_bottom) ? get_noc_addr(my_x, my_y + 1, semaphore_reader) : 0;
    const auto semaforo_esquerda_noc = (has_left) ? get_noc_addr(my_x - 1, my_y, semaphore_reader) : 0;
    const auto semaforo_direita_noc = (has_right) ? get_noc_addr(my_x + 1, my_y, semaphore_reader) : 0;

    for (uint32_t i = 0; i < n_iterations; i++) {
        DPRINT << "começando uma iteração" << ENDL();

        // Preparamos espaço para os buffers
        cb_reserve_back(cb_left, 1);
        cb_reserve_back(cb_right, 1);
        cb_reserve_back(cb_top, 1);
        cb_reserve_back(cb_bottom, 1);

        // Avisamos que estamos prontos para receber
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

        noc_async_atomic_barrier(); // BUG: talvez essa não seja a barreira correta.
                                    // Se o código não funcionar, tentar fazer um full_barrier

        // Esperamos os outros enviarem as tiles deles para nós
        DPRINT << "Recebendo tiles vizinhas" << ENDL();
        noc_semaphore_wait(semaphore_writer_ptr, quantidade_vizinhos * (i + 1));

        // Agora que recebemos as tiles dos vizinhos, vamos dar um push_back
        // nos cbs
        cb_push_back(cb_left, 1);
        cb_push_back(cb_right, 1);
        cb_push_back(cb_top, 1);
        cb_push_back(cb_bottom, 1);
    }

    DPRINT << "todas as iterações foram finalizadas" << ENDL();
}
