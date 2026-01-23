// NOTE: Esse código poderia ser utilizado em qualquer um dos cores (eu acho)
// mas vou colocar no reader

#include <cstdint>
#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

// **Runtime arguments**
//
// * num_workers
// * num_iteracoes
// * worker core x begin
// * worker core y begin
// * worker core x end
// * worker core y end
// * sem_start
// * sem_computed
//
void kernel_main() {
    DPRINT << "inicializando Reader" << ENDL();
    const uint32_t num_workers = get_arg_val<uint32_t>(0);
    const uint32_t num_iteracoes = get_arg_val<uint32_t>(1);
    const uint32_t worker_x_begin = get_arg_val<uint32_t>(2);
    const uint32_t worker_y_begin = get_arg_val<uint32_t>(3);
    const uint32_t worker_x_end = get_arg_val<uint32_t>(4);
    const uint32_t worker_y_end = get_arg_val<uint32_t>(5);

    const auto sem_start_id = get_semaphore(get_arg_val<uint32_t>(6));

    const auto sem_computed_id = get_semaphore(get_arg_val<uint32_t>(7));

    DPRINT << "Control: começando os trabalhos na range: {" << worker_x_begin << "," << worker_y_begin << "} -> {"
           << worker_x_end << "," << worker_y_end << "}" << ENDL();

    const auto sem_start_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_start_id);
    const auto sem_computed_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_computed_id);
    const auto sem_start_noc =
        get_noc_multicast_addr(worker_x_begin, worker_y_begin, worker_x_end, worker_y_end, sem_start_id);

    for (uint32_t i = 0; i < num_iteracoes; i++) {
        // Quando o computed se tornar num_workers, enviamos o sinal pra todos os
        // cores começarem a computar. Mas antes redefinimos o valor do semáforo
        DPRINT << "Control: esperando sem_computed atingir " << num_workers << ENDL();
        noc_semaphore_wait(sem_computed_ptr, num_workers * (i + 1));
        DPRINT << "Control: sem_computed atingiu " << num_workers << ENDL();

        noc_semaphore_set(sem_start_ptr, i + 1);

        DPRINT << "Control: enviando sinal de iniciar para todos os worker cores" << ENDL();
        noc_semaphore_set_multicast(sem_start_id, sem_start_noc, num_workers);
    }
    DPRINT << "Control: esperando última iteração finalizar" << ENDL();
    noc_semaphore_wait(sem_computed_ptr, num_workers * num_iteracoes);
}
