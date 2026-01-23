#include <fmt/ostream.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

// Esse programa tem como objetivo explorar a criação de um programa
// para o Tenstorrent por minha própria conta.
// Pretendo desenvolver as seguintes habilidades
// 1. Criar um Kernel do zero
// 2. Planejar quais buffers vou precisar
// 3. Planejar os circular buffers que vou utilizar
// 4. Planejar como será a comunicação multi-core
//
// Outras coisas que vou aprender também é a utilizar melhor as
// Matrix e Vector engines, coordenar os kernels, etc.
//
// Também estive evitando utilizar as melhores práticas de
// programação (como o try-catch), então vou me esforçar para
// utilizá-los quando eu encontrar a necessidade

using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// Retorna uma matriz após ler um arquivo
std::vector<float> read_matrix_from_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> data(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

// Escreve uma matriz em um arquivo
void write_matrix_to_file(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename, std::ios::binary);
    size_t size = data.size() * sizeof(float);
    file.write(reinterpret_cast<const char*>(data.data()), size);
}

int main() {
    // bool pass = true;
    try {
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

        // INFO: Core Range
        constexpr CoreCoord core_controle = {0, 0};
        constexpr CoreCoord core0 = {0, 1};
        constexpr CoreCoord core1 = {1, 1};
        // const auto core0_physical_coord = mesh_device->worker_core_from_logical_core(core0);
        // const auto core1_physical_coord = mesh_device->worker_core_from_logical_core(core1);

        tt::tt_metal::CoreRange worker_cores = tt::tt_metal::CoreRange(core0, core1);
        tt::tt_metal::CoreRangeSet all_cores =
            tt::tt_metal::CoreRangeSet(std::set{worker_cores, CoreRange(core_controle)});

        tt::tt_metal::CoreCoord control_phys = mesh_device->worker_core_from_logical_core(core_controle);
        tt::tt_metal::CoreCoord start_phys = mesh_device->worker_core_from_logical_core(worker_cores.start_coord);
        tt::tt_metal::CoreCoord end_phys = mesh_device->worker_core_from_logical_core(worker_cores.end_coord);

        distributed::MeshWorkload workload;
        auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());

        // INFO: Buffers

        fmt::print("lendo input.bin\n");
        std::vector<float> in_float = read_matrix_from_file(
            "/home/gian/Documentos/Unicamp/GeoBench/Tenstorrent/tt-metal/tt_metal/programming_examples/"
            "Jacobi_Tenstorrent/input.bin");

        std::vector<bfloat16> in(in_float.size());

        fmt::print("input: float -> bfloat\n");
        for (size_t i = 0; i < in_float.size(); ++i) {
            in[i] = bfloat16(in_float[i]);
        }

        const uint32_t width = worker_cores.end_coord.x - worker_cores.start_coord.x + 1;
        const uint32_t height = worker_cores.end_coord.y - worker_cores.start_coord.y + 1;
        uint32_t num_iterations;
        fmt::print("Quantas iterações: ");
        scanf("%d", &num_iterations);
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        const uint32_t num_tiles = in.size() / elements_per_tile;
        constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;
        const uint32_t dram_input_buffer_size = tile_size_bytes * num_tiles;
        constexpr uint32_t dram_LU_buffer_size = tile_size_bytes;
        fmt::print(
            "num_tiles {}, width: {}, height: {}, num_iterations: {}\n", num_tiles, width, height, num_iterations);

        // Vou precisar de um buffer na DRAM para:
        // - bufferA
        // - bufferB
        // - Shift Matrix L
        // - Shift Matrix U
        // - Shift Matrix LL
        // - Shift Matrix UU
        //
        // Não vou utilizar um buffer na L1, vou utilizar circular buffers

        fmt::print("criando os buffers na DRAM\n");
        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = tile_size_bytes,
            .buffer_type = BufferType::DRAM,
        };

        distributed::ReplicatedBufferConfig dram_input_buffer_config{
            .size = dram_input_buffer_size,
        };

        distributed::ReplicatedBufferConfig dram_LU_buffer_config{
            .size = dram_LU_buffer_size,
        };

        auto dram_buffer_A = distributed::MeshBuffer::create(dram_input_buffer_config, dram_config, mesh_device.get());
        auto dram_buffer_B = distributed::MeshBuffer::create(dram_input_buffer_config, dram_config, mesh_device.get());

        auto L_dram_buffer = distributed::MeshBuffer::create(dram_LU_buffer_config, dram_config, mesh_device.get());
        auto U_dram_buffer = distributed::MeshBuffer::create(dram_LU_buffer_config, dram_config, mesh_device.get());
        auto LL_dram_buffer = distributed::MeshBuffer::create(dram_LU_buffer_config, dram_config, mesh_device.get());
        auto UU_dram_buffer = distributed::MeshBuffer::create(dram_LU_buffer_config, dram_config, mesh_device.get());

        // Agora criamos também as matrizes de deslocamento L e U
        //
        // Se uma matriz A for multiplicada por uma dessas matrizes, ela será
        // deslocada.
        //
        // Ex: L✕A = A_{movido para baixo}

        fmt::print("Criando as matrizes auxiliares L, U, LL e UU\n");
        std::vector<bfloat16> L(elements_per_tile, 0);
        std::vector<bfloat16> U(elements_per_tile, 0);
        std::vector<bfloat16> LL(elements_per_tile, 0);
        std::vector<bfloat16> UU(elements_per_tile, 0);

        // o L precisa ser preenchido com uns em baixo da diagonal
        // e o U precisa ser preenchido com uns em cima da diagonal
        //
        // WARNING: não sei se isso tá correto
        // Para acessar um valor [i,j] fazemos i*TILE_WIDTH+j
        for (int idx = 0; idx < tt::constants::TILE_WIDTH - 1; idx++) {
            {
                int i = idx + 1;
                int j = idx;
                L[i * tt::constants::TILE_WIDTH + j] = bfloat16(1.0f);
            }
            {
                int i = idx;
                int j = idx + 1;
                U[i * tt::constants::TILE_WIDTH + j] = bfloat16(1.0f);
            }
        }

        // Para preencher o LL UU, precisamos apenas colocar um único
        // um em uma das extremidades

        // LL: Um único 1 na posição i = 31 e j=0
        LL[31 * tt::constants::TILE_WIDTH + 0] = 1;

        // UU: Um único 1 na posição i = 0 e j=31
        UU[31] = 1;

        Program program = CreateProgram();

        // INFO: Circular Buffers

        fmt::print("Criando os Circular Buffers\n");
        tt::CBIndex cb_in = tt::CBIndex::c_0;
        CreateCircularBuffer(
            program,
            worker_cores,
            CircularBufferConfig(2 * tile_size_bytes, {{cb_in, tt::DataFormat::Float16_b}})
                .set_page_size(cb_in, tile_size_bytes));

        // L e U são as matrizes de deslocamento
        // de uma casa
        // Guarda primeiro L e depois U
        tt::CBIndex cb_LU = tt::CBIndex::c_1;
        CreateCircularBuffer(
            program,
            worker_cores,
            CircularBufferConfig(2 * tile_size_bytes, {{cb_LU, tt::DataFormat::Float16_b}})
                .set_page_size(cb_LU, tile_size_bytes));

        // LL UU são as matrizes de deslocamento de 31 casas
        // por exemplo, pega a coluna da esquerda e coloca na direita
        tt::CBIndex cb_LLUU = tt::CBIndex::c_2;
        CreateCircularBuffer(
            program,
            worker_cores,
            CircularBufferConfig(2 * tile_size_bytes, {{cb_LLUU, tt::DataFormat::Float16_b}})
                .set_page_size(cb_LLUU, tile_size_bytes));

        tt::CBIndex cb_vizinho_cima = tt::CBIndex::c_3;
        CreateCircularBuffer(
            program,
            worker_cores,
            CircularBufferConfig(2 * tile_size_bytes, {{cb_vizinho_cima, tt::DataFormat::Float16_b}})
                .set_page_size(cb_vizinho_cima, tile_size_bytes));

        tt::CBIndex cb_vizinho_baixo = tt::CBIndex::c_4;
        CreateCircularBuffer(
            program,
            worker_cores,
            CircularBufferConfig(2 * tile_size_bytes, {{cb_vizinho_baixo, tt::DataFormat::Float16_b}})
                .set_page_size(cb_vizinho_baixo, tile_size_bytes));

        tt::CBIndex cb_vizinho_esquerda = tt::CBIndex::c_5;
        CreateCircularBuffer(
            program,
            worker_cores,
            CircularBufferConfig(2 * tile_size_bytes, {{cb_vizinho_esquerda, tt::DataFormat::Float16_b}})
                .set_page_size(cb_vizinho_esquerda, tile_size_bytes));

        tt::CBIndex cb_vizinho_direita = tt::CBIndex::c_6;
        CreateCircularBuffer(
            program,
            worker_cores,
            CircularBufferConfig(2 * tile_size_bytes, {{cb_vizinho_direita, tt::DataFormat::Float16_b}})
                .set_page_size(cb_vizinho_direita, tile_size_bytes));

        tt::CBIndex cb_out = tt::CBIndex::c_16;
        CreateCircularBuffer(
            program,
            worker_cores,
            CircularBufferConfig(2 * tile_size_bytes, {{cb_out, tt::DataFormat::Float16_b}})
                .set_page_size(cb_out, tile_size_bytes));

        tt::CBIndex cb_tmp = tt::CBIndex::c_17;
        CreateCircularBuffer(
            program,
            worker_cores,
            CircularBufferConfig(1 * tile_size_bytes, {{cb_tmp, tt::DataFormat::Float16_b}})
                .set_page_size(cb_tmp, tile_size_bytes));

        // INFO: Semáforos
        fmt::print("Criando os semáforos\n");

        //
        // **sem_id_computed**
        //
        // -------------------
        //
        // Ao ler, o Reader incrementa o valor do semáforo
        // E o core de controle é resposável por monitorar esse semáforo,
        // e ao atingir um determinado valor, o core de controle irá
        // incrementar o semáforo de start de cada tensix
        //
        // começamos com o valor num_workers para que o programa já comece
        const uint32_t sem_id_computed = CreateSemaphore(program, core_controle, worker_cores.size());

        // **sem_id_start**
        //
        // -------------------
        //
        // O Tensix de controle irá incrementar o semáforo de cada core
        // para avisar que o trabalho deve ser feito.
        // E o Tensix de trabalho irá setar a zero quando começar o trabalho
        const uint32_t sem_id_start = CreateSemaphore(program, all_cores, 0);

        // INFO: programando o core_controle

        fmt::print("Criando o kernel de controle\n");
        auto control = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "Jacobi_Tenstorrent/kernels/controle/control.cpp",
            core_controle,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
            });

        fmt::print(
            "Estamos lidando com {} woker cores, {}, {}\n",
            worker_cores.size(),
            worker_cores.start_coord.str(),
            worker_cores.end_coord.str());
        SetRuntimeArgs(
            program,
            control,
            core_controle,
            {
                (uint32_t)worker_cores.size(),
                num_iterations,
                (uint32_t)start_phys.x,
                (uint32_t)start_phys.y,
                (uint32_t)end_phys.x,
                (uint32_t)end_phys.y,
                sem_id_start,
                sem_id_computed,
            });

        // INFO: Programando os worker cores
        fmt::print("Criando o kernel dos worker cores\n");
        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(*dram_buffer_A).append_to(reader_compile_time_args);
        TensorAccessorArgs(*dram_buffer_B).append_to(reader_compile_time_args);
        TensorAccessorArgs(*L_dram_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*U_dram_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*LL_dram_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*UU_dram_buffer).append_to(reader_compile_time_args);
        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "Jacobi_Tenstorrent/kernels/worker/dataflow/read.cpp",
            worker_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = reader_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(*dram_buffer_A).append_to(writer_compile_time_args);
        TensorAccessorArgs(*dram_buffer_B).append_to(writer_compile_time_args);
        auto writer = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "Jacobi_Tenstorrent/kernels/worker/dataflow/write.cpp",
            worker_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = writer_compile_time_args});

        auto compute = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "Jacobi_Tenstorrent/kernels/worker/compute/compute.cpp",
            worker_cores,
            ComputeConfig{
                .math_fidelity =
                    MathFidelity::HiFi4});  // There's different math fidelity modes (for the tensor engine)

        uint32_t inA_addr = dram_buffer_A->address();
        uint32_t inB_addr = dram_buffer_B->address();
        uint32_t L_addr = L_dram_buffer->address();
        uint32_t U_addr = U_dram_buffer->address();
        uint32_t LL_addr = LL_dram_buffer->address();
        uint32_t UU_addr = UU_dram_buffer->address();

        uint32_t offset = 0;
        for (auto& core : corerange_to_cores(worker_cores)) {
            fmt::print("Criando RuntimeArgs para {}\n", core.str());
            SetRuntimeArgs(
                program,
                reader,
                core,
                {
                    inA_addr,
                    inB_addr,
                    offset,
                    /* num_tiles =*/(uint32_t)1,
                    num_iterations,
                    L_addr,
                    U_addr,
                    LL_addr,
                    UU_addr,
                    sem_id_start,
                    core.x - worker_cores.start_coord.x,
                    core.y - worker_cores.start_coord.y,
                    width,
                    height,
                });

            SetRuntimeArgs(
                program,
                compute,
                core,
                {
                    /* num_tiles= */ 1,
                    num_iterations,
                    core.x - worker_cores.start_coord.x,
                    core.y - worker_cores.start_coord.y,
                    width,
                    height,
                });

            SetRuntimeArgs(
                program,
                writer,
                core,
                {
                    inA_addr,
                    inB_addr,
                    offset,
                    /*num_tiles =*/1,
                    num_iterations,
                    sem_id_computed,
                    control_phys.x,
                    control_phys.y,
                });

            offset += 1;  // WARNING: assumindo uma tile por core
        }

        // E escrevemos todos eles na DRAM do device
        fmt::print("Escrevendo os buffers na DRAM\n");
        distributed::EnqueueWriteMeshBuffer(
            cq, dram_buffer_A, tilize_nfaces(in, tt::constants::TILE_HEIGHT*height, tt::constants::TILE_WIDTH*width));
        distributed::EnqueueWriteMeshBuffer(
            cq, L_dram_buffer, tilize_nfaces(L, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH));
        distributed::EnqueueWriteMeshBuffer(
            cq, U_dram_buffer, tilize_nfaces(U, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH));
        distributed::EnqueueWriteMeshBuffer(
            cq, LL_dram_buffer, tilize_nfaces(LL, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH));
        distributed::EnqueueWriteMeshBuffer(
            cq, UU_dram_buffer, tilize_nfaces(UU, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH));

        fmt::print("Enviando programa para o device\n");
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);

        std::vector<bfloat16> result_vec_tilized;

        // INFO: Se há um número par de iterações, então o resultado está no inA
        // mas se há um número ímpar de iterações, então o resultado está no inB
        if (num_iterations % 2 == 0) {
            fmt::print("Lendo resultado do bufer_A\n");
            distributed::EnqueueReadMeshBuffer(cq, result_vec_tilized, dram_buffer_A, true);
        } else {
            fmt::print("Lendo resultado do bufer_B\n");
            distributed::EnqueueReadMeshBuffer(cq, result_vec_tilized, dram_buffer_B, true);
        }

        std::vector<bfloat16> result_vec =
            untilize_nfaces(result_vec_tilized, tt::constants::TILE_HEIGHT*height, tt::constants::TILE_WIDTH*width);

        mesh_device->close();

        std::vector<float> result_vec_float(result_vec.size());

        fmt::print("output bfloat -> float\n");
        for (size_t i = 0; i < result_vec.size(); ++i) {
            result_vec_float[i] = (float)result_vec[i];
        }

        // testando se criei as matrizes auxiliares corretamente
        // for (int i = 0; i < height*32; ++i) {
        //     for (int j = 0; j < 32; ++j) {
        //         // Mapeia a coordenada (i, j) para os índices lineares de cada vetor
        //         result_vec_float[i * (width*32) + j] = (float)result_vec[i * 32 + j];
        //     }
        // }

        fmt::print("Escrevendo em output.bin\n");
        write_matrix_to_file(
            "/home/gian/Documentos/Unicamp/GeoBench/Tenstorrent/tt-metal/tt_metal/programming_examples/"
            "Jacobi_Tenstorrent/output.bin",
            result_vec_float);
    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception! what: {}\n", e.what());
        throw;
    }
}
