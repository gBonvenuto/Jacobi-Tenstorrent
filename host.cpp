#include <fmt/ostream.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>
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

/** **Funcionamento do Programa**
 *
 * É recebida uma matriz quadrada por meio do arquivo input.bin no formato float
 *
 * Essa matriz é transformada em bfloat16 e tiliziada
 *
 * Em seguida atribuímos um Tensix para ser responsável por cada tile. Esse Tensix
 * vai ler a sua tile na DRAM e vai enviá-la para seus vizinhos.
 *
 * E então começam as iterações, após cada iteração, os Tensix devem se comunicar
 * para que haja a transferência de tiles entre eles.
 *
 * Após as iterações, os Tensix enviam as tiles pelas quais eram responsáveis
 * de volta para a DRAM
 *
 * Espaço para melhorias:
 * - Há 8 registradores que podem ser utilizados, e estamos utilizando apenas 1.
 * - Há muita memória ainda que pode ser utilizada nos Tensix.
 *
 * Isso indica que podemos melhorar esse programa para ser capaz de processar
 * matrizes 8 vezes maiores, com cada Tensix sendo responsável por 8 tiles em
 * vez de 1
 *
 */

using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// Retorna uma matriz após ler um arquivo
// WARNING: assume que a matriz é quadrada
std::vector<float> read_matrix_from_file(const std::filesystem::path& filename) {
    std::ifstream file(filename, std::ios::binary);
    TT_FATAL(file.good(), "Não foi possível abrir {}", filename.string());
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> data(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    double element_count = data.size();
    double dimension = sqrt(element_count);
    fmt::print("A dimensão é {}\n", dimension);
    if (floor(dimension) != dimension) {
        TT_THROW("A Matriz não é quadrada ({} elementos)", element_count);
    }
    return data;
}

// Escreve uma matriz em um arquivo
void write_matrix_to_file(const std::filesystem::path& filename, const std::vector<float>& data) {
    std::ofstream file(filename, std::ios::binary);
    TT_FATAL(file.good(), "Não foi possível abrir {} para escrita", filename.string());
    size_t size = data.size() * sizeof(float);
    file.write(reinterpret_cast<const char*>(data.data()), size);
}

int main(int argc, char** argv) {
    uint32_t num_iterations = 1;
    if (argc > 1) {
        num_iterations = static_cast<uint32_t>(std::stoul(argv[1]));
    }
    const std::filesystem::path input_path = (argc > 2) ? argv[2] : std::filesystem::path("input.bin");
    const std::filesystem::path output_path = (argc > 3) ? argv[3] : std::filesystem::path("output.bin");
    // bool pass = true;
    try {
        // INFO: Buffers

        fmt::print("Iterações: {}\n", num_iterations);
        fmt::print("Lendo {}\n", input_path.string());
        std::vector<float> in_float = read_matrix_from_file(input_path);

        std::vector<bfloat16> in(in_float.size());

        fmt::print("input: float -> bfloat\n");

        for (size_t i = 0; i < in_float.size(); ++i) {
            in[i] = bfloat16(in_float[i]);
        }
        const double matrix_side_d = std::sqrt(static_cast<double>(in.size()));
        const uint32_t matrix_side = static_cast<uint32_t>(std::llround(matrix_side_d));

        TT_FATAL(
            matrix_side % tt::constants::TILE_WIDTH == 0,
            "A dimensão {} não é múltiplo de {} (tamanho da tile)",
            matrix_side,
            tt::constants::TILE_WIDTH);

        // Agora criamos também as matrizes de deslocamento L e U
        //
        // Se uma matriz A for multiplicada por uma dessas matrizes, ela será
        // deslocada.
        //
        // Ex: L✕A = A_{movido para baixo}

        fmt::print("Criando as matrizes auxiliares L, U, LL e UU\n");
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
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

        // INFO: Core Range
        const uint32_t width = matrix_side / tt::constants::TILE_WIDTH;
        const uint32_t height = width;

        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        constexpr tt::tt_metal::CoreCoord core_begin = {0, 0};
        const tt::tt_metal::CoreCoord core_end = {width - 1, height - 1};
        // const auto core0_physical_coord = mesh_device->worker_core_from_logical_core(core0);
        // const auto core1_physical_coord = mesh_device->worker_core_from_logical_core(core1);

        tt::tt_metal::CoreRange all_cores = tt::tt_metal::CoreRange(core_begin, core_end);

        // tt::tt_metal::CoreCoord start_phys = mesh_device->worker_core_from_logical_core(all_cores.start_coord);
        // tt::tt_metal::CoreCoord end_phys = mesh_device->worker_core_from_logical_core(all_cores.end_coord);

        distributed::MeshWorkload workload;
        auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());

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

        auto dram_buffer = distributed::MeshBuffer::create(dram_input_buffer_config, dram_config, mesh_device.get());

        auto L_dram_buffer = distributed::MeshBuffer::create(dram_LU_buffer_config, dram_config, mesh_device.get());
        auto U_dram_buffer = distributed::MeshBuffer::create(dram_LU_buffer_config, dram_config, mesh_device.get());
        auto LL_dram_buffer = distributed::MeshBuffer::create(dram_LU_buffer_config, dram_config, mesh_device.get());
        auto UU_dram_buffer = distributed::MeshBuffer::create(dram_LU_buffer_config, dram_config, mesh_device.get());

        Program program = CreateProgram();

        // INFO: Circular Buffers

        fmt::print("Criando os Circular Buffers\n");
        tt::CBIndex cb_in = tt::CBIndex::c_0;
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(tile_size_bytes, {{cb_in, tt::DataFormat::Float16_b}})
                .set_page_size(cb_in, tile_size_bytes));

        // L e U são as matrizes de deslocamento
        // de uma casa
        // Guarda primeiro L e depois U
        tt::CBIndex cb_LU = tt::CBIndex::c_1;
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(2 * tile_size_bytes, {{cb_LU, tt::DataFormat::Float16_b}})
                .set_page_size(cb_LU, tile_size_bytes));

        // LL UU são as matrizes de deslocamento de 31 casas
        // por exemplo, pega a coluna da esquerda e coloca na direita
        tt::CBIndex cb_LLUU = tt::CBIndex::c_2;
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(2 * tile_size_bytes, {{cb_LLUU, tt::DataFormat::Float16_b}})
                .set_page_size(cb_LLUU, tile_size_bytes));

        tt::CBIndex cb_vizinho_cima = tt::CBIndex::c_3;
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(tile_size_bytes, {{cb_vizinho_cima, tt::DataFormat::Float16_b}})
                .set_page_size(cb_vizinho_cima, tile_size_bytes));

        tt::CBIndex cb_vizinho_baixo = tt::CBIndex::c_4;
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(tile_size_bytes, {{cb_vizinho_baixo, tt::DataFormat::Float16_b}})
                .set_page_size(cb_vizinho_baixo, tile_size_bytes));

        tt::CBIndex cb_vizinho_esquerda = tt::CBIndex::c_5;
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(tile_size_bytes, {{cb_vizinho_esquerda, tt::DataFormat::Float16_b}})
                .set_page_size(cb_vizinho_esquerda, tile_size_bytes));

        tt::CBIndex cb_vizinho_direita = tt::CBIndex::c_6;
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(tile_size_bytes, {{cb_vizinho_direita, tt::DataFormat::Float16_b}})
                .set_page_size(cb_vizinho_direita, tile_size_bytes));

        tt::CBIndex cb_out = tt::CBIndex::c_16;
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(tile_size_bytes, {{cb_out, tt::DataFormat::Float16_b}})
                .set_page_size(cb_out, tile_size_bytes));

        // INFO: Semáforos
        fmt::print("Criando os semáforos\n");

        //
        // **semaphore_reader**
        //
        // -------------------
        //
        // Esse semáforo avisa os vizinhos de que o Tensix está pronto para
        // receber as suas tiles
        //
        const uint32_t semaphore_reader = CreateSemaphore(program, all_cores, 0);

        //
        // **semaphore_writer**
        //
        // -------------------
        //
        // Esse semáforo avisa os vizinhos de que o Tensix enviou as sua tile
        // para eles
        //
        const uint32_t semaphore_writer = CreateSemaphore(program, all_cores, 0);

        // INFO: Programando os Cores
        fmt::print(
            "Programando {} cores, {}, {}\n", all_cores.size(), all_cores.start_coord.str(), all_cores.end_coord.str());

        std::vector<uint32_t> reader_compile_time_args = {
            cb_in,
            cb_out,
            cb_LU,
            cb_LLUU,
            cb_vizinho_cima,
            cb_vizinho_baixo,
            cb_vizinho_esquerda,
            cb_vizinho_direita,
        };
        TensorAccessorArgs(*dram_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*L_dram_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*U_dram_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*LL_dram_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*UU_dram_buffer).append_to(reader_compile_time_args);
        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "Jacobi_Tenstorrent/kernels/worker/dataflow/read.cpp",
            all_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = reader_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args = {
            cb_out,
            cb_vizinho_cima,
            cb_vizinho_baixo,
            cb_vizinho_esquerda,
            cb_vizinho_direita,
        };
        TensorAccessorArgs(*dram_buffer).append_to(writer_compile_time_args);
        auto writer = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "Jacobi_Tenstorrent/kernels/worker/dataflow/write.cpp",
            all_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = writer_compile_time_args});

        std::vector<uint32_t> compute_compile_time_args = {
            cb_in,
            cb_out,
            cb_LU,
            cb_LLUU,
            cb_vizinho_cima,
            cb_vizinho_baixo,
            cb_vizinho_esquerda,
            cb_vizinho_direita,
        };
        auto compute = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "Jacobi_Tenstorrent/kernels/worker/compute/compute.cpp",
            all_cores,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .compile_args = compute_compile_time_args,
            });

        uint32_t in_addr = dram_buffer->address();
        uint32_t L_addr = L_dram_buffer->address();
        uint32_t U_addr = U_dram_buffer->address();
        uint32_t LL_addr = LL_dram_buffer->address();
        uint32_t UU_addr = UU_dram_buffer->address();

        for (const auto& core : corerange_to_cores(all_cores)) {
            const auto physical_core = mesh_device->worker_core_from_logical_core(core);
            const uint32_t logical_x = core.x - all_cores.start_coord.x;
            const uint32_t logical_y = core.y - all_cores.start_coord.y;
            const uint32_t tile_offset = logical_y * width + logical_x;

            fmt::print("Criando RuntimeArgs para {} (tile_offset={})\n", core.str(), tile_offset);

            SetRuntimeArgs(
                program,
                reader,
                core,
                {
                    in_addr,
                    tile_offset,
                    /* num_tiles =*/(uint32_t)1,
                    num_iterations,
                    L_addr,
                    U_addr,
                    LL_addr,
                    UU_addr,
                    semaphore_reader,
                    semaphore_writer,
                    logical_x,
                    logical_y,
                    width,
                    height,
                    static_cast<uint32_t>(physical_core.x),
                    static_cast<uint32_t>(physical_core.y),
                });

            SetRuntimeArgs(
                program,
                compute,
                core,
                {
                    /* num_tiles= */ 1,
                    num_iterations,
                    logical_x,
                    logical_y,
                    width,
                    height,
                });

            SetRuntimeArgs(
                program,
                writer,
                core,
                {
                    in_addr,
                    tile_offset,
                    /*num_tiles =*/1,
                    num_iterations,
                    semaphore_reader,
                    semaphore_writer,
                    logical_x,
                    logical_y,
                    width,
                    height,
                    static_cast<uint32_t>(physical_core.x),
                    static_cast<uint32_t>(physical_core.y),
                });
        }

        // E escrevemos todos eles na DRAM do device
        fmt::print("Escrevendo os buffers na DRAM\n");
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

        distributed::EnqueueWriteMeshBuffer(
            cq, dram_buffer, tilize_nfaces(in, tt::constants::TILE_HEIGHT * height, tt::constants::TILE_WIDTH * width));
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
        distributed::EnqueueMeshWorkload(cq, workload, true);

        // TODO: liberar memória no host enquanto esperamos o programa finalizar?

        distributed::Finish(cq);

        std::vector<bfloat16> result_vec_tilized;

        fmt::print("Lendo resultado do buffer\n");
        distributed::EnqueueReadMeshBuffer(cq, result_vec_tilized, dram_buffer, true);

        std::vector<bfloat16> result_vec =
            untilize_nfaces(result_vec_tilized, tt::constants::TILE_HEIGHT * height, tt::constants::TILE_WIDTH * width);

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

        fmt::print("Escrevendo em {}\n", output_path.string());
        write_matrix_to_file(output_path, result_vec_float);

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception! what: {}\n", e.what());
        throw;
    }
}
