#include <fmt/ostream.h>
#include <cstdint>
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
        constexpr tt::tt_metal::CoreCoord core = {0, 0};
        distributed::MeshWorkload workload;
        auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());

        // NOTE: vamos começar com apenas uma tile e um core
        constexpr uint32_t num_tiles = 1;
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;
        constexpr uint32_t dram_input_buffer_size = tile_size_bytes * num_tiles;
        constexpr uint32_t dram_LU_buffer_size = tile_size_bytes;

        // Vou precisar de um buffer na DRAM para:
        // - Input
        // - Output
        // - Shift Matrix L
        // - Shift Matrix U
        //
        // Não vou utilizar um buffer na L1, vou utilizar circular buffers

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

        auto input_dram_buffer =
            distributed::MeshBuffer::create(dram_input_buffer_config, dram_config, mesh_device.get());
        auto output_dram_buffer =
            distributed::MeshBuffer::create(dram_input_buffer_config, dram_config, mesh_device.get());

        auto L_dram_buffer = distributed::MeshBuffer::create(dram_LU_buffer_config, dram_config, mesh_device.get());
        auto U_dram_buffer = distributed::MeshBuffer::create(dram_LU_buffer_config, dram_config, mesh_device.get());

        std::vector<float> in_float = read_matrix_from_file(
            "/home/gian/Documentos/Unicamp/GeoBench/Tenstorrent/tt-metal/tt_metal/programming_examples/"
            "Jacobi_Tenstorrent/input.bin");

        std::vector<bfloat16> in(in_float.size());

        for (size_t i = 0; i < in_float.size(); ++i) {
            in[i] = bfloat16(in_float[i]);
        }

        // NOTE: Printei tanto no Python quanto aqui para testar se eles têm o mesmo valor
        fmt::print("\n\n{0}\n\n", (float)in[in.size() - 1]);

        // Agora criamos também as matrizes de deslocamento L e U
        //
        // Se uma matriz A for multiplicada por uma dessas matrizes, ela será
        // deslocada.
        //
        // Ex: L✕A = A_{movido para baixo}

        std::vector<bfloat16> L(elements_per_tile, 0);
        std::vector<bfloat16> U(elements_per_tile, 0);

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

        Program program = CreateProgram();

        // Agora criamos também os CBs
        // NOTE: Mesmo que tenha somente uma única tile por enquanto
        // já estou preparando os circular buffers para quando tiverem mais
        constexpr uint32_t tiles_per_cb = 2;

        tt::CBIndex cb_in = tt::CBIndex::c_0;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(tiles_per_cb * tile_size_bytes, {{cb_in, tt::DataFormat::Float16_b}})
                .set_page_size(cb_in, tile_size_bytes));

        tt::CBIndex cb_L = tt::CBIndex::c_1;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(tiles_per_cb * tile_size_bytes, {{cb_L, tt::DataFormat::Float16_b}})
                .set_page_size(cb_L, tile_size_bytes));

        tt::CBIndex cb_U = tt::CBIndex::c_2;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(tiles_per_cb * tile_size_bytes, {{cb_U, tt::DataFormat::Float16_b}})
                .set_page_size(cb_U, tile_size_bytes));

        tt::CBIndex cb_cima = tt::CBIndex::c_16;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(tiles_per_cb * tile_size_bytes, {{cb_cima, tt::DataFormat::Float16_b}})
                .set_page_size(cb_cima, tile_size_bytes));

        tt::CBIndex cb_baixo = tt::CBIndex::c_17;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(tiles_per_cb * tile_size_bytes, {{cb_baixo, tt::DataFormat::Float16_b}})
                .set_page_size(cb_baixo, tile_size_bytes));

        tt::CBIndex cb_esquerda = tt::CBIndex::c_18;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(tiles_per_cb * tile_size_bytes, {{cb_esquerda, tt::DataFormat::Float16_b}})
                .set_page_size(cb_esquerda, tile_size_bytes));

        tt::CBIndex cb_direita = tt::CBIndex::c_19;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(tiles_per_cb * tile_size_bytes, {{cb_direita, tt::DataFormat::Float16_b}})
                .set_page_size(cb_direita, tile_size_bytes));

        tt::CBIndex cb_out = tt::CBIndex::c_20;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(tiles_per_cb * tile_size_bytes, {{cb_out, tt::DataFormat::Float16_b}})
                .set_page_size(cb_out, tile_size_bytes));

        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(*input_dram_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*L_dram_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*U_dram_buffer).append_to(reader_compile_time_args);
        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "Jacobi_Tenstorrent/kernels/dataflow/read.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = reader_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(*output_dram_buffer).append_to(writer_compile_time_args);
        auto writer = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "Jacobi_Tenstorrent/kernels/dataflow/write.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = writer_compile_time_args});

        auto compute = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "Jacobi_Tenstorrent/kernels/compute/compute.cpp",
            core,
            ComputeConfig{
                .math_fidelity =
                    MathFidelity::HiFi4});  // There's different math fidelity modes (for the tensor engine)

        uint32_t in_addr = input_dram_buffer->address();
        uint32_t L_addr = L_dram_buffer->address();
        uint32_t U_addr = U_dram_buffer->address();
        uint32_t out_addr = output_dram_buffer->address();

        SetRuntimeArgs(program, reader, core, {in_addr, num_tiles, L_addr, U_addr});
        SetRuntimeArgs(program, compute, core, {num_tiles});
        SetRuntimeArgs(program, writer, core, {out_addr, num_tiles});

        // E escrevemos todos eles na DRAM do device
        distributed::EnqueueWriteMeshBuffer(
            cq, input_dram_buffer, tilize_nfaces(in, tt::constants::TILE_WIDTH, tt::constants::TILE_HEIGHT));
        distributed::EnqueueWriteMeshBuffer(
            cq, L_dram_buffer, tilize_nfaces(L, tt::constants::TILE_WIDTH, tt::constants::TILE_HEIGHT));
        distributed::EnqueueWriteMeshBuffer(
            cq, U_dram_buffer, tilize_nfaces(U, tt::constants::TILE_WIDTH, tt::constants::TILE_HEIGHT));

        fmt::print("enviando programa para o device\n");
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);

        std::vector<bfloat16> result_vec_tilized;
        distributed::EnqueueReadMeshBuffer(cq, result_vec_tilized, output_dram_buffer, true);

        fmt::print("result: {0}\n", (float)result_vec_tilized[result_vec_tilized.size() - 2]);

        std::vector<bfloat16> result_vec =
            untilize_nfaces(result_vec_tilized, tt::constants::TILE_WIDTH, tt::constants::TILE_HEIGHT);

        mesh_device->close();

        std::vector<float> result_vec_float(result_vec.size());

        for (size_t i = 0; i < result_vec.size(); ++i) {
            // result_vec_float[i] = (float)result_vec[i];
            result_vec_float[i] = (float)result_vec[i];
        }

        write_matrix_to_file(
            "/home/gian/Documentos/Unicamp/GeoBench/Tenstorrent/tt-metal/tt_metal/programming_examples/"
            "Jacobi_Tenstorrent/output.bin",
            result_vec_float);

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception! what: {}\n", e.what());
        throw;
    }
}
