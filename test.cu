#include "sgemm.cuh"
#include "test.cuh"
// #include "bt_tunner.cuh"
// #include "db_tunner.cuh"
// #include "f4_tunner.cuh"
#include <iostream>

int main() {
    test_sgemm(sgemm_cublas);
    // test_sgemm(sgemm_block_tile_v1<SGEMMConfigConst<128, 128, 8, 8, 8, 256>>);
    // test_sgemm(sgemm_block_tile_v1<SGEMMConfigConst<128, 128, 8, 8, 4, 512>>);
    // run_block_tile_tuning();
    // run_double_buffer_tuning();
    // test_sgemm(sgemm_float4_v1<SGEMMConfigConst<128, 128, 16, 16, 8, 128>>);
    // test_sgemm(sgemm_float4_v3<SGEMMConfigConst<128, 128, 16, 16, 8, 128>>);
    compare_sgemm(sgemm_cublas, sgemm_block_tile_v1<SGEMMConfigConst<128, 128, 16, 16, 8, 128>>);
    compare_sgemm(sgemm_block_tile_v1<SGEMMConfigConst<128, 128, 16, 16, 8, 128>>, sgemm_float4_v1<SGEMMConfigConst<128, 128, 16, 16, 8, 128>>);
    // compare_sgemm(sgemm_cublas, sgemm_float4_v2<SGEMMConfigConst<128, 128, 16, 16, 8, 128>>);
    // run_float4_tuning();
    // run_block_tile_tuning();
    return 0;
}