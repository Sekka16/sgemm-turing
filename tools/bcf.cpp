#include <bits/stdc++.h>

using namespace std;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int VEC = 4;
constexpr int PAD = 4;
constexpr int BLOCK_SIZE = 128;

int main() {
    // === AsT: 只看第 1 次写入（i = 0），不引入 t（等价于 t=0）===
    {
        cout << "AsT (first write only, no t):\n";
        const int stride_colA = BK / VEC;        // 以 float4 计的列线程数（=BK/4）
        const int APITCH      = BM + PAD;        // 行跨度（含 padding）
        const int WARPS       = BLOCK_SIZE / 32; // block 内 warp 数

        for (int warp = 0; warp < WARPS; ++warp) {
            cout << "  Warp " << warp << " : " << endl;
            std::set<int> banks;
            for (int lane = 0; lane < 32; ++lane) {
                int tid   = warp * 32 + lane;
                int r_idx = tid / stride_colA;          // 0..（这组里的“行”）
                int c_idx = tid % stride_colA;          // 0..（这组里的“列（以float4计）”）

                // 转置后写入 AsT：idx = (c_idx*4 /*+ t*/)*APITCH + r_idx
                // 这里固定 t=0，不影响冲突阶数判断
                int idx   = (c_idx * 4) * APITCH + r_idx;   // 以 float 为单位的索引
                int bank  = idx % 32;                       // bank = (byte/4) % 32
                banks.insert(bank);

                // 如需逐线程打印，打开下一行
                cout << " lane=" << lane << " bank=" << bank << endl;
            }
            cout << "distinct banks = " << banks.size() << "\n";
        }
    }

    // === Bs: 只看第 1 次写入（i = 0）===
    {
        cout << "Bs (first write only):\n";
        const int stride_col = BN / VEC;        // 以 float4 计的列线程数
        const int BPITCH     = BN + PAD;        // 行跨度（含 padding），这里仅作为常数项出现
        const int WARPS      = BLOCK_SIZE / 32; // 这个 block 内 warp 数

        for (int warp = 0; warp < WARPS; ++warp) {
            cout << "  Warp " << warp << " : " << endl;
            std::set<int> banks;
            for (int lane = 0; lane < 32; ++lane) {
                int tid   = warp * 32 + lane;      // 块内逻辑线程号
                int r_idx = tid / stride_col;      // 行索引（kk 行）
                int c_idx = tid % stride_col;      // 以 float4 计的列号

                // 第一次写入：i = 0 ⇒ r_b = r_idx
                int idx   = r_idx * BPITCH + c_idx * 4;  // 以 float 为单位的索引（t=0）
                int bank  = idx % 32;                    // bank = (byte/4) % 32
                banks.insert(bank);

                // 如需逐线程打印，打开下一行
                cout << " lane=" << lane << " bank=" << bank << endl;
            }
            cout << "distinct banks = " << banks.size() << "\n";
        }
    }

    return 0;
}