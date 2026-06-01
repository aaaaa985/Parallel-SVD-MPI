#include "gkh.h"

#include "givens.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <deque>
#include <queue>
#include "mpi.h"

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define GKH_USE_NEON 1
#else
#define GKH_USE_NEON 0
#endif

#include <pthread.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace
{

    // 活动块 [l, r]（闭区间）表示一个尚未完全收敛的上二对角子问题。
    // 在该区间内，超对角线元素非零，你可以认为通过这个抽象结构给矩阵“分块”。
    struct Block
    {
        int l;
        int r;
    };

    // MPI 消息标签
    static constexpr int TAG_TASK = 1;
    static constexpr int TAG_RESULT = 2;
    static constexpr int TAG_STOP = 3;

    // worker 返回状态
    static constexpr int RESULT_CONVERGED = 1; // 当前 block 已完全收敛为 1x1
    static constexpr int RESULT_SPLIT = 2;     // 当前 block 已分裂为多个 block
    static constexpr int RESULT_LIMIT = 3;     // 达到局部迭代上限，未能继续分裂

    // 旋转日志类型。
    // worker 不直接更新完整 U/V，而是把旋转记录返回给 master。
    enum RotationSide
    {
        ROT_RIGHT_V = 0,
        ROT_LEFT_U = 1
    };

    struct RotationLog
    {
        int side;
        int k0;
        int k1;
        double c;
        double s;
    };

    // ================= MPI 辅助函数前置声明 =================
    // 这些函数的定义在文件后部，但 MPI master/worker 函数会提前调用它们，
    // 因此需要先声明，避免 C++ 编译时出现 “not declared in this scope”

    static bool handle_diagonal_zeros(Matrix &U, Matrix &B, Matrix &V, double tol);

    static std::vector<Block> split_active_blocks(Matrix &B, int n, double tol);

    static Matrix extract_block(const Matrix &B, int l, int r);

    static void merge_block(Matrix &B, const Matrix &localB, int l, int r);

    static void replay_rotations(Matrix &U, Matrix &V, const std::vector<RotationLog> &logs);

    static void make_nonnegative_and_sort(Matrix &U, Matrix &B, Matrix &V);

    // MPI 任务池调度策略。
    // 0: 大块优先（默认主版本）；1: FIFO 动态队列；2: 小块优先。
    // 编译示例：
    //   -DGKH_TASK_POLICY=1  启用 FIFO
    //   -DGKH_TASK_POLICY=2  启用小块优先
#ifndef GKH_TASK_POLICY
#define GKH_TASK_POLICY 0
#endif

    static int block_len(const Block &blk)
    {
        return blk.r - blk.l + 1;
    }

    // 大块优先任务池：长度更大的 block 优先分配。
    struct BlockCmp
    {
        bool operator()(const Block &a, const Block &b) const
        {
            return block_len(a) < block_len(b);
        }
    };

    // 小块优先任务池：长度更小的 block 优先分配，用作对照策略。
    struct SmallBlockCmp
    {
        bool operator()(const Block &a, const Block &b) const
        {
            return block_len(a) > block_len(b);
        }
    };

    class TaskPool
    {
    public:
        void push(const Block &blk)
        {
#if GKH_TASK_POLICY == 1
            fifo_queue_.push_back(blk);
#elif GKH_TASK_POLICY == 2
            small_first_queue_.push(blk);
#else
            large_first_queue_.push(blk);
#endif
        }

        Block pop()
        {
#if GKH_TASK_POLICY == 1
            Block blk = fifo_queue_.front();
            fifo_queue_.pop_front();
            return blk;
#elif GKH_TASK_POLICY == 2
            Block blk = small_first_queue_.top();
            small_first_queue_.pop();
            return blk;
#else
            Block blk = large_first_queue_.top();
            large_first_queue_.pop();
            return blk;
#endif
        }

        bool empty() const
        {
#if GKH_TASK_POLICY == 1
            return fifo_queue_.empty();
#elif GKH_TASK_POLICY == 2
            return small_first_queue_.empty();
#else
            return large_first_queue_.empty();
#endif
        }

        int size() const
        {
#if GKH_TASK_POLICY == 1
            return static_cast<int>(fifo_queue_.size());
#elif GKH_TASK_POLICY == 2
            return static_cast<int>(small_first_queue_.size());
#else
            return static_cast<int>(large_first_queue_.size());
#endif
        }

    private:
#if GKH_TASK_POLICY == 1
        std::deque<Block> fifo_queue_;
#elif GKH_TASK_POLICY == 2
        std::priority_queue<Block, std::vector<Block>, SmallBlockCmp> small_first_queue_;
#else
        std::priority_queue<Block, std::vector<Block>, BlockCmp> large_first_queue_;
#endif
    };

    static const char *task_policy_name()
    {
#if GKH_TASK_POLICY == 1
        return "fifo";
#elif GKH_TASK_POLICY == 2
        return "small";
#else
        return "large";
#endif
    }

    // MPI profiling 统计。
    // 只在 master 上汇总并写入 files/mpi_profile_npX.csv，避免影响 main.cpp 的标准输出格式。
    struct MpiProfile
    {
        int call_id = 0;
        int mpi_size = 1;

        int init_total_blocks = 0;
        int init_nontrivial_blocks = 0;
        int max_initial_block_len = 0;

        long long tasks_sent = 0;
        long long tasks_completed = 0;
        long long returned_blocks = 0;
        long long rotation_logs = 0;
        long long result_limit_tasks = 0;

        int max_task_len = 0;
        int max_queue_size = 0;

        double master_send_ms = 0.0;
        double master_wait_result_ms = 0.0;
        double master_recv_payload_ms = 0.0;
        double master_merge_ms = 0.0;
        double master_replay_ms = 0.0;
        double master_cleanup_handle_ms = 0.0;

        double worker_wait_ms_sum = 0.0;
        double worker_alloc_ms_sum = 0.0;
        double worker_recv_block_ms_sum = 0.0;
        double worker_compute_ms_sum = 0.0;
        double worker_pack_ms_sum = 0.0;
        double worker_send_ms_sum = 0.0;

        long long bytes_to_workers = 0;
        long long bytes_from_workers = 0;
        long long worker_localB_bytes = 0;
        long long worker_local_iters_sum = 0;
    };

    static double mpi_elapsed_ms(double t0, double t1)
    {
        return (t1 - t0) * 1000.0;
    }

    static void write_mpi_profile_csv(const MpiProfile &prof)
    {
        std::string filename;
#if GKH_TASK_POLICY == 0
        // 默认主策略仍沿用原文件名，保证原有实验脚本和报告数据不受影响。
        filename = "files/mpi_profile_np" + std::to_string(prof.mpi_size) + ".csv";
#else
        // 对照策略写入带策略名的文件，避免覆盖大块优先主版本的 profiling。
        filename = std::string("files/mpi_profile_") + task_policy_name() +
                   "_np" + std::to_string(prof.mpi_size) + ".csv";
#endif

        std::ifstream check(filename);
        const bool need_header = (!check.good()) ||
                                 (check.peek() == std::ifstream::traits_type::eof());
        check.close();

        std::ofstream fout(filename, std::ios::app);
        if (!fout)
        {
            return;
        }

        if (need_header)
        {
            fout << "call_id,"
                 << "mpi_size,"
                 << "init_total_blocks,"
                 << "init_nontrivial_blocks,"
                 << "max_initial_block_len,"
                 << "tasks_sent,"
                 << "tasks_completed,"
                 << "returned_blocks,"
                 << "rotation_logs,"
                 << "result_limit_tasks,"
                 << "max_task_len,"
                 << "max_queue_size,"
                 << "master_send_ms,"
                 << "master_wait_result_ms,"
                 << "master_recv_payload_ms,"
                 << "master_merge_ms,"
                 << "master_replay_ms,"
                 << "master_cleanup_handle_ms,"
                 << "worker_wait_ms_sum,"
                 << "worker_alloc_ms_sum,"
                 << "worker_recv_block_ms_sum,"
                 << "worker_compute_ms_sum,"
                 << "worker_pack_ms_sum,"
                 << "worker_send_ms_sum,"
                 << "bytes_to_workers,"
                 << "bytes_from_workers,"
                 << "worker_localB_bytes,"
                 << "worker_local_iters_sum\n";
        }

        fout << prof.call_id << ','
             << prof.mpi_size << ','
             << prof.init_total_blocks << ','
             << prof.init_nontrivial_blocks << ','
             << prof.max_initial_block_len << ','
             << prof.tasks_sent << ','
             << prof.tasks_completed << ','
             << prof.returned_blocks << ','
             << prof.rotation_logs << ','
             << prof.result_limit_tasks << ','
             << prof.max_task_len << ','
             << prof.max_queue_size << ','
             << prof.master_send_ms << ','
             << prof.master_wait_result_ms << ','
             << prof.master_recv_payload_ms << ','
             << prof.master_merge_ms << ','
             << prof.master_replay_ms << ','
             << prof.master_cleanup_handle_ms << ','
             << prof.worker_wait_ms_sum << ','
             << prof.worker_alloc_ms_sum << ','
             << prof.worker_recv_block_ms_sum << ','
             << prof.worker_compute_ms_sum << ','
             << prof.worker_pack_ms_sum << ','
             << prof.worker_send_ms_sum << ','
             << prof.bytes_to_workers << ','
             << prof.bytes_from_workers << ','
             << prof.worker_localB_bytes << ','
             << prof.worker_local_iters_sum << '\n';
    }

    // 对矩阵 M 的两行 r0, r1 左乘 Givens 旋转 [c s; -s c]。
    // 即 M <- L * M，其中 L 只作用在第 r0/r1 两行上。
    // 这类逐元素线性组合很适合向量化，SIMD/多线程中你也可以顺手的事把他们做了。
    static void apply_left_rows(Matrix &M, int r0, int r1, double c, double s)
    {
        int cols = M.cols();
        double* __restrict row0 = &M.at(r0, 0);
        double* __restrict row1 = &M.at(r1, 0);
        int j = 0;

    #if GKH_USE_NEON
        float64x2_t vc = vdupq_n_f64(c);
        float64x2_t vs = vdupq_n_f64(s);
        float64x2_t vneg_s = vdupq_n_f64(-s);

        for (; j + 1 < cols; j += 2)
        {
            float64x2_t a = vld1q_f64(row0 + j);
            float64x2_t b = vld1q_f64(row1 + j);

            float64x2_t r0v = vaddq_f64(vmulq_f64(vc, a), vmulq_f64(vs, b));
            float64x2_t r1v = vaddq_f64(vmulq_f64(vneg_s, a), vmulq_f64(vc, b));

            vst1q_f64(row0 + j, r0v);
            vst1q_f64(row1 + j, r1v);
        }
    #endif

        // 尾部处理
        for (; j < cols; ++j)
        {
            double a = row0[j];
            double b = row1[j];
            row0[j] = c * a + s * b;
            row1[j] = -s * a + c * b;
        }
    }

    // 对矩阵 M 的两列 c0, c1 右乘 Givens 旋转 [c s; -s c]。
    // 即 M <- M * R，其中 R 只作用在第 c0/c1 两列上。
    static void apply_right_cols(Matrix &M, int c0, int c1, double c, double s)
    {
        // 只做轻量优化，不做SIMD，避免出现非连续访问带来的性能问题。
        int rows = M.rows();
        for (int i = 0; i < rows; ++i)
        {
            double* row = &M.at(i, 0);
            double a = row[c0];
            double b = row[c1];
            row[c0] = a * c - b * s;
            row[c1] = a * s + b * c;
        }
    }

    // 对矩阵 M 的两行 r0, r1 左乘 Givens 旋转，但只更新列区间 [c_begin, c_end]。
    // 这个函数主要用于 B 的块内更新，避免不同 block 并行时写到彼此的区域。
    static void apply_left_rows_range(Matrix &M, int r0, int r1, double c, double s, int c_begin, int c_end)
    {
        double *__restrict row0 = &M.at(r0, 0);
        double *__restrict row1 = &M.at(r1, 0);

        int j = c_begin;

    #if GKH_USE_NEON
        float64x2_t vc = vdupq_n_f64(c);
        float64x2_t vs = vdupq_n_f64(s);
        float64x2_t vneg_s = vdupq_n_f64(-s);

        for (; j + 1 <= c_end; j += 2)
        {
            float64x2_t a = vld1q_f64(row0 + j);
            float64x2_t b = vld1q_f64(row1 + j);

            float64x2_t r0v = vaddq_f64(vmulq_f64(vc, a), vmulq_f64(vs, b));
            float64x2_t r1v = vaddq_f64(vmulq_f64(vneg_s, a), vmulq_f64(vc, b));

            vst1q_f64(row0 + j, r0v);
            vst1q_f64(row1 + j, r1v);
        }
    #endif

        for (; j <= c_end; ++j)
        {
            double a = row0[j];
            double b = row1[j];

            row0[j] = c * a + s * b;
            row1[j] = -s * a + c * b;
        }
    }

    // 对矩阵 M 的两列 c0, c1 右乘 Givens 旋转，但只更新行区间 [r_begin, r_end]。
    // 这个函数主要用于 B 的块内更新，避免不同 block 并行时写到彼此的区域。
    static void apply_right_cols_range(Matrix &M, int c0, int c1,
                                       double c, double s,
                                       int r_begin, int r_end)
    {
        for (int i = r_begin; i <= r_end; ++i)
        {
            double *row = &M.at(i, 0);

            double a = row[c0];
            double b = row[c1];

            row[c0] = a * c - b * s;
            row[c1] = a * s + b * c;
        }
    }

    static void accumulate_left_into_U(Matrix &U, int r0, int r1, double c, double s)
    {
        // 我们该怎样积累 U 和 V 的更新呢？
        // 以此处 U 的积累为例，让我们B <- L * B 时，我们必须维护的等式是 A = U * B * V^T
        // 如果 A = U * B * V^T 不成立，那么我们最终的SVD结果显然不是 A 的正确分解。
        // 由于正交矩阵和其转置的乘积是I，一个自然的想法是让 U <- U * L^T。
        // 这样就变成 A = (U * L^T) * (L * B) * V^T = U * B * V^T，等式得以保持。

        // 由于 L^T = [c -s; s c]，此处复用“右乘两列”接口并传入 -s。
        apply_right_cols(U, r0, r1, c, -s);
    }

    // 计算活动块 [l, r] 对应 B^T B 右下 2x2 主子块的 Wilkinson 偏移。
    // 偏移用于加速 QR 迭代收敛，并让 bulge chasing 过程更稳定。
    static double block_wilkinson_shift(const Matrix &B, int l, int r)
    {
        if (r == l)
        {
            return B.at(l, l) * B.at(l, l);
        }

        const double d1 = B.at(r - 1, r - 1);
        const double e1 = B.at(r - 1, r);
        const double d2 = B.at(r, r);
        const double e0 = (r - 1 > l) ? B.at(r - 2, r - 1) : 0.0;

        const double a = d1 * d1 + e0 * e0;
        const double b = d1 * e1;
        const double d = d2 * d2 + e1 * e1;

        const double tr = a + d;
        const double det = a * d - b * b;
        double disc = 0.25 * tr * tr - det;
        if (disc < 0.0)
        {
            disc = 0.0;
        }

        const double root = std::sqrt(disc);
        const double lam1 = 0.5 * tr + root;
        const double lam2 = 0.5 * tr - root;
        return (std::fabs(lam1 - d) <= std::fabs(lam2 - d)) ? lam1 : lam2;
    }

    // 将上二对角结构以外、且绝对值很小的元素强制置零。
    static void cleanup_bidiagonal(Matrix &B, double tol)
    {
        for (int i = 0; i < B.rows(); ++i)
        {
            for (int j = 0; j < B.cols(); ++j)
            {
                if (j != i && j != i + 1 && std::fabs(B.at(i, j)) <= tol)
                {
                    B.at(i, j) = 0.0;
                }
            }
        }
    }

    // 对活动块 [l, r] 执行一次“单块 GKH bulge chasing”迭代。
    // 流程：首次右乘引入 bulge -> 首次左乘消 bulge -> 交替右乘/左乘将 bulge 追赶到块末端。
    static void one_block_step(Matrix &U, Matrix &B, Matrix &V, int l, int r)
    {
        if (r <= l)
        {
            return;
        }

        const double mu = block_wilkinson_shift(B, l, r);

        double c = 1.0;
        double s = 0.0;
        double rr = 0.0;

        // 首次右乘：由 (d_l^2-mu, d_l*e_l) 构造。
        const double x = B.at(l, l) * B.at(l, l) - mu;
        const double z = B.at(l, l) * B.at(l, l + 1);
        givens_rotation(x, z, c, s, rr, false);
        apply_right_cols_range(B, l, l + 1, c, s, l, r);
        apply_right_cols(V, l, l + 1, c, s);

        // 首次左乘：消去 (l+1, l)。
        givens_rotation(B.at(l, l), B.at(l + 1, l), c, s, rr, true);
        apply_left_rows_range(B, l, l + 1, c, s, l, r);
        accumulate_left_into_U(U, l, l + 1, c, s);

        for (int k = l + 1; k <= r - 1; ++k)
        {
            // 右乘：消去 (k-1, k+1)
            givens_rotation(B.at(k - 1, k), B.at(k - 1, k + 1), c, s, rr, false);
            apply_right_cols_range(B, k, k + 1, c, s, l, r);
            apply_right_cols(V, k, k + 1, c, s);

            // 左乘：消去 (k+1, k)
            givens_rotation(B.at(k, k), B.at(k + 1, k), c, s, rr, true);
            apply_left_rows_range(B, k, k + 1, c, s, l, r);
            accumulate_left_into_U(U, k, k + 1, c, s);
        }
    }

    // MPI worker 使用的单块迭代版本。
    // 它只更新局部 B，不直接更新完整 U/V，而是记录 Givens 旋转日志。
    // global_offset 用于把局部下标转换成全局下标。
    static void one_block_step_record(Matrix &B,
                                      int local_l,
                                      int local_r,
                                      int global_offset,
                                      std::vector<RotationLog> &logs)
    {
        if (local_r <= local_l)
        {
            return;
        }

        const double mu = block_wilkinson_shift(B, local_l, local_r);

        double c = 1.0;
        double s = 0.0;
        double rr = 0.0;

        const double x = B.at(local_l, local_l) * B.at(local_l, local_l) - mu;
        const double z = B.at(local_l, local_l) * B.at(local_l, local_l + 1);

        givens_rotation(x, z, c, s, rr, false);
        apply_right_cols_range(B, local_l, local_l + 1, c, s, local_l, local_r);
        logs.push_back({ROT_RIGHT_V,
                        global_offset + local_l,
                        global_offset + local_l + 1,
                        c,
                        s});

        givens_rotation(B.at(local_l, local_l), B.at(local_l + 1, local_l), c, s, rr, true);
        apply_left_rows_range(B, local_l, local_l + 1, c, s, local_l, local_r);
        logs.push_back({ROT_LEFT_U,
                        global_offset + local_l,
                        global_offset + local_l + 1,
                        c,
                        s});

        for (int k = local_l + 1; k <= local_r - 1; ++k)
        {
            givens_rotation(B.at(k - 1, k), B.at(k - 1, k + 1), c, s, rr, false);
            apply_right_cols_range(B, k, k + 1, c, s, local_l, local_r);
            logs.push_back({ROT_RIGHT_V,
                            global_offset + k,
                            global_offset + k + 1,
                            c,
                            s});

            givens_rotation(B.at(k, k), B.at(k + 1, k), c, s, rr, true);
            apply_left_rows_range(B, k, k + 1, c, s, local_l, local_r);
            logs.push_back({ROT_LEFT_U,
                            global_offset + k,
                            global_offset + k + 1,
                            c,
                            s});
        }
    }

    // worker 对一个局部 block 反复做 bulge chasing，直到：
    // 1. block 已经完全收敛为 1x1；
    // 2. block 分裂成多个子 block；
    // 3. 达到局部迭代上限。
    static int solve_local_block_until_split(Matrix &localB,
                                             int global_l,
                                             int max_local_iter,
                                             double tol,
                                             std::vector<Block> &new_blocks,
                                             std::vector<RotationLog> &logs,
                                             int &local_iters)
    {
        const int len = localB.cols();

        int status = RESULT_LIMIT;
        local_iters = 0;

        for (int iter = 0; iter < max_local_iter; ++iter)
        {
            cleanup_bidiagonal(localB, tol);

            std::vector<Block> blocks = split_active_blocks(localB, len, tol);

            bool all_singletons = true;
            for (const auto &blk : blocks)
            {
                if (blk.r > blk.l)
                {
                    all_singletons = false;
                    break;
                }
            }

            if (all_singletons)
            {
                status = RESULT_CONVERGED;
                break;
            }

            if (blocks.size() > 1)
            {
                status = RESULT_SPLIT;
                break;
            }

            one_block_step_record(localB, 0, len - 1, global_l, logs);
            ++local_iters;
        }

        cleanup_bidiagonal(localB, tol);
        std::vector<Block> local_blocks = split_active_blocks(localB, len, tol);

        new_blocks.clear();
        for (auto blk : local_blocks)
        {
            blk.l += global_l;
            blk.r += global_l;
            new_blocks.push_back(blk);
        }

        return status;
    }

    static void send_task_to_worker(int worker, const Matrix &B, const Block &blk)
    {
        const int len = blk.r - blk.l + 1;
        int header[3] = {blk.l, blk.r, len};

        Matrix localB = extract_block(B, blk.l, blk.r);

        MPI_Send(header, 3, MPI_INT, worker, TAG_TASK, MPI_COMM_WORLD);
        MPI_Send(localB.data(), localB.size(), MPI_DOUBLE, worker, TAG_TASK, MPI_COMM_WORLD);
    }

    static void send_stop_to_worker(int worker)
    {
        int header[3] = {-1, -1, 0};
        MPI_Send(header, 3, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
    }

    static void gkh_worker_loop(int max_iter, double tol)
    {
        while (true)
        {
            MPI_Status status;
            int header[3] = {0, 0, 0};

            const double t_wait0 = MPI_Wtime();
            MPI_Recv(header, 3, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            const double t_wait1 = MPI_Wtime();

            if (status.MPI_TAG == TAG_STOP)
            {
                break;
            }

            const int global_l = header[0];
            const int global_r = header[1];
            const int len = header[2];

            const double t_alloc0 = MPI_Wtime();
            Matrix localB(len, len, 0.0);
            const double t_alloc1 = MPI_Wtime();

            const double t_recv0 = MPI_Wtime();
            MPI_Recv(localB.data(), localB.size(), MPI_DOUBLE, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            const double t_recv1 = MPI_Wtime();

            std::vector<Block> new_blocks;
            std::vector<RotationLog> logs;
            int local_iters = 0;

            const double t_compute0 = MPI_Wtime();
            const int local_status = solve_local_block_until_split(localB,
                                                                   global_l,
                                                                   max_iter,
                                                                   tol,
                                                                   new_blocks,
                                                                   logs,
                                                                   local_iters);
            const double t_compute1 = MPI_Wtime();

            const double t_pack0 = MPI_Wtime();

            int result_header[6];
            result_header[0] = global_l;
            result_header[1] = global_r;
            result_header[2] = len;
            result_header[3] = static_cast<int>(new_blocks.size());
            result_header[4] = static_cast<int>(logs.size());
            result_header[5] = local_status;

            std::vector<int> block_buf;
            block_buf.reserve(new_blocks.size() * 2);
            for (const auto &blk : new_blocks)
            {
                block_buf.push_back(blk.l);
                block_buf.push_back(blk.r);
            }

            std::vector<int> rot_ints;
            std::vector<double> rot_doubles;

            rot_ints.reserve(logs.size() * 3);
            rot_doubles.reserve(logs.size() * 2);

            for (const auto &rot : logs)
            {
                rot_ints.push_back(rot.side);
                rot_ints.push_back(rot.k0);
                rot_ints.push_back(rot.k1);
                rot_doubles.push_back(rot.c);
                rot_doubles.push_back(rot.s);
            }

            const double t_pack1 = MPI_Wtime();

            const long long bytes_recv =
                static_cast<long long>(3 * sizeof(int)) +
                static_cast<long long>(localB.size()) * static_cast<long long>(sizeof(double));

            long long bytes_sent =
                static_cast<long long>(6 * sizeof(int)) +
                static_cast<long long>(localB.size()) * static_cast<long long>(sizeof(double)) +
                static_cast<long long>(block_buf.size()) * static_cast<long long>(sizeof(int)) +
                static_cast<long long>(rot_ints.size()) * static_cast<long long>(sizeof(int)) +
                static_cast<long long>(rot_doubles.size()) * static_cast<long long>(sizeof(double));

            const double t_send0 = MPI_Wtime();

            MPI_Send(result_header, 6, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
            MPI_Send(localB.data(), localB.size(), MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD);

            if (!block_buf.empty())
            {
                MPI_Send(block_buf.data(),
                         static_cast<int>(block_buf.size()),
                         MPI_INT,
                         0,
                         TAG_RESULT,
                         MPI_COMM_WORLD);
            }

            if (!rot_ints.empty())
            {
                MPI_Send(rot_ints.data(),
                         static_cast<int>(rot_ints.size()),
                         MPI_INT,
                         0,
                         TAG_RESULT,
                         MPI_COMM_WORLD);
                
                MPI_Send(rot_doubles.data(),
                         static_cast<int>(rot_doubles.size()),
                         MPI_DOUBLE,
                         0,
                         TAG_RESULT,
                         MPI_COMM_WORLD);
            }

            const double t_send1 = MPI_Wtime();

            // 发送 profiling 数据。为了不改 result_header 的基本语义，单独追加三组 profile 消息。
            double prof_doubles[6];
            prof_doubles[0] = mpi_elapsed_ms(t_wait0, t_wait1);
            prof_doubles[1] = mpi_elapsed_ms(t_alloc0, t_alloc1);
            prof_doubles[2] = mpi_elapsed_ms(t_recv0, t_recv1);
            prof_doubles[3] = mpi_elapsed_ms(t_compute0, t_compute1);
            prof_doubles[4] = mpi_elapsed_ms(t_pack0, t_pack1);
            prof_doubles[5] = mpi_elapsed_ms(t_send0, t_send1);

            long long prof_longs[3];
            prof_longs[0] = bytes_recv;
            prof_longs[1] = bytes_sent;
            prof_longs[2] = static_cast<long long>(localB.size()) * static_cast<long long>(sizeof(double));

            int prof_ints[5];
            prof_ints[0] = local_iters;
            prof_ints[1] = len;
            prof_ints[2] = static_cast<int>(new_blocks.size());
            prof_ints[3] = static_cast<int>(logs.size());
            prof_ints[4] = local_status;

            MPI_Send(prof_doubles, 6, MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD);
            MPI_Send(prof_longs, 3, MPI_LONG_LONG, 0, TAG_RESULT, MPI_COMM_WORLD);
            MPI_Send(prof_ints, 5, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
        }
    }

    static bool gkh_master_loop(Matrix &U, Matrix &B, Matrix &V,
                                int max_iter, double tol, int mpi_size)
    {
        const int n = B.cols();

        static int profile_call_counter = 0;

        MpiProfile prof;
        prof.call_id = ++profile_call_counter;
        prof.mpi_size = mpi_size;

        cleanup_bidiagonal(B, tol);
        handle_diagonal_zeros(U, B, V, tol);

        std::vector<Block> init_blocks = split_active_blocks(B, n, tol);

        prof.init_total_blocks = static_cast<int>(init_blocks.size());

        TaskPool task_queue;

        for (const auto &blk : init_blocks)
        {
            const int len = blk.r - blk.l + 1;
            prof.max_initial_block_len = std::max(prof.max_initial_block_len, len);
        
            if (blk.r > blk.l)
            {
                ++prof.init_nontrivial_blocks;
                task_queue.push(blk);
            }
        }

        prof.max_queue_size = std::max(prof.max_queue_size, static_cast<int>(task_queue.size()));

        if (task_queue.empty())
        {
            cleanup_bidiagonal(B, tol);
            for (int i = 0; i < n - 1; ++i)
            {
                B.at(i, i + 1) = 0.0;
            }
            make_nonnegative_and_sort(U, B, V);
        
            write_mpi_profile_csv(prof);
            return true;
        }

        std::deque<int> idle_workers;
        for (int worker = 1; worker < mpi_size; ++worker)
        {
            idle_workers.push_back(worker);
        }

        int busy_workers = 0;
        bool global_converged = true;

        auto dispatch_tasks = [&]()
        {
            while (!idle_workers.empty() && !task_queue.empty())
            {
                int worker = idle_workers.front();
                idle_workers.pop_front();

                Block blk = task_queue.pop();

                const int len = blk.r - blk.l + 1;
                prof.max_task_len = std::max(prof.max_task_len, len);

                const double t_send0 = MPI_Wtime();
                send_task_to_worker(worker, B, blk);
                const double t_send1 = MPI_Wtime();
        
                prof.master_send_ms += mpi_elapsed_ms(t_send0, t_send1);
                prof.bytes_to_workers +=
                    static_cast<long long>(3 * sizeof(int)) +
                    static_cast<long long>(len) * static_cast<long long>(len) * static_cast<long long>(sizeof(double));

                ++prof.tasks_sent;
                ++busy_workers;

                prof.max_queue_size = std::max(prof.max_queue_size, static_cast<int>(task_queue.size()));
            }
        };

        dispatch_tasks();

        while (busy_workers > 0)
        {
            MPI_Status status;
            int result_header[6] = {0, 0, 0, 0, 0, 0};

            const double t_wait_result0 = MPI_Wtime();
            MPI_Recv(result_header, 6, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
            const double t_wait_result1 = MPI_Wtime();

            prof.master_wait_result_ms += mpi_elapsed_ms(t_wait_result0, t_wait_result1);

            const int worker = status.MPI_SOURCE;
            const int global_l = result_header[0];
            const int global_r = result_header[1];
            const int len = result_header[2];
            const int num_blocks = result_header[3];
            const int num_rotations = result_header[4];
            const int local_status = result_header[5];

            const double t_recv_payload0 = MPI_Wtime();

            Matrix localB(len, len, 0.0);
            MPI_Recv(localB.data(), localB.size(), MPI_DOUBLE, worker, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            std::vector<Block> new_blocks;
            if (num_blocks > 0)
            {
                std::vector<int> block_buf(num_blocks * 2);
                MPI_Recv(block_buf.data(),
                         static_cast<int>(block_buf.size()),
                         MPI_INT,
                         worker,
                         TAG_RESULT,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

                for (int i = 0; i < num_blocks; ++i)
                {
                    new_blocks.push_back({block_buf[2 * i], block_buf[2 * i + 1]});
                }
            }

            std::vector<RotationLog> logs;
            if (num_rotations > 0)
            {
                std::vector<int> rot_ints(num_rotations * 3);
                std::vector<double> rot_doubles(num_rotations * 2);

                MPI_Recv(rot_ints.data(),
                         static_cast<int>(rot_ints.size()),
                         MPI_INT,
                         worker,
                         TAG_RESULT,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

                MPI_Recv(rot_doubles.data(),
                         static_cast<int>(rot_doubles.size()),
                         MPI_DOUBLE,
                         worker,
                         TAG_RESULT,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

                logs.reserve(num_rotations);
                for (int i = 0; i < num_rotations; ++i)
                {
                    RotationLog rot;
                    rot.side = rot_ints[3 * i];
                    rot.k0 = rot_ints[3 * i + 1];
                    rot.k1 = rot_ints[3 * i + 2];
                    rot.c = rot_doubles[2 * i];
                    rot.s = rot_doubles[2 * i + 1];
                    logs.push_back(rot);
                }
            }

            const double t_recv_payload1 = MPI_Wtime();
            prof.master_recv_payload_ms += mpi_elapsed_ms(t_recv_payload0, t_recv_payload1);

            double worker_prof_doubles[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            long long worker_prof_longs[3] = {0, 0, 0};
            int worker_prof_ints[5] = {0, 0, 0, 0, 0};

            MPI_Recv(worker_prof_doubles, 6, MPI_DOUBLE, worker, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(worker_prof_longs, 3, MPI_LONG_LONG, worker, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(worker_prof_ints, 5, MPI_INT, worker, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            prof.worker_wait_ms_sum += worker_prof_doubles[0];
            prof.worker_alloc_ms_sum += worker_prof_doubles[1];
            prof.worker_recv_block_ms_sum += worker_prof_doubles[2];
            prof.worker_compute_ms_sum += worker_prof_doubles[3];
            prof.worker_pack_ms_sum += worker_prof_doubles[4];
            prof.worker_send_ms_sum += worker_prof_doubles[5];

            prof.bytes_from_workers += worker_prof_longs[1];
            prof.worker_localB_bytes += worker_prof_longs[2];
            prof.worker_local_iters_sum += worker_prof_ints[0];

            ++prof.tasks_completed;
            prof.returned_blocks += num_blocks;
            prof.rotation_logs += num_rotations;

            if (local_status == RESULT_LIMIT)
            {
                ++prof.result_limit_tasks;
            }

            const double t_merge0 = MPI_Wtime();
            merge_block(B, localB, global_l, global_r);
            const double t_merge1 = MPI_Wtime();
            prof.master_merge_ms += mpi_elapsed_ms(t_merge0, t_merge1);

            const double t_replay0 = MPI_Wtime();
            replay_rotations(U, V, logs);
            const double t_replay1 = MPI_Wtime();
            prof.master_replay_ms += mpi_elapsed_ms(t_replay0, t_replay1);

            const double t_cleanup0 = MPI_Wtime();
            cleanup_bidiagonal(B, tol);
            handle_diagonal_zeros(U, B, V, tol);
            const double t_cleanup1 = MPI_Wtime();
            prof.master_cleanup_handle_ms += mpi_elapsed_ms(t_cleanup0, t_cleanup1);

            if (local_status == RESULT_LIMIT)
            {
                global_converged = false;
            }

            for (const auto &blk : new_blocks)
            {
                if (blk.r > blk.l)
                {
                    task_queue.push(blk);
                }
            }

            prof.max_queue_size = std::max(prof.max_queue_size, static_cast<int>(task_queue.size()));

            --busy_workers;
            idle_workers.push_back(worker);

            dispatch_tasks();
        }

        for (int worker = 1; worker < mpi_size; ++worker)
        {
            send_stop_to_worker(worker);
        }

        cleanup_bidiagonal(B, tol);
        for (int i = 0; i < n - 1; ++i)
        {
            B.at(i, i + 1) = 0.0;
        }
        make_nonnegative_and_sort(U, B, V);

        write_mpi_profile_csv(prof);

        return global_converged && task_queue.empty();
    }

    // Pthread 工作线程共享的任务上下文
    struct BlockTaskContext
    {
        Matrix *U;
        Matrix *B;
        Matrix *V;
        const std::vector<Block> *jobs;
        std::atomic<int> *next_job;
    };

    // Pthread 工作线程入口函数
    static void *block_worker_func(void *arg)
    {
        BlockTaskContext *ctx = static_cast<BlockTaskContext *>(arg);

        while (true)
        {
            int pos = ctx->next_job->fetch_add(1, std::memory_order_relaxed);
            int total = static_cast<int>(ctx->jobs->size());

            if (pos >= total)
            {
                break;
            }

            // 为了尽量保持原串行版本“从右到左”的处理倾向，
            // 这里按 jobs 的逆序取任务。
            const Block &blk = (*(ctx->jobs))[total - 1 - pos];

            one_block_step(*(ctx->U), *(ctx->B), *(ctx->V), blk.l, blk.r);
        }

        return nullptr;
    }

    // 获取本次 GKH block 并行阶段使用的线程数
    // 由于 test.sh/qsub 运行方式下环境变量无法稳定传递到计算节点，
    // 因此这里通过手动修改返回值测试 1、2、4、8 线程
    static int get_gkh_thread_count()
    {
        return 8;
    }

    // OpenMP 调度策略：
    // 0: schedule(static)
    // 1: schedule(dynamic, 1)
    // 2: schedule(dynamic, 2)
    // 3: schedule(guided)
    #ifndef OMP_SCHEDULE_KIND
    #define OMP_SCHEDULE_KIND 2
    #endif

    // 并行执行当前迭代轮次中的所有非平凡活动块
    static void run_block_steps_pthread(Matrix &U, Matrix &B, Matrix &V, const std::vector<Block> &blocks)
    {
        std::vector<Block> jobs;
        jobs.reserve(blocks.size());

        for (const auto &blk : blocks)
        {
            if (blk.r > blk.l)
            {
                jobs.push_back(blk);
            }
        }

        if (jobs.empty())
        {
            return;
        }

        int thread_count = get_gkh_thread_count();

        if (thread_count > static_cast<int>(jobs.size()))
        {
            thread_count = static_cast<int>(jobs.size());
        }

        // 任务太少时直接串行，避免 pthread 创建开销反而拖慢。
        if (thread_count <= 1)
        {
            for (int i = static_cast<int>(jobs.size()) - 1; i >= 0; --i)
            {
                one_block_step(U, B, V, jobs[i].l, jobs[i].r);
            }
            return;
        }

        std::atomic<int> next_job(0);

        BlockTaskContext ctx;
        ctx.U = &U;
        ctx.B = &B;
        ctx.V = &V;
        ctx.jobs = &jobs;
        ctx.next_job = &next_job;

        // 最多启动 thread_count - 1 个子线程，主线程也参与计算。
        std::vector<pthread_t> workers(thread_count - 1);

        for (int i = 0; i < thread_count - 1; ++i)
        {
            pthread_create(&workers[i], nullptr, block_worker_func, &ctx);
        }

        // 主线程也作为工作线程执行任务。
        block_worker_func(&ctx);

        for (int i = 0; i < thread_count - 1; ++i)
        {
            pthread_join(workers[i], nullptr);
        }
    }

    #ifdef USE_OPENMP
    static void run_block_steps_openmp(Matrix &U, Matrix &B, Matrix &V, const std::vector<Block> &blocks)
    {
        std::vector<Block> jobs;
        jobs.reserve(blocks.size());

        for (const auto &blk : blocks)
        {
            if (blk.r > blk.l)
            {
                jobs.push_back(blk);
            }
        }

        if (jobs.empty())
        {
            return;
        }

        int thread_count = get_gkh_thread_count();

        if (thread_count > static_cast<int>(jobs.size()))
        {
            thread_count = static_cast<int>(jobs.size());
        }

        if (thread_count <= 1)
        {
            for (int i = static_cast<int>(jobs.size()) - 1; i >= 0; --i)
            {
                one_block_step(U, B, V, jobs[i].l, jobs[i].r);
            }
            return;
        }

        omp_set_num_threads(thread_count);

    #if OMP_SCHEDULE_KIND == 0
    #pragma omp parallel for schedule(static) default(none) shared(U, B, V, jobs)
        for (int pos = 0; pos < static_cast<int>(jobs.size()); ++pos)
        {
            int idx = static_cast<int>(jobs.size()) - 1 - pos;
            const Block &blk = jobs[idx];
            one_block_step(U, B, V, blk.l, blk.r);
        }
    #elif OMP_SCHEDULE_KIND == 1
    #pragma omp parallel for schedule(dynamic, 1) default(none) shared(U, B, V, jobs)
        for (int pos = 0; pos < static_cast<int>(jobs.size()); ++pos)
        {
            int idx = static_cast<int>(jobs.size()) - 1 - pos;
            const Block &blk = jobs[idx];
            one_block_step(U, B, V, blk.l, blk.r);
        }
    #elif OMP_SCHEDULE_KIND == 2
    #pragma omp parallel for schedule(dynamic, 2) default(none) shared(U, B, V, jobs)
        for (int pos = 0; pos < static_cast<int>(jobs.size()); ++pos)
        {
            int idx = static_cast<int>(jobs.size()) - 1 - pos;
            const Block &blk = jobs[idx];
            one_block_step(U, B, V, blk.l, blk.r);
        }
    #elif OMP_SCHEDULE_KIND == 3
    #pragma omp parallel for schedule(guided) default(none) shared(U, B, V, jobs)
        for (int pos = 0; pos < static_cast<int>(jobs.size()); ++pos)
        {
            int idx = static_cast<int>(jobs.size()) - 1 - pos;
            const Block &blk = jobs[idx];
            one_block_step(U, B, V, blk.l, blk.r);
        }
    #else
    #error "Unknown OMP_SCHEDULE_KIND"
    #endif
    }
    #endif

    static void run_block_steps_parallel(Matrix &U, Matrix &B, Matrix &V, const std::vector<Block> &blocks)
    {
    #ifdef USE_OPENMP
        run_block_steps_openmp(U, B, V, blocks);
    #else
        run_block_steps_pthread(U, B, V, blocks);
    #endif
    }

    // 处理“对角元 d_k 近零但超对角 e_k 未近零”的情况。
    // 思路与单块追赶类似：先右乘把 e_i 消掉，再左乘清理新引入的次对角 bulge，
    // 把这个问题逐步向右传递，直到块末端。
    static bool chase_zero_diagonal(Matrix &U, Matrix &B, Matrix &V, int k, double tol)
    {
        const int m = B.rows();
        const int n = B.cols();
        if (k < 0 || k >= n - 1)
        {
            return false;
        }

        // d_k ~ 0 且 e_k 还未收敛时，按 lim_1 思路进行压缩追赶：
        // 1) 右乘消去第 k 行的 e_k；2) 左乘消去引入的次对角 bulge；
        // 然后把问题传递到下一行，直到末端。
        if (std::fabs(B.at(k, k + 1)) <= tol)
        {
            return false;
        }

        bool changed = false;
        for (int i = k; i <= n - 2; ++i)
        {
            double c = 1.0;
            double s = 0.0;
            double rr = 0.0;

            // 右乘：使第 i 行满足 [d_i, e_i] * G = [r, 0]。
            givens_rotation(B.at(i, i), B.at(i, i + 1), c, s, rr, false);
            apply_right_cols(B, i, i + 1, c, s);
            apply_right_cols(V, i, i + 1, c, s);

            // 左乘：消去 (i+1, i) 处由右乘引入的 bulge。
            if (i + 1 < m)
            {
                givens_rotation(B.at(i, i), B.at(i + 1, i), c, s, rr, true);
                apply_left_rows(B, i, i + 1, c, s);
                accumulate_left_into_U(U, i, i + 1, c, s);
            }

            changed = true;
        }

        cleanup_bidiagonal(B, tol);
        return changed;
    }

    // 扫描所有 d_k≈0 的位置：若对应 e_k 仍显著非零，则调用追赶过程压缩该异常结构。
    // 返回值表示本轮是否对 B/U/V 做了实际更新。
    static bool handle_diagonal_zeros(Matrix &U, Matrix &B, Matrix &V, double tol)
    {
        const int n = B.cols();
        bool changed = false;

        const double eps = std::numeric_limits<double>::epsilon();
        const double diag_tol = tol;
        const double super_tol = tol * (1.0 + 10.0 * eps);

        for (int k = 0; k < n - 1; ++k)
        {
            if (std::fabs(B.at(k, k)) <= diag_tol && std::fabs(B.at(k, k + 1)) > super_tol)
            {
                if (chase_zero_diagonal(U, B, V, k, tol))
                {
                    changed = true;
                }
            }
        }

        return changed;
    }

    // 根据超对角线是否“足够小”对问题进行分块。
    // 若 |e_k| <= tol*(|d_k|+|d_{k+1}|+1)，认为该位置可解耦并直接置零。
    // 最终会得到一系列小矩阵。
    static std::vector<Block> split_active_blocks(Matrix &B, int n, double tol)
    {
        for (int k = 0; k < n - 1; ++k)
        {
            const double a = std::fabs(B.at(k, k));
            const double d = std::fabs(B.at(k + 1, k + 1));
            const double crit = tol * (a + d + 1.0);
            if (std::fabs(B.at(k, k + 1)) <= crit)
            {
                B.at(k, k + 1) = 0.0;
            }
        }

        std::vector<Block> blocks;
        int l = 0;
        while (l < n)
        {
            int r = l;
            while (r < n - 1 && std::fabs(B.at(r, r + 1)) > 0.0)
            {
                ++r;
            }
            blocks.push_back({l, r});
            l = r + 1;
        }
        return blocks;
    }

    // 从全局 B 中提取活动块 [l, r]，得到局部 len x len 矩阵。
    // 当前 GKH 活动块只涉及 B[l:r, l:r]。
    static Matrix extract_block(const Matrix &B, int l, int r)
    {
        const int len = r - l + 1;
        Matrix localB(len, len, 0.0);

        // Matrix 使用 row-major 连续存储。局部块的每一行在 localB 中连续，
        // 在全局 B 中对应行的 [l, r] 区间也连续，因此可以按行拷贝，
        // 避免双层 at() 访问带来的重复下标计算和函数调用开销。
        const int global_cols = B.cols();
        for (int i = 0; i < len; ++i)
        {
            const double *src = B.data() + static_cast<long long>(l + i) * global_cols + l;
            double *dst = localB.data() + static_cast<long long>(i) * len;
            std::copy(src, src + len, dst);
        }

        return localB;
    }

    // 将 worker 返回的局部 block 写回全局 B。
    static void merge_block(Matrix &B, const Matrix &localB, int l, int r)
    {
        const int len = r - l + 1;

        // 与 extract_block 对称地按行连续写回，提高 cache 友好性。
        const int global_cols = B.cols();
        for (int i = 0; i < len; ++i)
        {
            double *dst = B.data() + static_cast<long long>(l + i) * global_cols + l;
            const double *src = localB.data() + static_cast<long long>(i) * len;
            std::copy(src, src + len, dst);
        }
    }

    // 在 master 上重放 worker 记录的 Givens 旋转，从而更新全局 U/V。
    static void replay_rotations(Matrix &U, Matrix &V, const std::vector<RotationLog> &logs)
    {
        for (const auto &rot : logs)
        {
            if (rot.side == ROT_RIGHT_V)
            {
                apply_right_cols(V, rot.k0, rot.k1, rot.c, rot.s);
            }
            else
            {
                accumulate_left_into_U(U, rot.k0, rot.k1, rot.c, rot.s);
            }
        }
    }

    // 收尾步骤：
    // 1) 把奇异值（对角元）统一调整为非负；
    // 2) 按降序重排奇异值，同时同步重排 U、V 对应列。
    // 最终得到常见的 SVD 规范形式：sigma_1 >= sigma_2 >= ... >= 0。
    // 这个函数你不用太在意，后续任务也不会明确涉及它。
    static void make_nonnegative_and_sort(Matrix &U, Matrix &B, Matrix &V)
    {
        const int m = B.rows();
        const int n = B.cols();

        for (int i = 0; i < n; ++i)
        {
            if (B.at(i, i) < 0.0)
            {
                B.at(i, i) = -B.at(i, i);
                for (int r = 0; r < m; ++r)
                {
                    U.at(r, i) = -U.at(r, i);
                }
            }
        }

        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i)
        {
            idx[i] = i;
        }
        std::sort(idx.begin(), idx.end(), [&](int a, int b)
                  { return B.at(a, a) > B.at(b, b); });

        Matrix U2 = U;
        Matrix V2 = V;
        Matrix D(B.rows(), B.cols(), 0.0);

        for (int new_i = 0; new_i < n; ++new_i)
        {
            const int old_i = idx[new_i];
            D.at(new_i, new_i) = B.at(old_i, old_i);

            for (int r = 0; r < U.rows(); ++r)
            {
                U2.at(r, new_i) = U.at(r, old_i);
            }
            for (int r = 0; r < V.rows(); ++r)
            {
                V2.at(r, new_i) = V.at(r, old_i);
            }
        }

        U = U2;
        V = V2;
        B = D;
    }

} // namespace

// 从“上二对角矩阵 B”出发执行 Golub-Kahan SVD 迭代（改进版）：
// - 输入输出满足 A = U * B * V^T 不变；
// - 迭代中自动分块、处理对角近零、并在每个活动块上做 bulge chasing；
// - 成功收敛后，B 被整理为非负且降序的对角矩阵（其对角元即奇异值）。
bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V, int max_iter, double tol)
{
    const int m = B.rows();
    const int n = B.cols();

    if (m < n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: requires m >= n");
    }
    if (U.rows() != m || U.cols() != m)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: U must be m x m");
    }
    if (V.rows() != n || V.cols() != n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: V must be n x n");
    }

    bool converged = false;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        // 清理数值噪声，并优先处理 d_k≈0 的特殊情形。
        cleanup_bidiagonal(B, tol);
        handle_diagonal_zeros(U, B, V, tol);

        // 根据超对角线断点拆分活动块
        // 这里子矩阵间是相互独立的，所以此处具有很大的并行潜力：你可以尝试多线程/多进程进行处理
        // 但根据算法，收集 Givens 旋转并更新 U/V 需要在每个块内顺序执行，所以这可能给并行带来麻烦。
        std::vector<Block> blocks = split_active_blocks(B, n, tol);

        // 若全部是 1x1 块，说明所有超对角都已收敛为 0。
        bool all_singletons = true;
        for (const auto &blk : blocks)
        {
            if (blk.r > blk.l)
            {
                all_singletons = false;
                break;
            }
        }

        if (all_singletons)
        {
            converged = true;
            break;
        }

        // 从右到左处理每个非平凡块，减少末端块对前面块的干扰。
        //for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i)
        //{
            //if (blocks[i].r > blocks[i].l)
            //{
                //one_block_step(U, B, V, blocks[i].l, blocks[i].r);
            //}
        //}
        // 对本轮尚未收敛的活动块进行并行 GKH 迭代
        run_block_steps_parallel(U, B, V, blocks);
    }

    // 迭代结束后统一结构清理与标准化输出。
    cleanup_bidiagonal(B, tol);
    for (int i = 0; i < n - 1; ++i)
    {
        B.at(i, i + 1) = 0.0;
    }
    make_nonnegative_and_sort(U, B, V);

    return converged;
}

bool gkh_svd_from_bidiagonal_mpi(Matrix &U, Matrix &B, Matrix &V,
                                 int max_iter, double tol)
{
    int rank = 0;
    int size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size <= 1)
    {
        return gkh_svd_from_bidiagonal(U, B, V, max_iter, tol);
    }

    if (rank == 0)
    {
        return gkh_master_loop(U, B, V, max_iter, tol, size);
    }

    gkh_worker_loop(max_iter, tol);
    return true;
}