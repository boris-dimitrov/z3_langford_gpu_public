// Copyright 2016 Boris Dimitrov, Portola Valley, CA 94028.
// Questions? Contact http://www.facebook.com/boris
//
// This program counts all permutations of the sequence 1, 1, 2, 2, 3, 3, ..., n, n
// in which the two appearances of each m are separated by precisely m other numbers.
// These permutations are called Langford pairings.  For n=3, only one such exists,
// (modulo left-to-right reversal):  2, 3, 1, 2, 1, 3.  Pairings exist only for n
// that are congruent with 0 or 3 mod 4, and their count grows very rapidly with n.
// See http://www.dialectrix.com/langford.html or Knuth volume 4a page 1 (chapter 7).
//
// The crux of this program does not use off-chip memory;  everything fits in GPU L2
// and (I hope) in CPU L1 cache.  No floating point, just int ops and array derefs.
//
// Compile on ubuntu with cuda8rc as follows:
//
//    nvcc -o langford -std=c++11 langford.cu
//
// Comparing this same algorithm on three different processors,
//
//               22nm i7-4790K        14nm E5-2699v4     16nm TitanX Pascal GPU
//                  4c @ 4 GHz         22c @ 2.8 GHz             28c @ 1.82 GHz
//   --------------------------------------------------------------------------
//
//   n = 15       28.1 seconds           8.9 seconds               2.35 seconds
//
//   n = 16      309.1 seconds          92.4 seconds               23.5 seconds
//
//   n = 19                n/a           41.62 hours                10.72 hours
//   --------------------------------------------------------------------------
//   normalized           1.0x                 3.25x                      12.5x
//   perf units
//
//   cost per
//   unit perf            $335                $1,415                        $96
//   (fall 2016)
//
// Matching perf to hardware,
//
//      1.0x == Late-2014 iMac5k with i7-4790K CPU running all 4 cores at 4 GHz
//
//     3.25x == Early-2016 Xeon E5-2699 v4 CPU running all 22 cores at 2.8 GHz
//              (overall about 3.25x faster than the reference iMac)
//
//     12.5x == Q3-2016 Titan X Pascal GPU running all 28 SM's at 1.82 GHz
//
//       25x == dual GPU (projection)
//
// Each GPU SM is 3.05x faster than its Xeon counterpart (2.67x with Turbo Boost).
// Overall, the $1,200 Pascal GPU is 3.9x faster than the $4,600 Broadwell CPU.
// That makes GPU 15x cheaper than CPU per unit of perf.
//
// This is GPU version z3 and the comparison was to CPU version z2
// (available upon request).
//

// The value of N is the only "input" to this program.
constexpr int n = 16;
static_assert(n >= 7, "so we can unroll n <= 6");

#include <iostream>
#include <chrono>
using namespace std;
using chrono::steady_clock;

#include <cuda_runtime.h>

struct Seconds {
    chrono::duration<double> d;
    Seconds(chrono::duration<double> d_) : d(d_) {}
};

ostream& operator << (ostream& out, const Seconds& seconds) {
    return (out << seconds.d.count());
}

constexpr int n_minus_1 = n - 1;
constexpr int two_n = 2 * n;

constexpr int padded_n = ((n + 3) >> 2) << 2;

constexpr int64_t lsb = 1;
constexpr int64_t msb = lsb << (2 * n - 1);
constexpr int64_t full = msb | (msb - 1);

// Fewer threads per block => more threads can be packed in L2 => higher "occupancy",
// which would normally be a good thing, except... it causes higher power draw,
// and occupancy > 0.25 actually results in lower perf.
//
// for n=15, threads_per_block=12 is optimal, with 16, 8, and 32 all pretty close
// for n=16, threads_per_blocks=16 and 32 take 23.5 seconds;  12 takes 24.1 seconds;  8 takes 25 sec
// for n=19, threads_per_blocks=32 takes 13.13 hours; 16 takes 11 hours; 8 takes 10.7 hours
// so 8 works fine overall, but we can micro-optimize for some n
constexpr int threads_per_block = (n == 15) ? 12
                                            : (n == 16) ? 16
                                                        : 8;

constexpr int div_up(int p, int q) {
    return (p + (q - 1)) / q;
}

//  We launch (n-1) * (2*n) * (2*n) * (2*n) threads.   Each thread starts with a specific initial
//  placement of m = 1, 2, 3, 4 and its job is to extend and complete that placement in every
//  possible way for m = 5, 6, ..., 2*n.  The placement of m = 1 is restricted to positions
//  0, 1, ..., n-2 in order to avoid double-counting left-to-right reversal twins.
//
//  The threads are organized in a 3D grid of blocks, with each block comprising a small number
//  of threads, say 8 or 16 or 32.  The threads within a block share the L2 of their streaming
//  multiprocessor.  They divide that L2 into separate areas used for their stacks and heaps.
//
//  All threads in a block share the placement of m = 3 and m = 4 which is directly given by
//  the block's y and z coordinates.  The block's x coordinate and each thread's x index together
//  encode the placement of m = 1 and m = 2 in that thread.
//
//  Alternative mappings (for example, a simple 1-D mapping) are possible, with similar perf.
//  This 3-D one was chosen after the 1-D version bumped into some undocumented per-dimension
//  limits in Cuda-8rc.

constexpr int blocks_x = div_up((n-1)*(2*n), threads_per_block);
constexpr int blocks_y = 2*n;
constexpr int blocks_z = 2*n;

dim3 blocks(blocks_x, blocks_y, blocks_z);

constexpr int total_threads_padded = blocks_x * blocks_y * blocks_z * threads_per_block;

__global__ void dfs(int64_t* d_count) {
    // shared vars are per thread-block and held in L2 cache
    // local scalar vars are per thread and held in registers
    __shared__ int64_t l2_stacks[threads_per_block][padded_n << 1];
    __shared__ int64_t l2_heaps[threads_per_block][padded_n];
    int64_t a, m, pos_k, avail, cnt;
    int top;
    // Nomenclature:
    //
    //     0 <= me <= n-2 is the left pos for m = 1
    //     0 <= me2 < 2*n is the left pos for m = 2, and me2r = me2 + 3 is the right pos
    //     0 <= me3 < 2*n is the left pos for m = 3, and me3r = me3 + 4 is the right pos
    //     0 <= me4 < 2*n is the left pos for m = 4, and me4r = me4 + 5 is the right pps
    //
    // By construction,
    //
    //     me3 = blockIdx.y,
    //     me4 = blockIdx.z,
    //     me and me2 are packed, in a special way, into (blockIdx.x, threadIdx.x)
    //
    const int tmp = blockIdx.x * threads_per_block + threadIdx.x;
    const int me = tmp / two_n;
    if (me < n_minus_1) {
        const int me2r = tmp - me * two_n + 3;
        const int me3r = blockIdx.y + 4;
        const int me4r = blockIdx.z + 5;
        if (me2r < two_n && me3r < two_n && me4r < two_n) {
            const int64_t a1 = (lsb << me) | (lsb << (me + 2));
            const int64_t a2 = (lsb << (me2r - 3)) | (lsb << (me2r));
            const int64_t a3 = (lsb << blockIdx.y) | (lsb << (me3r));
            const int64_t a4 = (lsb << blockIdx.z) | (lsb << (me4r));
            a = a1 | a2 | a3 | a4;
            // are a1, a2, a3, a4 pairwise disjoint?  that means we have valid placement for m=1,2,3,4
            if (a == (a1 + a2 + a3 + a4)) {
                // compute all positions where m = 5 can be placed, given that m=1,2,3,4 have been
                // placed already in the positions given by a1, a2, a3, a4 above
                avail = a ^ full;         // invert a;  note upper bits >= pos 2*n are all 1 (important
                avail &= (avail >> 6);    // for the correctness of these two lines as a block).
                // can our valid placement for m=1,2,3,4 be continued for m=5?
                if (avail) {
                    cnt = 0;
                    m = 5;
                    top = 0;
                    // record all possible continuations for m=4 into the stack and start DFS loop
                    auto& stack = l2_stacks[threadIdx.x];
                    auto& heap = l2_heaps[threadIdx.x];
                    stack[top++] = avail;
                    stack[top++] = m;
                    heap[m-1] = a;
                    while (top) {
                        m = stack[top - 1];
                        avail = stack[top - 2];
                        // extract the lowest bit that is set to 1 in avail
                        pos_k = avail & ~(avail - 1);
                        // clear that bit
                        avail ^= pos_k;
                        // "pop" that bit from the hybrid stack s
                        if (avail) {
                            stack[top - 2] = avail;
                        } else {
                            top -= 2;
                        }
                        // place m in that position
                        a  = heap[m-1] | pos_k | (pos_k << (m + 1));
                        ++m;
                        // the "avail" computed below has bit "k" set to 1 if and only if
                        // both of the positions "k" and "k + m + 1" in "a" contain 0
                        avail = a ^ full;
                        avail &= (avail >> (m + 1));
                        if (avail) {
                            if (m == n) {
                                // we've found another langford pairing, count it
                                ++cnt;
                            } else {
                                // push all possible ways to place m, to be explored in subsequent iterations
                                stack[top++] = avail;
                                stack[top++] = m;
                                heap[m-1] = a;
                            }
                        }
                    }
                    // Write this thread's result to off-chip memory.
                    const int blid = blockIdx.x + blocks_x * (blockIdx.y + blocks_y * blockIdx.z);
                    d_count[blid * threads_per_block + threadIdx.x] = cnt;
                }
            }
        }
    }
}

void cdo(const char* txt, cudaError_t err) {
    if (err == cudaSuccess) {
        return;
    }
    cout << "Failed to " << txt << " (Error code " << cudaGetErrorString(err) << ")\n" << flush;
    exit(-1);
}

void run() {
    int64_t* count;
    int64_t* d_count;
    int64_t total;
    {
        cout << "\n";
        cout << "\n";
        cout << "------\n";
        cout << "Computing Langford number L(2,n) for n = " << n << ".\n";
        cout << "\n";
        cout << "GPU init " << flush;
    }
    auto t0 = steady_clock::now();
    auto seconds_since = [](decltype(t0) start) {
        return Seconds(steady_clock::now() - start);
    };
    cudaEvent_t start, stop;
    cdo("create start timer event",
        cudaEventCreate(&start));
    cdo("create stop timer event",
        cudaEventCreate(&stop, cudaEventBlockingSync));
    cdo("allocate host memory",
        cudaMallocHost(&count, total_threads_padded * sizeof(*count)));
    for (int i=0; i<total_threads_padded; ++i) {
        count[i] = -1;
    }
    cdo("allocate GPU memory",
        cudaMalloc(&d_count, total_threads_padded * sizeof(*count)));
    cdo("copy initial values from host to GPU",
        cudaMemcpy(d_count, count, sizeof(*count) * total_threads_padded, cudaMemcpyDeviceToHost));
    cdo("insert start event",
        cudaEventRecord(start));
    {
        cout << "took " << seconds_since(t0) << " sec.\n";
        cout << "\n";
        cout << "Dispatching " << total_threads_padded << " threads (" << threads_per_block << " per block, "
             << blocks.x << " x " << blocks.y << " x " << blocks.z << " blocks).\n" << flush;
    }
    auto t1 = steady_clock::now();
    dfs<<<blocks, threads_per_block>>>(d_count);
    cdo("DFS kernel launch",
        cudaPeekAtLastError());
    cdo("insert stop event",
        cudaEventRecord(stop));
    {
        cout << "\n";
        cout << "GPU computation " << flush;
    }
    cdo("wait for GPU computation to complete",
        cudaEventSynchronize(stop));
    float milliseconds = 0;
    cdo("compute GPU elapsed time",
        cudaEventElapsedTime(&milliseconds, start, stop));
    {
        cout << "took " << milliseconds / 1000.0 << " sec on GPU clock, "
             << seconds_since(t1) << " sec on host clock.\n";
        cout << "\n";
        cout << "CPU reduction " << flush;
    }
    auto t2 = steady_clock::now();
    cdo("copy result from GPU to host",
        cudaMemcpy(count, d_count, sizeof(*count) * total_threads_padded, cudaMemcpyDeviceToHost));
    total = 0;
    for (int i=0; i<total_threads_padded; ++i) {
        if (count[i] >= 0) {
            total += count[i];
        }
    }
    {
        int64_t known_results[64];
        memset(&known_results[0], 0, sizeof(known_results));
        known_results[3]  = 1;
        known_results[4]  = 1;
        known_results[7]  = 26;
        known_results[8]  = 150;
        known_results[11] = 17792;
        known_results[12] = 108144;
        known_results[15] = 39809640ll;
        known_results[16] = 326721800ll;
        known_results[19] = 256814891280ll;
        known_results[20] = 2636337861200ll;
        known_results[23] = 3799455942515488ll;
        known_results[24] = 46845158056515936ll;
        // beyond 23, count does not fit in 64 bits and is not definitively known!
        cout << "took " << seconds_since(t2) << " sec.\n";
        cout << "\n";
        cout << "Result " << total << " for n = " << n << " " << flush;
        cout << ((total == known_results[n]) ? "matches" : "** DOES NOT MATCH **")
             << " previously known result.\n";
        cout << "------\n";
        cout << "\n";
        cout << "\n";
    }
    cdo("free GPU memory for result",
        cudaFree(d_count));
    cdo("free host memory for result",
        cudaFreeHost(count));
}

int main(int argc, char** argv) {
    run();
    return 0;
}
