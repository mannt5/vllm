#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if defined(USE_ROCM)
typedef __hip_bfloat16 nv_bfloat16;
#endif

#include <iostream>
#include <array>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

namespace vllm {
#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

constexpr int kMaxBlocks = 36;
// Counter may overflow, but it's fine since unsigned int overflow is
// well-defined behavior.
using FlagType = uint32_t;

enum CA_2STAGE_COMPRESSION {
  NONE,
  SYMM_4BIT,
  SYMM_8BIT,
  ASYMM_4BIT,
  ASYMM_8BIT,
  CAST_4BIT,
  CAST_8BIT
};

struct Signal {
  alignas(128) FlagType start[kMaxBlocks][8];
  alignas(128) FlagType end[kMaxBlocks][8];
  alignas(128) FlagType _flag[kMaxBlocks];  // incremental flags for each rank
};

struct __align__(16) RankData {
  const void*
#if !defined(USE_ROCM)
      __restrict__
#endif
      ptrs[8];
};

struct __align__(16) RankSignals {
  Signal* signals[8];
};

// like std::array, but aligned
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T, int sz, int BITS>
struct quantized_t {
  T scale;
  char data[sz / (8 / BITS)] = {0};
  using type = T;
  static constexpr int size = sz;
  static constexpr int bits = BITS;
};

template <typename T, int sz, int BITS>
struct asymm_quantized_t {
  T scale, zero;
  unsigned char data[sz / (8 / BITS)] = {0};
  using type = T;
  static constexpr int size = sz;

  static_assert(BITS == 4 or BITS == 8);
  static constexpr int bits = BITS;
};

// use packed type to maximize memory efficiency
// goal: generate ld.128 and st.128 instructions
template <typename T>
struct packed_t {
  // the (P)acked type for load/store
  using P = array_t<T, 16 / sizeof(T)>;
  // the (A)ccumulator type for reduction
  using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

// scalar cast functions
template <typename I, typename O>
DINLINE O cast_s(I val);

template <>
DINLINE float cast_s(float val) {
  return val;
}

template <>
DINLINE half cast_s(half val) {
  return val;
}

template <>
DINLINE float cast_s(half val) {
  return __half2float(val);
}

template <>
DINLINE half cast_s(float val) {
  return __float2half(val);
}

template <>
DINLINE char cast_s(float val) {
  return static_cast<char>(val);
}

template <>
DINLINE unsigned char cast_s(float val) {
  return static_cast<unsigned char>(val);
}

template <>
DINLINE char cast_s(half val) {
  return static_cast<char>(__half2float(val));
}

template <>
DINLINE unsigned char cast_s(half val) {
  return static_cast<unsigned char>(__half2float(val));
}

template <>
DINLINE float cast_s(char val) {
  return static_cast<float>(val);
}

template <>
DINLINE float cast_s(unsigned char val) {
  return static_cast<float>(val);
}

template <>
DINLINE half cast_s(char val) {
  return __float2half(static_cast<float>(val));
}

template <>
DINLINE half cast_s(unsigned char val) {
  return __float2half(static_cast<float>(val));
}

template <typename T>
DINLINE T max_abs_s(T a, T b);

template <typename T>
DINLINE T max_s(T a, T b);

template <typename T>
DINLINE T min_s(T a, T b);

template <>
DINLINE float max_abs_s(float a, float b) {
  return fmaxf(fabsf(a), fabsf(b));
}

template <>
DINLINE float max_s(float a, float b) {
  return fmaxf(a, b);
}

template <>
DINLINE float min_s(float a, float b) {
  return fminf(a, b);
}

template <>
DINLINE half max_abs_s(half a, half b) {
  return __hmax(__habs(a), __habs(b));
}

template <>
DINLINE half max_s(half a, half b) {
  return __hmax(a, b);
}

template <>
DINLINE half min_s(half a, half b) {
  return __hmin(a, b);
}

template <typename T>
DINLINE T add_s(T a, T b);

template <>
DINLINE float add_s(float a, float b) {
  return a + b;
}

template <>
DINLINE half add_s(half a, half b) {
  return __hadd(a, b);
}

template <typename T>
DINLINE T sub_s(T a, T b);

template <>
DINLINE float sub_s(float a, float b) {
  return a - b;
}

template <>
DINLINE half sub_s(half a, half b) {
  return __hsub(a, b);
}

template <typename T>
DINLINE T mul_s(T a, T b);

template <>
DINLINE float mul_s(float a, float b) {
  return a * b;
}

template <>
DINLINE half mul_s(half a, half b) {
  return __hmul(a, b);
}

template <typename T>
DINLINE T div_s(T a, T b);

template <>
DINLINE float div_s(float a, float b) {
  return a / b;
}

template <>
DINLINE half div_s(half a, half b) {
  return __hdiv(a, b);
}

// scalar add functions
// for some reason when compiling with Pytorch, the + operator for half and
// bfloat is disabled so we call the intrinsics directly
DINLINE half& assign_add(half& a, half b) {
  a = __hadd(a, b);
  return a;
}
DINLINE float& assign_add(float& a, float b) { return a += b; }

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
template <>
DINLINE nv_bfloat16 cast_s(nv_bfloat16 val) {
  return val;
}

template <>
DINLINE float cast_s(nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <>
DINLINE nv_bfloat16 cast_s(float val) {
  return __float2bfloat16(val);
}

template <>
DINLINE char cast_s(nv_bfloat16 val) {
  return static_cast<char>(__bfloat162float(val));
}

template <>
DINLINE unsigned char cast_s(nv_bfloat16 val) {
  return static_cast<unsigned char>(__bfloat162float(val));
}

template <>
DINLINE nv_bfloat16 cast_s(char val) {
  return __float2bfloat16(static_cast<float>(val));
}

template <>
DINLINE nv_bfloat16 cast_s(unsigned char val) {
  return __float2bfloat16(static_cast<float>(val));
}

template <>
DINLINE nv_bfloat16 max_abs_s(nv_bfloat16 a, nv_bfloat16 b) {
  return __hmax(__habs(a), __habs(b));
}

template <>
DINLINE nv_bfloat16 max_s(nv_bfloat16 a, nv_bfloat16 b) {
  return __hmax(a, b);
}

template <>
DINLINE nv_bfloat16 min_s(nv_bfloat16 a, nv_bfloat16 b) {
  return __hmin(a, b);
}

template <>
DINLINE nv_bfloat16 add_s(nv_bfloat16 a, nv_bfloat16 b) {
  return __hadd(a, b);
}

template <>
DINLINE nv_bfloat16 sub_s(nv_bfloat16 a, nv_bfloat16 b) {
  return __hsub(a, b);
}

template <>
DINLINE nv_bfloat16 mul_s(nv_bfloat16 a, nv_bfloat16 b) {
  return __hmul(a, b);
}

template <>
DINLINE nv_bfloat16 div_s(nv_bfloat16 a, nv_bfloat16 b) {
  return __hdiv(a, b);
}

DINLINE nv_bfloat16& assign_add(nv_bfloat16& a, nv_bfloat16 b) {
  a = __hadd(a, b);
  return a;
}
#endif

template <typename T, int N>
DINLINE array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) {
    assign_add(a.data[i], b.data[i]);
  }
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) {
      out.data[i] = cast_s<T, float>(val.data[i]);
    }
    return out;
  }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++) {
      out.data[i] = cast_s<float, typename O::type>(val.data[i]);
    }
    return out;
  }
}

template <typename A, typename C>
DINLINE C symm_compress(A val) {
  using C_type = typename C::type;
  using A_type = typename A::type;
  constexpr int BITS = C::bits;
  constexpr int NUM_PACKED_VALS = 8 / BITS;
  constexpr unsigned char NUM_LEVELS = (1 << (BITS - 1)) - 1;
  A_type scale = cast_s<float, A_type>(0.0);
  C out;
#pragma unroll
  for (int i = 0; i < A::size; i++) {
    scale = max_abs_s(scale, val.data[i]);
  }
  const A_type num_levels = cast_s<unsigned char, A_type>(NUM_LEVELS);

  scale = div_s(scale, num_levels);

#pragma unroll
  for (int i = 0; i < A::size; i++) {
    int j = i / NUM_PACKED_VALS;
    char tmp = cast_s<A_type, char>(div_s(val.data[i], scale));
    out.data[j] |= (tmp << (i % NUM_PACKED_VALS * BITS)) & 0xff;
  }
  out.scale = cast_s<A_type, C_type>(scale);
  return out;
}

template <typename A, typename C>
DINLINE C asymm_compress(A val) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  C out;
  using A_type = typename A::type;
  using C_type = typename C::type;
  const int BITS = C::bits;
  constexpr int NUM_PACKED_VALS = 8 / BITS;
  constexpr unsigned char NUM_LEVELS = (1 << BITS) - 1;
  A_type zero = val.data[0];
  A_type scale = cast_s<float, A_type>(0.0);

#pragma unroll
  for (int i = 0; i < A::size; i++) {
    scale = max_s(scale, val.data[i]);
    zero = min_s(zero, val.data[i]);
  }

  const A_type num_levels = cast_s<unsigned char, A_type>(NUM_LEVELS);
  scale = sub_s(scale, zero);
  scale = div_s(scale, num_levels);

#pragma unroll
  for (int i = 0; i < A::size; i++) {
    int j = i / NUM_PACKED_VALS;
    unsigned char tmp =
        cast_s<A_type, unsigned char>(div_s(sub_s(val.data[i], zero), scale));
    out.data[j] |= (tmp << (i % NUM_PACKED_VALS * BITS)) & 0xff;
  }
  out.scale = cast_s<A_type, C_type>(scale);
  out.zero = cast_s<A_type, C_type>(zero);
  return out;
}

template <typename A, typename C, int BITS>
DINLINE C cast_compress(A val) {
  constexpr int NUM_PACKED_VALS = 8 / BITS;
  C out;

  for (int i = 0; i < A::size; i++) {
    int j = i / NUM_PACKED_VALS;
    char tmp = cast_s<typename A::type, char>(val.data[i]);
    // if (blockIdx.x * blockDim.x + threadIdx.x == 0)
    //   printf("%i ", static_cast<int>(tmp & 0xff));
    out.data[j] |= (tmp << (i % NUM_PACKED_VALS * BITS)) & 0xff;
  }
  // if (blockIdx.x * blockDim.x + threadIdx.x == 0)
  //   printf("\n");
  return out;
}

template <typename A, typename C, int BITS>
DINLINE C compress(A val) {
  if constexpr (std::is_same<A, C>::value) {
    return val;
  } else if constexpr (std::is_same<C, quantized_t<typename C::type, C::size,
                                                   BITS>>::value) {
    return symm_compress<A, C>(val);
  } else if constexpr (std::is_same<C,
                                    asymm_quantized_t<typename C::type, C::size,
                                                      BITS>>::value) {
    return asymm_compress<A, C>(val);
  } else {
    static_assert(std::is_same<C, array_t<char, C::size>>::value);
    return cast_compress<A, C, BITS>(val);
  }
}

template <typename P, typename C>
DINLINE P symm_decompress(C val) {
  using C_type = typename C::type;
  using P_type = typename P::type;
  const int BITS = C::bits;
  constexpr int NUM_PACKED_VALS = 8 / BITS;
  constexpr int NUM_LEVELS = (1 << (BITS - 1)) - 1;
  P out;
  auto scale = val.scale;
#pragma unroll
  for (int i = 0; i < P::size; i++) {
    char tmp =
        ((val.data[i / NUM_PACKED_VALS]) >> (i % NUM_PACKED_VALS * BITS)) &
        NUM_LEVELS;
    out.data[i] =
        cast_s<C_type, P_type>(mul_s(cast_s<char, C_type>(tmp), scale));
  }
  return out;
}

template <typename P, typename C>
DINLINE P asymm_decompress(C val) {
  using P_type = typename P::type;
  using C_type = typename C::type;
  const int BITS = C::bits;
  constexpr int NUM_PACKED_VALS = 8 / BITS;
  constexpr int NUM_LEVELS = (1 << BITS) - 1;
  P out;
  C_type scale = val.scale;
  C_type zero = val.zero;

#pragma unroll
  for (int i = 0; i < P::size; i++) {
    unsigned char tmp =
        ((val.data[i / NUM_PACKED_VALS]) >> (i % NUM_PACKED_VALS * BITS)) &
        NUM_LEVELS;
    out.data[i] = cast_s<C_type, P_type>(
        add_s(mul_s(cast_s<unsigned char, C_type>(tmp), scale), zero));
  }
  return out;
}

template <typename P, typename C, int BITS>
DINLINE P cast_decompress(C val) {
  constexpr int NUM_PACKED_VALS = 8 / BITS;
  constexpr int NUM_LEVELS = (1 << (BITS - 1)) - 1;
  P out;
  for (int i = 0; i < P::size; i++) {
    char tmp =
        ((val.data[i / NUM_PACKED_VALS]) >> (i % NUM_PACKED_VALS * BITS)) &
        NUM_LEVELS;
    // if (blockIdx.x * blockDim.x + threadIdx.x == 0)
    //   printf("%i ", tmp);
    out.data[i] = cast_s<char, typename P::type>(tmp);
  }
  // if (blockIdx.x * blockDim.x + threadIdx.x == 0)
  //   printf("\n");
  return out;
}

template <typename P, typename C, int BITS>
DINLINE P decompress(C val) {
  using T = typename P::type;
  using A_type = float;
  if constexpr (std::is_same<P, C>::value) {
    return val;
  } else if constexpr (std::is_same<
                           quantized_t<typename C::type, C::size, BITS>,
                           C>::value) {
    return symm_decompress<P, C>(val);
  } else if constexpr (std::is_same<
                           asymm_quantized_t<typename C::type, C::size, BITS>,
                           C>::value) {
    return asymm_decompress<P, C>(val);
  } else {
    static_assert(std::is_same<C, array_t<char, C::size>>::value);
    return cast_decompress<P, C, BITS>(val);
  }
}

#if !defined(USE_ROCM)

static DINLINE void st_flag_release(FlagType* flag_addr, FlagType flag) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag),
               "l"(flag_addr));
  #else
  asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag),
               "l"(flag_addr));
  #endif
}

static DINLINE FlagType ld_flag_acquire(FlagType* flag_addr) {
  FlagType flag;
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
  #else
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;"
               : "=r"(flag)
               : "l"(flag_addr));
  #endif
  return flag;
}

static DINLINE void st_flag_volatile(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_volatile(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.volatile.global.u32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
  return flag;
}
#endif

// This function is meant to be used as the first synchronization in the all
// reduce kernel. Thus, it doesn't need to make any visibility guarantees for
// prior memory accesses. Note: volatile writes will not be reordered against
// other volatile writes.
template <int ngpus>
DINLINE void start_sync(const RankSignals& sg, Signal* self_sg, int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
#if !defined(USE_ROCM)
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->start[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->start[blockIdx.x][threadIdx.x];
    // Write the expected counter value to peer and wait for correct value
    // from peer.
    st_flag_volatile(peer_counter_ptr, flag);
    while (ld_flag_volatile(self_counter_ptr) != flag);
#else
    // simultaneously write to the corresponding flag of all ranks.
    // Latency = 1 p2p write
    __scoped_atomic_store_n(&sg.signals[threadIdx.x]->start[blockIdx.x][rank],
                            flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    // wait until we got true from all ranks
    while (__scoped_atomic_load_n(&self_sg->start[blockIdx.x][threadIdx.x],
                                  __ATOMIC_RELAXED,
                                  __MEMORY_SCOPE_DEVICE) < flag);
#endif
  }
  __syncthreads();
  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

// This function is meant to be used as the second or the final
// synchronization barrier in the all reduce kernel. If it's the final
// synchronization barrier, we don't need to make any visibility guarantees
// for prior memory accesses.
template <int ngpus, bool final_sync = false>
DINLINE void end_sync(const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads();
  // eliminate the case that prior writes are not visible after signals become
  // visible. Note that I did not managed to make this happen through a lot of
  // testing. Might be the case that hardware provides stronger guarantee than
  // the memory model.
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
#if !defined(USE_ROCM)
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->end[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->end[blockIdx.x][threadIdx.x];
    // Write the expected counter value to peer and wait for correct value from
    // peer.
    if constexpr (!final_sync) {
      st_flag_release(peer_counter_ptr, flag);
      while (ld_flag_acquire(self_counter_ptr) != flag);
    } else {
      st_flag_volatile(peer_counter_ptr, flag);
      while (ld_flag_volatile(self_counter_ptr) != flag);
    }
  }
  if constexpr (!final_sync) __syncthreads();
#else
  if (threadIdx.x < ngpus) {
    // simultaneously write to the corresponding flag of all ranks.
    // Latency = 1 p2p write
    __scoped_atomic_store_n(&sg.signals[threadIdx.x]->end[blockIdx.x][rank],
                            flag,
                            final_sync ? __ATOMIC_RELAXED : __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
    // wait until we got true from all ranks
    while (
        __scoped_atomic_load_n(&self_sg->end[blockIdx.x][threadIdx.x],
                               final_sync ? __ATOMIC_RELAXED : __ATOMIC_ACQUIRE,
                               __MEMORY_SCOPE_DEVICE) < flag);
  }
  __syncthreads();
#endif
  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <typename P, int ngpus, typename A, typename C, int BITS = 8>
DINLINE C packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  if constexpr (!std::is_same<C, P>::value) {
    return compress<A, C, BITS>(tmp);
  } else {
    return downcast<C>(tmp);
  }
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(1024, 1)
    cross_device_reduce_1stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // note: we don't reorder the address so the accumulation order is the same
  // for all ranks, ensuring bitwise identical results
  auto dp = *_dp;
  start_sync<ngpus>(sg, self_sg, rank);
  // do the actual reduction
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] =
        packed_reduce<P, ngpus, A, P>((const P**)&dp.ptrs[0], idx);
  }
  end_sync<ngpus, true>(sg, self_sg, rank);
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

template <typename T, int ngpus,
          CA_2STAGE_COMPRESSION compression_type = CA_2STAGE_COMPRESSION::NONE>
__global__ void __launch_bounds__(1024, 1)
    cross_device_reduce_2stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size,
                               int synt_compression_factor = 1) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int BITS = (compression_type == CA_2STAGE_COMPRESSION::ASYMM_4BIT ||
                        compression_type == CA_2STAGE_COMPRESSION::SYMM_4BIT ||
                        compression_type == CA_2STAGE_COMPRESSION::CAST_4BIT)
                           ? 4
                           : 8;
  using C = std::conditional_t<
      (compression_type == CA_2STAGE_COMPRESSION::NONE), P,
      std::conditional_t<
          (compression_type == CA_2STAGE_COMPRESSION::SYMM_4BIT ||
           compression_type == CA_2STAGE_COMPRESSION::SYMM_8BIT),
          quantized_t<typename P::type, P::size, BITS>,
          std::conditional_t<
              (compression_type == CA_2STAGE_COMPRESSION::ASYMM_4BIT ||
               compression_type == CA_2STAGE_COMPRESSION::ASYMM_8BIT),
              asymm_quantized_t<typename P::type, P::size, BITS>,
              array_t<char, P::size / (8 / BITS)>>>>;

  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  C* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<C>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  start_sync<ngpus>(sg, self_sg, rank);

  // stage 1: reduce scatter
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A, C, BITS>(ptrs, idx);
  }
  // multi_gpu_barrier<ngpus, false, true>(sg, self_sg, rank);
  end_sync<ngpus>(sg, self_sg, rank);

  // stage 2: allgather. Note: it's important to match the tid between
  // the two stages, because visibility across devices is only guaranteed
  // between threads that have the same tid. If thread i computes the sum of
  // start + i in the first stage, then thread i also gathers start + i from
  // all ranks.
  largest_part /= synt_compression_factor;
  C tmp;
  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        tmp = tmps[i][idx];
        ((P*)result)[dst_idx] = decompress<P, C, BITS>(tmp);
      }
    }
  }
}

using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;
static_assert(sizeof(IPC_KEY) == sizeof(cudaIpcMemHandle_t));
static_assert(alignof(IPC_KEY) == alignof(cudaIpcMemHandle_t));

class CustomAllreduce {
 public:
  int rank_;
  int world_size_;
  bool full_nvlink_;

  RankSignals sg_;
  // Stores an map from a pointer to its peer pointters from all ranks.
  std::unordered_map<void*, RankData*> buffers_;
  Signal* self_sg_;

  // Stores rank data from all ranks. This is mainly for cuda graph purposes.
  // For cuda graph to work, all kernel arguments must be fixed during graph
  // capture time. However, the peer pointers are not known during graph
  // capture time. Therefore, during capture, we increment the rank data
  // pointer and use that as the argument to the kernel. The kernel arguments
  // are stored in graph_unreg_buffers_. The actual peer pointers will be
  // filled in at the memory pointed to by the pointers in
  // graph_unreg_buffers_ when the IPC handles are exchanged between ranks.
  //
  // The overall process looks like this:
  // 1. Graph capture.
  // 2. Each rank obtains the IPC handles for each addresses used during cuda
  // graph capture using get_graph_buffer_ipc_meta.
  // 3. (In Python) all gather the IPC handles.
  // 4. Obtain the peer pointers by opening the IPC handles, and store them in
  // the rank data array at corresponding positions.
  RankData *d_rank_data_base_, *d_rank_data_end_;
  std::vector<void*> graph_unreg_buffers_;
  // a map from IPC handles to opened IPC pointers
  std::map<IPC_KEY, char*> ipc_handles_;

  /**
   * Signals are an array of ipc-enabled buffers from all ranks.
   * For each of the buffer, the layout is as follows:
   * | -- sizeof(Signal) -- | ------ a few MB ----- |
   * The first section is for allreduce synchronization, and the second
   * section is for storing the intermediate results required by some
   * allreduce algos.
   *
   * Note: this class does not own any device memory. Any required buffers
   * are passed in from the constructor.
   */
  CustomAllreduce(Signal** signals, void* rank_data, size_t rank_data_sz,
                  int rank, int world_size, bool full_nvlink = true)
      : rank_(rank),
        world_size_(world_size),
        full_nvlink_(full_nvlink),
        self_sg_(signals[rank]),
        d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
        d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    for (int i = 0; i < world_size_; i++) {
      sg_.signals[i] = signals[i];
    }
  }

  char* open_ipc_handle(const void* ipc_handle) {
    auto [it, new_handle] =
        ipc_handles_.insert({*((IPC_KEY*)ipc_handle), nullptr});
    if (new_handle) {
      char* ipc_ptr;
      CUDACHECK(cudaIpcOpenMemHandle((void**)&ipc_ptr,
                                     *((const cudaIpcMemHandle_t*)ipc_handle),
                                     cudaIpcMemLazyEnablePeerAccess));
      it->second = ipc_ptr;
    }
    return it->second;
  }

  std::pair<std::string, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
    auto num_buffers = graph_unreg_buffers_.size();
    auto handle_sz = sizeof(cudaIpcMemHandle_t);
    std::string handles(handle_sz * num_buffers, static_cast<char>(0));
    std::vector<int64_t> offsets(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto ptr = graph_unreg_buffers_[i];
      void* base_ptr;
      // note: must share the base address of each allocation, or we get wrong
      // address
      if (cuPointerGetAttribute(&base_ptr,
#if defined(USE_ROCM)
                                HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
#else
                                CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
#endif
                                (CUdeviceptr)ptr) != CUDA_SUCCESS)
        throw std::runtime_error("failed to get pointer attr");
      CUDACHECK(cudaIpcGetMemHandle(
          (cudaIpcMemHandle_t*)&handles[i * handle_sz], base_ptr));
      offsets[i] = ((char*)ptr) - ((char*)base_ptr);
    }
    return std::make_pair(handles, offsets);
  }

  void check_rank_data_capacity(size_t num = 1) {
    if (d_rank_data_base_ + num > d_rank_data_end_)
      throw std::runtime_error(
          "Rank data buffer is overflowed by " +
          std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
  }

  /**
   * Register already-shared IPC pointers.
   */
  void register_buffer(void** ptrs) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) {
      data.ptrs[i] = ptrs[i];
    }
    auto d_data = d_rank_data_base_++;
    CUDACHECK(
        cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    buffers_[ptrs[rank_]] = d_data;
  }

  // Note: when registering graph buffers, we intentionally choose to not
  // deduplicate the addresses. That means if the allocator reuses some
  // addresses, they will be registered again. This is to account for the
  // remote possibility of different allocation patterns between ranks. For
  // example, rank 1 may get the same input address for the second allreduce,
  // but rank 2 got a different address. IPC handles have internal reference
  // counting mechanism so overhead should be small.
  void register_graph_buffers(
      const std::vector<std::string>& handles,
      const std::vector<std::vector<int64_t>>& offsets) {
    auto num_buffers = graph_unreg_buffers_.size();
    check_rank_data_capacity(num_buffers);
    std::vector<RankData> rank_data(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto self_ptr = graph_unreg_buffers_[i];
      auto& rd = rank_data[i];
      for (int j = 0; j < world_size_; j++) {
        if (j != rank_) {
          char* handle =
              open_ipc_handle(&handles[j][i * sizeof(cudaIpcMemHandle_t)]);
          handle += offsets[j][i];
          rd.ptrs[j] = handle;
        } else {
          rd.ptrs[j] = self_ptr;
        }
      }
    }
    CUDACHECK(cudaMemcpy(d_rank_data_base_, rank_data.data(),
                         sizeof(RankData) * num_buffers,
                         cudaMemcpyHostToDevice));
    d_rank_data_base_ += num_buffers;
    graph_unreg_buffers_.clear();
  }

  /**
   * Performs allreduce, assuming input has already been registered.
   *
   * Block and grid default configs are results after careful grid search.
   * Using 36 blocks give the best or close to the best runtime on the devices
   * I tried: A100, A10, A30, T4, V100. You'll notice that NCCL kernels also
   * only take a small amount of SMs. Not quite sure the underlying reason,
   * but my guess is that too many SMs will cause contention on NVLink bus.
   */
  template <typename T>
  void allreduce(cudaStream_t stream, T* input, T* output, int size,
#if !defined(USE_ROCM)
                 int threads = 512, int block_limit = 36)
#else
                 int threads = 512, int block_limit = 16)
#endif
  {
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error(
          "custom allreduce currently requires input length to be multiple "
          "of " +
          std::to_string(d));
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " +
                               std::to_string(kMaxBlocks) + ". Got " +
                               std::to_string(block_limit));

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CUDACHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " +
            std::to_string(reinterpret_cast<uint64_t>(input)) +
            " is not registered!");
      ptrs = it->second;
    }

    CA_2STAGE_COMPRESSION compress_type =
        std::getenv("VLLM_CA_2STAGE_COMPRESS_TYPE")
            ? static_cast<CA_2STAGE_COMPRESSION>(
                  std::stoi(std::getenv("VLLM_CA_2STAGE_COMPRESS_TYPE")))
            : CA_2STAGE_COMPRESSION::NONE;
    int synt_compression_factor =
        std::getenv("VLLM_CA_EXP_SYNT_COMPRESSION_FACTOR")
            ? std::stoi(std::getenv("VLLM_CA_EXP_SYNT_COMPRESSION_FACTOR"))
            : 1;
    size /= d;
    bool do_compress = compress_type != CA_2STAGE_COMPRESSION::NONE or
                       synt_compression_factor > 1;

    threads = std::getenv("VLLM_CA_THREADS")
                  ? std::stoi(std::getenv("VLLM_CA_THREADS"))
                  : threads;
    auto bytes = size * sizeof(typename packed_t<T>::P);
    int blocks = std::min(block_limit, (size + threads - 1) / threads);
#define KL(ngpus, name)                                                       \
  name<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, \
                                                 rank_, size);
#define REDUCE_CASE(ngpus)                                                    \
  case ngpus: {                                                               \
    if (world_size_ == 2 and !do_compress) {                                  \
      KL(ngpus, cross_device_reduce_1stage);                                  \
    } else if (full_nvlink_) {                                                \
      if (!do_compress && ((world_size_ <= 4 && bytes < 512 * 1024) ||        \
                           (world_size_ <= 8 && bytes < 256 * 1024))) {       \
        KL(ngpus, cross_device_reduce_1stage);                                \
      } else {                                                                \
        switch (compress_type) {                                              \
          case CA_2STAGE_COMPRESSION::NONE: {                                 \
            cross_device_reduce_2stage<T, ngpus, CA_2STAGE_COMPRESSION::NONE> \
                <<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, \
                                                 rank_, size,                 \
                                                 synt_compression_factor);    \
            break;                                                            \
          }                                                                   \
          case CA_2STAGE_COMPRESSION::SYMM_4BIT: {                            \
            cross_device_reduce_2stage<T, ngpus,                              \
                                       CA_2STAGE_COMPRESSION::SYMM_4BIT>      \
                <<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, \
                                                 rank_, size,                 \
                                                 synt_compression_factor);    \
            break;                                                            \
          }                                                                   \
          case CA_2STAGE_COMPRESSION::SYMM_8BIT: {                            \
            cross_device_reduce_2stage<T, ngpus,                              \
                                       CA_2STAGE_COMPRESSION::SYMM_8BIT>      \
                <<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, \
                                                 rank_, size,                 \
                                                 synt_compression_factor);    \
            break;                                                            \
          }                                                                   \
          case CA_2STAGE_COMPRESSION::ASYMM_4BIT: {                           \
            cross_device_reduce_2stage<T, ngpus,                              \
                                       CA_2STAGE_COMPRESSION::ASYMM_4BIT>     \
                <<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, \
                                                 rank_, size,                 \
                                                 synt_compression_factor);    \
            break;                                                            \
          }                                                                   \
          case CA_2STAGE_COMPRESSION::ASYMM_8BIT: {                           \
            cross_device_reduce_2stage<T, ngpus,                              \
                                       CA_2STAGE_COMPRESSION::ASYMM_8BIT>     \
                <<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, \
                                                 rank_, size,                 \
                                                 synt_compression_factor);    \
            break;                                                            \
          }                                                                   \
          case CA_2STAGE_COMPRESSION::CAST_4BIT: {                            \
            cross_device_reduce_2stage<T, ngpus,                              \
                                       CA_2STAGE_COMPRESSION::CAST_4BIT>      \
                <<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, \
                                                 rank_, size,                 \
                                                 synt_compression_factor);    \
            break;                                                            \
          }                                                                   \
          case CA_2STAGE_COMPRESSION::CAST_8BIT: {                            \
            cross_device_reduce_2stage<T, ngpus,                              \
                                       CA_2STAGE_COMPRESSION::CAST_8BIT>      \
                <<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, \
                                                 rank_, size,                 \
                                                 synt_compression_factor);    \
            break;                                                            \
          }                                                                   \
        }                                                                     \
      }                                                                       \
    }                                                                         \
    break;                                                                    \
  }

    switch (world_size_) {
      REDUCE_CASE(2)
      REDUCE_CASE(4)
      REDUCE_CASE(6)
      REDUCE_CASE(8)
      default:
        throw std::runtime_error(
            "custom allreduce only supports num gpus in (2,4,6,8). Actual "
            "num "
            "gpus = " +
            std::to_string(world_size_));
    }
#undef REDUCE_CASE
#undef KL
  }

  ~CustomAllreduce() {
    for (auto [_, ptr] : ipc_handles_) {
      CUDACHECK(cudaIpcCloseMemHandle(ptr));
    }
  }
};

/**
 * To inspect PTX/SASS, copy paste this header file to compiler explorer and
 add a template instantiation:
 * template void vllm::CustomAllreduce::allreduce<half>(cudaStream_t, half *,
 half *, int, int, int);
*/
}  // namespace vllm