//===- CRunnerUtils.cpp - Utils for MLIR execution ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements basic functions to manipulate structured MLIR types at
// runtime. Entities in this file are meant to be retargetable, including on
// targets without a C++ runtime, and must be kept C compatible.
//
//===----------------------------------------------------------------------===//

#include "XsmmRunnerUtils.h"
#include "libxsmm.h" // NOLINT [build/include_subdir]

extern "C" void _mlir_ciface_xsmm_matmul_invoke(const libxsmm_datatype dtype,
                                                int64_t funcAddr,
                                                UnrankedMemRefType<char> *A,
                                                UnrankedMemRefType<char> *B,
                                                UnrankedMemRefType<char> *C) {
  DynamicMemRefType<char> matrixA = DynamicMemRefType<char>(*A);
  DynamicMemRefType<char> matrixB = DynamicMemRefType<char>(*B);
  DynamicMemRefType<char> matrixC = DynamicMemRefType<char>(*C);

  //   std::cout << "matrix A: \n";
  //   printMemRefMetaData(std::cout, DynamicMemRefType<char>(*A));
  //    std::cout << "\n";
  //   std::cout << "matrix B: \n";
  //   printMemRefMetaData(std::cout, DynamicMemRefType<char>(*B));
  //   std::cout << "\n";
  //   std::cout << "matrix C: \n";
  //   printMemRefMetaData(std::cout, DynamicMemRefType<char>(*C));
  //   std::cout << "\n";
  //   std::cout << "funcAddr: " << funcAddr << "\n";

  //    int64_t M = matrixC.sizes[0];
  //    int64_t N = matrixC.sizes[1];
  //    int64_t K = matrixA.sizes[1];
  //
  //    for (int i = 0; i < M; i++) {
  //      for (int j = 0; j < N; j++) {
  //        for (int k = 0; k < K; k++) {
  //          float *curr_addr_a = i * matrixA.strides[0] + addr_a + k;
  //          float *curr_addr_b = k * matrixB.strides[0] + addr_b + j;
  //          float *curr_addr_c = i * matrixC.strides[0] + addr_c + j;
  //          *curr_addr_c += (*curr_addr_a) * (*curr_addr_b);
  //        }
  //      }
  //    }

  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;

  if (dtype == LIBXSMM_DATATYPE_F32) {
    float *addr_a = (float *)matrixA.data + matrixA.offset;
    float *addr_b = (float *)matrixB.data + matrixB.offset;
    float *addr_c = (float *)matrixC.data + matrixC.offset;
    // LIBXSMM col-major change A with B.
    gemm_param.a.primary = (void *)addr_b;
    gemm_param.b.primary = (void *)addr_a;
    gemm_param.c.primary = (void *)addr_c;

  } else if (dtype == LIBXSMM_DATATYPE_BF16) {
    bf16 *addr_a = (bf16 *)matrixA.data + matrixA.offset;
    bf16 *addr_b = (bf16 *)matrixB.data + matrixB.offset;
    bf16 *addr_c = (bf16 *)matrixC.data + matrixC.offset;
    //  LIBXSMM col-major change A with B.
    gemm_param.a.primary = (void *)addr_b;
    gemm_param.b.primary = (void *)addr_a;
    gemm_param.c.primary = (void *)addr_c;
  }
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(funcAddr);
  sgemm.gemm(&gemm_param);
}

extern "C" int64_t
_mlir_ciface_xsmm_matmul_dispatch(const libxsmm_datatype dtype, int64_t m,
                                  int64_t n, int64_t k, int64_t lda,
                                  int64_t ldb, int64_t ldc) {
  // std::cout << "lda: " << lda << "\n";
  // std::cout << "ldb: " << ldb << "\n";
  // std::cout << "ldc: " << ldc << "\n";

  // std::cout << "m: " << m << "\n";
  // std::cout << "n: " << n << "\n";
  // std::cout << "k: " << k << "\n";

  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags = 0;

  // See:
  // https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
  // LIBXSMM col-major change m with n.
  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb;
  l_shape.ldb = lda;
  l_shape.ldc = ldc;
  l_shape.a_in_type = dtype;
  l_shape.b_in_type = dtype;
  l_shape.out_type = dtype;
  l_shape.comp_type = dtype;

  auto sgemm = libxsmm_dispatch_gemm_v2(l_shape, l_flags, l_prefetch_flags);
  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" int64_t
_mlir_ciface_xsmm_unary_dispatch(const libxsmm_datatype dtype, int64_t m,
                                 int64_t n, int64_t ldi, int64_t ldo,
                                 int64_t type, int64_t bcast_type) {

  // std::cout << "ldi: " << ldi << "\n";
  // std::cout << "ldo: " << ldo << "\n";
  // std::cout << "m: " << m << "\n";
  // std::cout << "n: " << n << "\n";
  // std::cout << "type: " << type << "\n";
  // std::cout << "bcast_type: " << bcast_type << "\n";

  libxsmm_meltw_unary_flags unary_flags =
      static_cast<libxsmm_meltw_unary_flags>(bcast_type);

  libxsmm_meltw_unary_shape unary_shape;

  // Row major to col major swap m with n.
  unary_shape.m = static_cast<libxsmm_blasint>(n);
  unary_shape.n = static_cast<libxsmm_blasint>(m);
  unary_shape.in0_type = dtype;
  unary_shape.comp_type = dtype;
  unary_shape.out_type = dtype;
  unary_shape.ldi = static_cast<libxsmm_blasint>(ldi);
  unary_shape.ldo = static_cast<libxsmm_blasint>(ldo);

  libxsmm_meltwfunction_unary kernel = libxsmm_dispatch_meltw_unary_v2(
      static_cast<libxsmm_meltw_unary_type>(type), unary_shape,
      static_cast<libxsmm_bitfield>(unary_flags));

  return reinterpret_cast<int64_t>(kernel);
}

extern "C" int64_t _mlir_ciface_xsmm_binary_dispatch(
    const libxsmm_datatype dtype, int64_t m, int64_t n, int64_t ldiLhs,
    int64_t ldiRhs, int64_t ldo, int64_t type, int64_t bcast_type) {

  libxsmm_meltw_binary_flags binary_flags =
      static_cast<libxsmm_meltw_binary_flags>(bcast_type);

  libxsmm_meltw_binary_shape binary_shape;

  // Row major to col major swap m with n.
  binary_shape.m = static_cast<libxsmm_blasint>(n);
  binary_shape.n = static_cast<libxsmm_blasint>(m);
  binary_shape.in0_type = dtype;
  binary_shape.in1_type = dtype;
  binary_shape.comp_type = dtype;
  binary_shape.out_type = dtype;
  binary_shape.ldi = static_cast<libxsmm_blasint>(ldiLhs);
  binary_shape.ldi2 = static_cast<libxsmm_blasint>(ldiRhs);
  binary_shape.ldo = static_cast<libxsmm_blasint>(ldo);

  libxsmm_meltwfunction_binary kernel = libxsmm_dispatch_meltw_binary_v2(
      static_cast<libxsmm_meltw_binary_type>(type), binary_shape,
      static_cast<libxsmm_bitfield>(binary_flags));

  return reinterpret_cast<int64_t>(kernel);
}

extern "C" void
_mlir_ciface_xsmm_unary_invoke(const libxsmm_datatype dType, int64_t addr,
                               UnrankedMemRefType<char> *input,
                               UnrankedMemRefType<char> *output) {
  // std::cout << "tensor input: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<char>(*input));
  // std::cout << "tensor output: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<char>(*output));
  DynamicMemRefType<char> tensorA = DynamicMemRefType<char>(*input);
  DynamicMemRefType<char> tensorB = DynamicMemRefType<char>(*output);

  libxsmm_meltwfunction_unary kernel =
      reinterpret_cast<libxsmm_meltwfunction_unary>(addr);
  libxsmm_meltw_unary_param param;
  if (dType == LIBXSMM_DATATYPE_F32) {
    float *addr_a = (float *)tensorA.data + tensorA.offset;
    float *addr_b = (float *)tensorB.data + tensorB.offset;

    param.in.primary = (void *)addr_a;
    param.out.primary = (void *)addr_b;
  } else if (dType == LIBXSMM_DATATYPE_BF16) {

    bf16 *addr_a = (bf16 *)tensorA.data + tensorA.offset;
    bf16 *addr_b = (bf16 *)tensorB.data + tensorB.offset;

    param.in.primary = (void *)addr_a;
    param.out.primary = (void *)addr_b;
  }
  kernel(&param);
}

extern "C" void _mlir_ciface_xsmm_binary_invoke(const libxsmm_datatype dType,
                                                int64_t addr,
                                                UnrankedMemRefType<char> *lhs,
                                                UnrankedMemRefType<char> *rhs) {

  DynamicMemRefType<char> tensorLhs = DynamicMemRefType<char>(*lhs);
  DynamicMemRefType<char> tensorRhs = DynamicMemRefType<char>(*rhs);
  libxsmm_meltwfunction_binary kernel =
      reinterpret_cast<libxsmm_meltwfunction_binary>(addr);
  libxsmm_meltw_binary_param param;

  if (dType == LIBXSMM_DATATYPE_F32) {
    float *addr_tensor_lhs = (float *)tensorLhs.data + tensorLhs.offset;
    float *addr_tensor_rhs = (float *)tensorRhs.data + tensorRhs.offset;
    param.in0.primary = (void *)addr_tensor_lhs;
    param.in1.primary = (void *)addr_tensor_rhs;
    param.out.primary = (void *)addr_tensor_rhs;

  } else if (dType == LIBXSMM_DATATYPE_BF16) {
    bf16 *addr_tensor_lhs = (bf16 *)tensorLhs.data + tensorLhs.offset;
    bf16 *addr_tensor_rhs = (bf16 *)tensorRhs.data + tensorRhs.offset;
    param.in0.primary = (void *)addr_tensor_lhs;
    param.in1.primary = (void *)addr_tensor_rhs;
    param.out.primary = (void *)addr_tensor_rhs;
  }
  kernel(&param);
}

extern "C" void
_mlir_ciface_xsmm_unary_scalar_invoke(const libxsmm_datatype dType,
                                      int64_t addr, float input,
                                      UnrankedMemRefType<char> *output) {
  DynamicMemRefType<char> tensorB = DynamicMemRefType<char>(*output);
  libxsmm_meltwfunction_unary kernel =
      reinterpret_cast<libxsmm_meltwfunction_unary>(addr);
  libxsmm_meltw_unary_param param;

  param.in.primary = (void *)&input;
  param.out.primary = (void *)tensorB.data;
  kernel(&param);
}

LIBXSMM_INLINE void matrix_copy_NC_to_NCNC(float *src, float *dst, int T, int N,
                                           int C, int bn, int bc) {
  int t, n1, n2, c1, c2;
  int nBlocks = N / bn;
  int cBlocks = C / bc;
  LIBXSMM_VLA_DECL(3, float, real_src, src, N, C);
  LIBXSMM_VLA_DECL(5, float, real_dst, dst, nBlocks, cBlocks, bn, bc);

  for (t = 0; t < T; t++) {
    for (n1 = 0; n1 < nBlocks; n1++) {
      for (c1 = 0; c1 < cBlocks; c1++) {
        for (n2 = 0; n2 < bn; n2++) {
          for (c2 = 0; c2 < bc; c2++) {
            LIBXSMM_VLA_ACCESS(5, real_dst, t, n1, c1, n2, c2, nBlocks, cBlocks,
                               bn, bc) =
                LIBXSMM_VLA_ACCESS(3, real_src, t, n1 * bn + n2, c1 * bc + c2,
                                   N, C);
          }
        }
      }
    }
  }
}

extern "C" void _mlir_ciface_matrix_copy_NC_to_NCNC(
    UnrankedMemRefType<float> *input, UnrankedMemRefType<float> *output,
    int64_t N, int64_t C, int64_t n, int64_t c) {
  // std::cout << "\n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<float>(*input));
  // std::cout << "\n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<float>(*output));
  // std::cout << "\n";
  DynamicMemRefType<float> tensorInput = DynamicMemRefType<float>(*input);
  DynamicMemRefType<float> tensorOutput = DynamicMemRefType<float>(*output);

  float *addr_input = tensorInput.data + tensorInput.offset;
  float *addr_output = tensorOutput.data + tensorOutput.offset;

  matrix_copy_NC_to_NCNC(addr_input, addr_output, 1, C, N, c, n);
}

extern "C" void _mlir_ciface_xsmm_brgemm_invoke(const libxsmm_datatype dType,
                                                int64_t addr,
                                                UnrankedMemRefType<char> *A,
                                                UnrankedMemRefType<char> *B,
                                                UnrankedMemRefType<char> *C,
                                                int64_t numBatches) {
  // std::cout << "numBatch: " << numBatches << "\n";
  // std::cout << "\n A: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<char>(*A));
  // std::cout << "\n B: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<char>(*B));
  // std::cout << "\n C: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<char>(*C));
  // std::cout << "\n";

  DynamicMemRefType<char> tensorA = DynamicMemRefType<char>(*A);
  DynamicMemRefType<char> tensorB = DynamicMemRefType<char>(*B);
  DynamicMemRefType<char> tensorC = DynamicMemRefType<char>(*C);

  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(addr);
  unsigned long long numBatchesVar = numBatches;
  if (dType == LIBXSMM_DATATYPE_F32) {
    float *addr_tensorA = (float *)tensorA.data + tensorA.offset;
    float *addr_tensorB = (float *)tensorB.data + tensorB.offset;
    float *addr_tensorC = (float *)tensorC.data + tensorC.offset;
    gemm_param.a.primary = (void *)addr_tensorB;
    gemm_param.b.primary = (void *)addr_tensorA;
    gemm_param.c.primary = (void *)addr_tensorC;
  } else if (dType == LIBXSMM_DATATYPE_BF16) {
    bf16 *addr_tensorA = (bf16 *)tensorA.data + tensorA.offset;
    bf16 *addr_tensorB = (bf16 *)tensorB.data + tensorB.offset;
    bf16 *addr_tensorC = (bf16 *)tensorC.data + tensorC.offset;
    gemm_param.a.primary = (void *)addr_tensorB;
    gemm_param.b.primary = (void *)addr_tensorA;
    gemm_param.c.primary = (void *)addr_tensorC;
  }
  gemm_param.op.tertiary = (void *)&numBatchesVar;
  sgemm.gemm(&gemm_param);
}

extern "C" int64_t
_mlir_ciface_xsmm_brgemm_dispatch(const libxsmm_datatype dtype, int64_t m,
                                  int64_t n, int64_t k, int64_t lda,
                                  int64_t ldb, int64_t ldc) {
  // std::cout << "lda: " << lda << "\n";
  // std::cout << "lbd: " << ldb << "\n";
  // std::cout << "ldc: " << ldc << "\n";
  // std::cout << "m: " << m << "\n";
  // std::cout << "n: " << n << "\n";
  // std::cout << "k: " << k << "\n";

  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;
  // TODO: move stride computation to dispatch
  // operation as in: https://github.com/plaidml/plaidml/pull/1983
  auto typeSize = dtype == LIBXSMM_DATATYPE_F32 ? sizeof(float) : sizeof(bf16);
  libxsmm_blasint stride_a = lda * m * typeSize;
  libxsmm_blasint stride_b = ldb * k * typeSize;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags = 0;
  libxsmm_gemm_batch_reduce_config l_brconfig;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = dtype;
  l_shape.b_in_type = dtype;
  l_shape.out_type = dtype;
  l_shape.comp_type = dtype;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
  l_brconfig.br_stride_a_hint = stride_b;
  l_brconfig.br_stride_b_hint = stride_a;
  l_brconfig.br_unroll_hint = 0;

  auto sgemm = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch_flags,
                                          l_brconfig);

  return reinterpret_cast<int64_t>(sgemm);
}

//----------------------------------------------------------------------------//
// BRGEMM connection on the IREE side.
//----------------------------------------------------------------------------//

extern "C" int iree_xsmm_brgemm_dispatch_f32(void *context, void *params,
                                             void *reserved) {
  typedef struct {
    int64_t res;
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
  } xsmm_brgemm_dispatch_f32_t;
  xsmm_brgemm_dispatch_f32_t *p = (xsmm_brgemm_dispatch_f32_t *)params;
  p->res = _mlir_ciface_xsmm_brgemm_dispatch(LIBXSMM_DATATYPE_F32, p->m, p->n,
                                             p->k, p->lda, p->ldb, p->ldc);
  return 0;
}

extern "C" int iree_xsmm_matmul_dispatch_f32(void *context, void *params,
                                             void *reserved) {
  typedef struct {
    int64_t res;
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
  } xsmm_matmul_dispatch_f32_t;
  xsmm_matmul_dispatch_f32_t *p = (xsmm_matmul_dispatch_f32_t *)params;
  p->res = _mlir_ciface_xsmm_matmul_dispatch(LIBXSMM_DATATYPE_F32, p->m, p->n,
                                             p->k, p->lda, p->ldb, p->ldc);
  return 0;
}

extern "C" int iree_xsmm_unary_dispatch(void *context, void *params,
                                        void *reserved) {
  typedef struct {
    int64_t res;
    int64_t m;
    int64_t n;
    int64_t ldi;
    int64_t ldo;
    int64_t type;
    int64_t bcast_type;
  } xsmm_unary_dispatch;
  xsmm_unary_dispatch *p = (xsmm_unary_dispatch *)params;
  p->res = _mlir_ciface_xsmm_unary_dispatch(
      LIBXSMM_DATATYPE_F32, p->m, p->n, p->ldi, p->ldo, p->type, p->bcast_type);
  return 0;
}

// TODO: struct slicing. BRGEMM struct is the same as the GEMM one plus the
// batch parameter.
extern "C" int iree_xsmm_brgemm_invoke_f32(void *context, void *params,
                                           void *reserved) {
  typedef struct {
    int64_t addr;
    float *pA;
    int64_t offA;
    float *pB;
    int64_t offB;
    float *pC;
    int64_t offC;
    int64_t numBatches;
  } xsmm_brgemm_invoke_f32_t;
  xsmm_brgemm_invoke_f32_t *p = (xsmm_brgemm_invoke_f32_t *)params;

  float *addr_tensorA = p->pA + p->offA;
  float *addr_tensorB = p->pB + p->offB;
  float *addr_tensorC = p->pC + p->offC;

  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(p->addr);
  unsigned long long numBatchesVar = p->numBatches;
  // LIBXSMM col-major change A with B.
  gemm_param.a.primary = (void *)addr_tensorB;
  gemm_param.b.primary = (void *)addr_tensorA;
  gemm_param.c.primary = (void *)addr_tensorC;
  gemm_param.op.tertiary = (void *)&numBatchesVar;
  sgemm.gemm(&gemm_param);

  return 0;
}

extern "C" int iree_xsmm_matmul_invoke_f32(void *context, void *params,
                                           void *reserved) {
  typedef struct {
    int64_t addr;
    float *pA;
    int64_t offA;
    float *pB;
    int64_t offB;
    float *pC;
    int64_t offC;
  } xsmm_matmul_invoke_f32_t;
  xsmm_matmul_invoke_f32_t *p = (xsmm_matmul_invoke_f32_t *)params;

  float *addr_tensorA = p->pA + p->offA;
  float *addr_tensorB = p->pB + p->offB;
  float *addr_tensorC = p->pC + p->offC;

  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;
  // LIBXSMM col-major change A with B.
  gemm_param.a.primary = (void *)addr_tensorB;
  gemm_param.b.primary = (void *)addr_tensorA;
  gemm_param.c.primary = (void *)addr_tensorC;
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(p->addr);
  sgemm.gemm(&gemm_param);

  return 0;
}

extern "C" int iree_xsmm_unary_invoke(void *context, void *params,
                                      void *reserved) {
  typedef struct {
    int64_t addr;
    float *pA;
    int64_t offA;
    float *pB;
    int64_t offB;
  } xsmm_unary_invoke;
  xsmm_unary_invoke *p = (xsmm_unary_invoke *)params;

  float *addr_a = p->pA + p->offA;
  float *addr_b = p->pB + p->offB;

  libxsmm_meltwfunction_unary kernel =
      reinterpret_cast<libxsmm_meltwfunction_unary>(p->addr);
  libxsmm_meltw_unary_param param;
  param.in.primary = (void *)addr_a;
  param.out.primary = (void *)addr_b;
  kernel(&param);

  return 0;
}
