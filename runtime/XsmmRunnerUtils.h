//===- CRunnerUtils.h - Utils for debugging MLIR execution ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares basic classes and functions to manipulate structured MLIR
// types at runtime. Entities in this file must be compliant with C++11 and be
// retargetable, including on targets without a C++ runtime.
//
//===----------------------------------------------------------------------===//

#ifndef TPP_EXECUTIONENGINE_CRUNNERUTILS_H
#define TPP_EXECUTIONENGINE_CRUNNERUTILS_H

#include "libxsmm.h"
#include "mlir/ExecutionEngine/Float16bits.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t
xsmm_gemm_dispatch(const libxsmm_datatype, int64_t, int64_t, int64_t, int64_t,
                   int64_t, int64_t, const libxsmm_gemm_flags);

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t xsmm_unary_dispatch(
    const libxsmm_meltw_unary_type, const libxsmm_datatype, int64_t, int64_t,
    int64_t, int64_t, const libxsmm_meltw_unary_flags);

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t xsmm_binary_dispatch(
    const libxsmm_meltw_binary_type, const libxsmm_datatype, int64_t, int64_t,
    int64_t, int64_t, int64_t, const libxsmm_meltw_binary_flags);

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t
xsmm_brgemm_dispatch(const libxsmm_datatype, int64_t, int64_t, int64_t, int64_t,
                     int64_t, int64_t, const libxsmm_gemm_flags);

extern "C" MLIR_RUNNERUTILS_EXPORT void
xsmm_gemm_invoke(const libxsmm_datatype dType, int64_t addr, void *alignedPtrA,
                 int64_t offsetA, void *alignedPtrB, int64_t offsetB,
                 void *alignedPtrC, int64_t offsetC);

extern "C" MLIR_RUNNERUTILS_EXPORT void
xsmm_unary_invoke(const libxsmm_datatype dType, int64_t addr,
                  void *alignedPtrIn, int64_t offsetIn, void *alignedPtrOut,
                  int64_t offsetOut);

extern "C" MLIR_RUNNERUTILS_EXPORT void
xsmm_unary_scalar_invoke(const libxsmm_datatype, int64_t addr, float scalar,
                         void *alignedPtrOut, int64_t offsetOut);

extern "C" MLIR_RUNNERUTILS_EXPORT void
xsmm_binary_invoke(const libxsmm_datatype dType, int64_t addr,
                   void *alignedPtrLhs, int64_t offsetLhs, void *alignedPtrRhs,
                   int64_t offsetRhs, void *alignedPtrOut, int64_t offsetOut);

extern "C" MLIR_RUNNERUTILS_EXPORT void
xsmm_brgemm_invoke(const libxsmm_datatype dType, int64_t addr,
                   void *alignedPtrA, int64_t offsetA, void *alignedPtrB,
                   int64_t offsetB, void *alignedPtrC, int64_t offsetC,
                   int64_t numBatches);

extern "C" MLIR_RUNNERUTILS_EXPORT void xsmm_fused_brgemm_invoke(
    const libxsmm_datatype dType, int64_t addr, void *alignedPtrA,
    int64_t offsetA, void *alignedPtrB, int64_t offsetB, void *alignedPtrC,
    int64_t offsetC, void *alignedPtrD, int64_t offsetD, int64_t numBatches);

//----------------------------------------------------------------------------//
// BRGEMM connection on the IREE side.
//----------------------------------------------------------------------------//

/// Eternal functions imported in IREE must pass everything via void*.
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_brgemm_dispatch(void *context, void *params, void *reserved);
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_gemm_dispatch(void *context, void *params, void *reserved);
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_unary_dispatch(void *context, void *params, void *reserved);
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_binary_dispatch(void *context, void *params, void *reserved);

extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_brgemm_invoke(void *context, void *params, void *reserved);
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_gemm_invoke(void *context, void *params, void *reserved);
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_unary_invoke(void *context, void *params, void *reserved);
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_binary_invoke(void *context, void *params, void *reserved);

#endif // TPP_EXECUTIONENGINE_CRUNNERUTILS_H
