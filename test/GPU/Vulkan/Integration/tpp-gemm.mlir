// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=vulkan \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

// TODO enable when SPIRV annotations can be automatically applied
// TODO enable when SCF to SPIRV conversion is added
// XFAIL:*

func.func @entry() {
  %0 = memref.alloc() : memref<8x8xf32>
  %1 = memref.alloc() : memref<8x8xf32>
  %2 = memref.alloc() : memref<8x8xf32>

  %cst0 = arith.constant 0.0 : f32
  %cst1 = arith.constant 1.0 : f32
  %cst2 = arith.constant 2.0 : f32

  %cast_a = memref.cast %0 : memref<8x8xf32> to memref<*xf32>
  gpu.host_register %cast_a : memref<*xf32>
  %cast_b = memref.cast %1 : memref<8x8xf32> to memref<*xf32>
  gpu.host_register %cast_b : memref<*xf32>
  %cast_c = memref.cast %2 :memref<8x8xf32> to memref<*xf32>
  gpu.host_register %cast_c : memref<*xf32>

  linalg.fill ins(%cst1 : f32) outs(%0 : memref<8x8xf32>)
  linalg.fill ins(%cst2 : f32) outs(%1 : memref<8x8xf32>)
  linalg.fill ins(%cst0 : f32) outs(%2 : memref<8x8xf32>)

  tpp.gemm ins(%0 : memref<8x8xf32>, %1 : memref<8x8xf32>, %2: memref<8x8xf32>)
           outs(%2: memref<8x8xf32>)

  call @printMemrefF32(%cast_c) : (memref<*xf32>) -> ()

  memref.dealloc %0 : memref<8x8xf32>
  memref.dealloc %1 : memref<8x8xf32>
  memref.dealloc %2 : memref<8x8xf32>

  return
}

func.func private @printMemrefF32(memref<*xf32>)

// CHECK-COUNT-8: {{\[}}16,   16,   16,   16,   16,   16,   16,   16{{\]}}
