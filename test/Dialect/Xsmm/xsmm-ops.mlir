// RUN: tpp-opt %s | tpp-opt | FileCheck %s

// CHECK-LABEL: @xsmm_dialect
func.func @xsmm_dialect(%arg0: memref<2x2xf32>,
                        %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>) {

  // CHECK: xsmm.binary
  xsmm.binary add(dataType f32, %arg0, %arg1)
    : (memref<2x2xf32>, memref<2x2xf32>) -> ()

  // CHECK: xsmm.unary
  xsmm.unary relu(dataType f32, %arg0)
    : (memref<2x2xf32>) -> ()

  // CHECK: xsmm.binary.dispatch
  %0 = xsmm.binary.dispatch add [3, 2, 1] (broadcast none dataType f32)

  // CHECK: xsmm.unary.dispatch
  %1 = xsmm.unary.dispatch identity [3, 2, 1] (broadcast row dataType f32)

  // CHECK: xsmm.matmul
  xsmm.matmul (dataType f32, %arg0, %arg1, %arg2) 
    : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()

  // CHECK: xsmm.matmul.dispatch
  %2 = xsmm.matmul.dispatch [3, 2, 1, 3, 2, 1] flags = (none) data_type = f32
  // CHECK-NEXT: xsmm.matmul.dispatch
  %3 = xsmm.matmul.dispatch [3, 2, 1, 3, 2, 1] flags = (beta_0) data_type = f32
  // CHECK-NEXT: xsmm.matmul.dispatch
  %4 = xsmm.matmul.dispatch [3, 2, 1, 3, 2, 1] flags = (beta_0) data_type = bf16
  // CHECK-NEXT: xsmm.matmul.dispatch
  %5 = xsmm.matmul.dispatch [3, 2, 1, 3, 2, 1] flags = (vnni_a, vnni_b) data_type = bf16
  // CHECK-NEXT: xsmm.brgemm.dispatch
  %6 = xsmm.brgemm.dispatch [3, 2, 1, 3, 2, 1] flags = (vnni_a, vnni_b) data_type = bf16
  // CHECK-NEXT: xsmm.brgemm.dispatch
  %7 = xsmm.brgemm.dispatch [3, 2, 1, 3, 2, 1] flags = (beta_0) data_type = bf16
  // CHECK-NEXT: xsmm.brgemm.dispatch
  %8 = xsmm.brgemm.dispatch [3, 2, 1, 3, 2, 1] flags = (beta_0) data_type = f32
  // CHECK-NEXT: xsmm.brgemm.dispatch
  %9 = xsmm.brgemm.dispatch [3, 2, 1, 3, 2, 1] flags = (none) data_type = f32
  // CHECK: xsmm.matmul.dispatch {{.*}} {myAttr = "myattr"}
  %10 = xsmm.matmul.dispatch [3, 2, 1, 3, 2, 1] flags = (none) data_type = f32 {myAttr = "myattr"}

  return
}
