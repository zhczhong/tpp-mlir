import numpy
import sys
import argparse

numpy.set_printoptions(threshold=sys.maxsize)


def generate_single_matmul_mlir(M, N, K):
    mat_A = numpy.random.rand(M, K)
    mat_B = numpy.random.rand(K, N)
    mat_C = numpy.dot(mat_A, mat_B)
    block_start = "{"
    block_end = "}"
    mlir_code = f'''
func.func @entry() {block_start}
  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32
  // Initialize various matrices.
  %da = arith.constant dense<{numpy.array2string(mat_A, separator=", ", max_line_width=100000)}> : tensor<{M}x{K}xf32>
  %db = arith.constant dense<{numpy.array2string(mat_B, separator=", ", max_line_width=100000)}> : tensor<{K}x{N}xf32>
  // Call kernel.
  %C = arith.constant dense<0.0> : tensor<{M}x{N}xf32>
  %0 = linalg.matmul ins(%da, %db : tensor<{M}x{K}xf32>, tensor<{K}x{N}xf32>) outs(%C : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %result = arith.constant dense<{numpy.array2string(mat_C, separator=", ", max_line_width=100000)}> : tensor<{M}x{N}xf32>
  %threshold = arith.constant 0.001: f32
  check.expect_almost_eq(%result, %0, %threshold): tensor<{M}x{N}xf32>, tensor<{M}x{N}xf32>, f32
  return
{block_end}
    '''
    return mlir_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLIR Correctness Check")
    parser.add_argument('--M', type=int, default=4)
    parser.add_argument('--N', type=int, default=4)
    parser.add_argument('--K', type=int, default=4)
    args = parser.parse_args()
    code = generate_single_matmul_mlir(args.M, args.N, args.K)
    print(code)
