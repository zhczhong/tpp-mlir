// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// BENCH_TOTAL_FLOPS: 537395200

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @entry(%arg0: tensor<256x1024xf32>, %arg1: tensor<256x1024xf32>) -> tensor<256x1024xf32> {
    %weights = arith.constant dense<1.0> : tensor<1024x1024xf32>
    %bias = arith.constant dense<1.0> : tensor<256x1024xf32>

    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %weights : tensor<256x1024xf32>, tensor<1024x1024xf32>) outs(%arg1 : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<256x1024xf32>
    %1 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%bias : tensor<256x1024xf32>) outs(%0 : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %in, %out : f32
      linalg.yield %3 : f32
    } -> tensor<256x1024xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%1 : tensor<256x1024xf32>) {
    ^bb0(%out: f32):
      %3 = arith.maximumf %out, %cst : f32
      linalg.yield %3 : f32
    } -> tensor<256x1024xf32>
    %3 = bufferization.materialize_in_destination %2 in %arg1 : (tensor<256x1024xf32>, tensor<256x1024xf32>) -> tensor<256x1024xf32>
    return %3 : tensor<256x1024xf32>
  }
}
