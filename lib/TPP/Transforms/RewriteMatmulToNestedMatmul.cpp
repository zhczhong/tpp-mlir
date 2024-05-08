/*******************************************************************************
 * Copyright 2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <iostream>
////////////////////////////////////////////////////
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include <utility>

using namespace mlir;
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::linalg;
using namespace mlir::scf;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_REWRITEMATMULTONESTEDMATMUL
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

struct MatmulConfig {
  int MBlock, NBlock, KBlock;
  int MThreads, NThreads, KThreads;
  int innerMostMBlock, innerMostNBlock, innerMostKBlock;
};

template <typename T> inline T divAndCeil(T a, T b) { return (a - 1) / b + 1; }

// TODO: Simple TTI interface

template <typename LinalgOpTy>
MatmulConfig getDefaultMatmulConfig(LinalgOpTy &linalgOp) {
  auto M = linalgOp.getShape(linalgOp.getDpsInputOperand(0))[0];
  auto N = linalgOp.getShape(linalgOp.getDpsInputOperand(1))[1];
  auto K = linalgOp.getShape(linalgOp.getDpsInputOperand(1))[0];
  MatmulConfig cfg;

  // innermost Block
  auto defaultBlock = 32;
  cfg.innerMostMBlock = M % defaultBlock == 0 ? defaultBlock : M;
  cfg.innerMostNBlock = N % defaultBlock == 0 ? defaultBlock : N;
  cfg.innerMostKBlock = K % defaultBlock == 0 ? defaultBlock : K;

  // Number of block
  auto MNumBlock = M / cfg.innerMostMBlock;
  auto NNumBlock = N / cfg.innerMostNBlock;
  auto KNumBlock = K / cfg.innerMostKBlock;

  // Threads
  cfg.MThreads = 1;
  cfg.NThreads = 1;
  cfg.KThreads = 1;

  // Block
  cfg.MBlock = divAndCeil((int)MNumBlock, cfg.MThreads) * cfg.innerMostMBlock;
  cfg.NBlock = divAndCeil((int)NNumBlock, cfg.NThreads) * cfg.innerMostNBlock;
  cfg.KBlock = divAndCeil((int)KNumBlock, cfg.KThreads) * cfg.innerMostKBlock;
  return cfg;
}

/// Returns true if the maximum tile offset `tileSize * numThreads-1` is less
/// than `iterationSize`.
static bool canOmitTileOffsetInBoundsCheck(OpFoldResult tileSize,
                                           OpFoldResult numThreads,
                                           OpFoldResult iterationSize) {
  std::optional<int64_t> tileSizeConst = getConstantIntValue(tileSize);
  std::optional<int64_t> numThreadsConst = getConstantIntValue(numThreads);
  std::optional<int64_t> iterSizeConst = getConstantIntValue(iterationSize);
  if (!tileSizeConst || !numThreadsConst || !iterSizeConst)
    return false;
  return *tileSizeConst * (*numThreadsConst - 1) < *iterSizeConst;
}

/// Build an `affine_max` of all the `vals`.
static OpFoldResult buildMax(OpBuilder &b, Location loc,
                             ArrayRef<OpFoldResult> vals) {
  return affine::makeComposedFoldedAffineMax(
      b, loc, AffineMap::getMultiDimIdentityMap(vals.size(), loc.getContext()),
      vals);
}
/// Build an `affine_min` of all the `vals`.
static OpFoldResult buildMin(OpBuilder &b, Location loc,
                             ArrayRef<OpFoldResult> vals) {
  return affine::makeComposedFoldedAffineMin(
      b, loc, AffineMap::getMultiDimIdentityMap(vals.size(), loc.getContext()),
      vals);
}

/// Fill out the `tiledOffsets` and `tiledSizes` to be used to tile to a given
/// number of threads.
static void calculateTileOffsetsAndSizes(
    RewriterBase &b, Location loc, scf::ForallOp forallOp,
    ArrayRef<OpFoldResult> numThreads, SmallVector<Range> loopRanges,
    bool omitTileOffsetBoundsCheck,
    std::optional<ArrayRef<OpFoldResult>> nominalTileSizes,
    SmallVector<OpFoldResult> &tiledOffsets,
    SmallVector<OpFoldResult> &tiledSizes) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(forallOp.getBody(0));

  ValueRange threadIds = forallOp.getInductionVars();
  SmallVector<OpFoldResult> nonZeroNumThreads =
      llvm::to_vector(llvm::make_filter_range(numThreads, [](OpFoldResult ofr) {
        return !isConstantIntValue(ofr, 0);
      }));
  int64_t nLoops = loopRanges.size();
  tiledOffsets.reserve(nLoops);
  tiledSizes.reserve(nLoops);
  for (unsigned loopIdx = 0, threadIdIdx = 0; loopIdx < nLoops; ++loopIdx) {
    bool overflow = loopIdx >= numThreads.size();
    bool isZero = !overflow && isConstantIntValue(numThreads[loopIdx], 0);
    // Degenerate case: take the whole domain.
    if (overflow || isZero) {
      tiledOffsets.push_back(loopRanges[loopIdx].offset);
      tiledSizes.push_back(loopRanges[loopIdx].size);
      continue;
    }

    // Tiled case: compute the offset and size.
    AffineExpr i, j, m, n, o;
    bindDims(b.getContext(), i, j);
    bindSymbols(b.getContext(), m, n, o);
    OpFoldResult size = loopRanges[loopIdx].size;
    OpFoldResult offset = loopRanges[loopIdx].offset;
    OpFoldResult threadId = threadIds[threadIdIdx];
    // Symbolic fixed max size per thread.
    // TODO: floor + 0/1 depending on case for better load-balancing.
    OpFoldResult tileSizePerThread =
        nominalTileSizes.has_value()
            ? (*nominalTileSizes)[loopIdx]
            : makeComposedFoldedAffineApply(
                  b, loc, m.ceilDiv(n),
                  ArrayRef<OpFoldResult>{size, nonZeroNumThreads[threadIdIdx]});

    // Dynamic offset shifted by threadId * maxSizePerThread.
    OpFoldResult offsetPerThread = makeComposedFoldedAffineApply(
        b, loc, i + j * m, {offset, threadId, tileSizePerThread});
    // Dynamic upper-bound depending on the threadId.
    OpFoldResult residualTileSize = makeComposedFoldedAffineApply(
        b, loc, i + j * m - n,
        {offset, nonZeroNumThreads[threadIdIdx], tileSizePerThread, size});
    if (!isConstantIntValue(residualTileSize, 0)) {
      OpFoldResult sizeMinusOffsetPerThread = makeComposedFoldedAffineApply(
          b, loc, -i + m, {offsetPerThread, size});
      tileSizePerThread =
          buildMin(b, loc, {sizeMinusOffsetPerThread, tileSizePerThread});
    }

    tiledOffsets.push_back(offsetPerThread);
    // TODO: if tileSizePerThread <= 0 early exit.
    if (!omitTileOffsetBoundsCheck &&
        !canOmitTileOffsetInBoundsCheck(tileSizePerThread,
                                        nonZeroNumThreads[threadIdIdx], size))
      tileSizePerThread =
          buildMax(b, loc, {b.getIndexAttr(0), tileSizePerThread});

    tiledSizes.push_back(tileSizePerThread);
    ++threadIdIdx;
  }
}

template <typename LoopTy>
static FailureOr<TiledLinalgOp>
tileLinalgOpImpl(RewriterBase &b, LinalgOp op, ArrayRef<OpFoldResult> tileSizes,
                 const LinalgTilingOptions &options) {
  OpBuilder::InsertionGuard g(b);

  auto nLoops = op.getNumLoops();
  // Initial tile sizes may be too big, only take the first nLoops.
  tileSizes = tileSizes.take_front(nLoops);

  if (llvm::all_of(tileSizes, [](OpFoldResult ofr) {
        return getConstantIntValue(ofr) == static_cast<int64_t>(0);
      })) {
    TiledLinalgOp tiledOp;
    tiledOp.op = cast<LinalgOp>(b.clone(*op.getOperation()));
    tiledOp.tensorResults.assign(tiledOp.op->result_begin(),
                                 tiledOp.op->result_end());
    return tiledOp;
  }

  // 1. Build the tiled loop ranges.
  SmallVector<OpFoldResult> allShapeSizes =
      op.createFlatListOfOperandDims(b, op.getLoc());
  AffineMap shapeSizesToLoopsMap = op.getShapesToLoopsMap();
  if (!shapeSizesToLoopsMap)
    return failure();

  auto [loopRanges, loopIndexToRangeIndex] = makeTiledLoopRanges(
      b, op.getLoc(), shapeSizesToLoopsMap, allShapeSizes, tileSizes);

  SmallVector<utils::IteratorType, 4> iteratorTypes;
  for (const auto &attr : enumerate(op.getIteratorTypesArray())) {
    if (loopIndexToRangeIndex.count(attr.index()))
      iteratorTypes.push_back(attr.value());
  }
  // If interchangeVector is empty, use the identity. Build the permutation map
  // otherwise.
  auto invPermutationMap =
      AffineMap::getMultiDimIdentityMap(tileSizes.size(), b.getContext());
  if (!options.interchangeVector.empty()) {
    // Based on the pruned iterations (due to zero tile size), recompute the
    // interchange vector.
    SmallVector<unsigned, 4> interchangeVector;
    interchangeVector.reserve(options.interchangeVector.size());
    for (auto pos : options.interchangeVector) {
      auto it = loopIndexToRangeIndex.find(pos);
      if (it == loopIndexToRangeIndex.end())
        continue;
      interchangeVector.push_back(it->second);
    }
    // Interchange vector is guaranteed to be a permutation,
    // `inversePermutation` must succeed.
    invPermutationMap = inversePermutation(
        AffineMap::getPermutationMap(interchangeVector, b.getContext()));
    assert(invPermutationMap);
    SmallVector<int64_t> permutation(interchangeVector.begin(),
                                     interchangeVector.end());
    applyPermutationToVector(loopRanges, permutation);
    applyPermutationToVector(iteratorTypes, permutation);
  }

  // Handle distribution. Create a vector of the same size of loops that are to
  // be tiled.
  SmallVector<linalg::ProcInfo> procInfo;
  if (options.distribution) {
    procInfo.resize(
        iteratorTypes.size(),
        linalg::ProcInfo{nullptr, nullptr, linalg::DistributionMethod::None});
    // Collect loop ranges of tiled loops, loops that are parallel.
    SmallVector<Range> parallelLoopRanges;
    for (const auto &iteratorType : llvm::enumerate(iteratorTypes)) {
      if (!isParallelIterator(iteratorType.value()))
        break;
      parallelLoopRanges.push_back(loopRanges[iteratorType.index()]);
    }
    auto returnedProcInfo =
        options.distribution->procInfo(b, op.getLoc(), parallelLoopRanges);
    unsigned procIdIdx = 0;
    // Update the distribution information for the loops.
    for (const auto &iteratorType : llvm::enumerate(iteratorTypes)) {
      if (!isParallelIterator(iteratorType.value()))
        break;
      procInfo[iteratorType.index()] = returnedProcInfo[procIdIdx++];
    }
  }

  // 2. Create the tiled loops.
  LinalgOp res = op;
  SmallVector<Value, 4> ivs, tensorResults;
  auto tiledLoopBodyBuilder =
      [&](OpBuilder &builder, Location loc, ValueRange localIvs,
          ValueRange operandValuesToUse) -> scf::ValueVector {
    ivs.assign(localIvs.begin(), localIvs.end());

    // When an `interchangeVector` is present, it has been applied to the
    // loop ranges and the iterator types. Apply its inverse to the
    // resulting loop `ivs` to match the op definition.
    SmallVector<Value, 4> interchangedIvs;
    if (!options.interchangeVector.empty()) {
      for (AffineExpr result : invPermutationMap.getResults())
        interchangedIvs.push_back(
            ivs[cast<AffineDimExpr>(result).getPosition()]);
    } else {
      interchangedIvs.assign(ivs.begin(), ivs.end());
    }

    // Tile the `operandValuesToUse` that either match the `op` operands
    // themselves or the tile loop arguments forwarding them.
    assert(operandValuesToUse.size() ==
               static_cast<size_t>(op->getNumOperands()) &&
           "expect the number of operands and inputs and outputs to match");
    SmallVector<Value> valuesToTile = operandValuesToUse;
    SmallVector<OpFoldResult> sizeBounds =
        makeComposedFoldedMultiResultAffineApply(b, loc, shapeSizesToLoopsMap,
                                                 allShapeSizes);
    SmallVector<Value> tiledOperands = makeTiledShapes(
        b, loc, op, valuesToTile, getAsOpFoldResult(interchangedIvs), tileSizes,
        sizeBounds,
        /*omitPartialTileCheck=*/false);

    SmallVector<Type> resultTensorTypes =
        getTensorOutputTypes(op, tiledOperands);
    res = clone(b, op, resultTensorTypes, tiledOperands);
    tensorResults =
        insertSlicesBack(builder, loc, op, tiledOperands, res->getResults());
    return scf::ValueVector(tensorResults.begin(), tensorResults.end());
  };
  GenerateLoopNest<LoopTy>::doit(b, op.getLoc(), loopRanges, op, iteratorTypes,
                                 tiledLoopBodyBuilder, procInfo);

  // 3. Transform IndexOp results w.r.t. the tiling.
  transformIndexOps(b, res, ivs, loopIndexToRangeIndex);

  // 4. Gather the newly created loops and return them with the new op.
  SmallVector<Operation *, 8> loops;
  loops.reserve(ivs.size());
  for (auto iv : ivs) {
    if (isa<BlockArgument>(iv)) {
      loops.push_back(cast<BlockArgument>(iv).getOwner()->getParentOp());
      assert(loops.back() && "no owner found for induction variable!");
    } else {
      // TODO: Instead of doing this, try to recover the ops used instead of the
      // loop.
      loops.push_back(nullptr);
    }
  }

  // 5. Get the tensor results from the outermost loop if available. Otherwise
  // use the previously captured `tensorResults`.
  Operation *outermostLoop = nullptr;
  for (Operation *loop : loops)
    if ((outermostLoop = loop))
      break;

  return TiledLinalgOp{
      res, loops, outermostLoop ? outermostLoop->getResults() : tensorResults};
}

FailureOr<linalg::ForallReductionTilingResult>
tileAllUsingForall(RewriterBase &b, PartialReductionOpInterface op,
                   ArrayRef<OpFoldResult> numThreads,
                   ArrayRef<OpFoldResult> tileSizes,
                   std::optional<ArrayAttr> mapping) {
  Location loc = op.getLoc();
  OpBuilder::InsertionGuard g(b);

  // Ops implementing PartialReductionOpInterface are expected to implement
  // TilingInterface.
  // TODO: proper core mechanism to tie interfaces together.
  auto tilingInterfaceOp = cast<TilingInterface>(op.getOperation());

  // Ops implementing PartialReductionOpInterface are not necessarily expected
  // to implement TilingInterface.. This cast is unsafe atm.
  // TODO: proper core mechanism to tie interfaces together.
  // TODO: this function requires a pair of interfaces ..
  auto destinationStyleOp =
      dyn_cast<DestinationStyleOpInterface>(op.getOperation());
  if (!destinationStyleOp)
    return b.notifyMatchFailure(op, "not a destination style op");

  // Actually this only work for Linalg ops atm.
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op.getOperation());
  if (!linalgOp)
    return b.notifyMatchFailure(op, "not a linalg op");

  SmallVector<Range> iterationDomain = tilingInterfaceOp.getIterationDomain(b);
  if (op->getNumResults() != 1)
    return b.notifyMatchFailure(
        op, "don't support ops with multiple results for now");

  SmallVector<utils::IteratorType> iterators =
      tilingInterfaceOp.getLoopIteratorTypes();
  SmallVector<int> redDims;
  for (auto [idx, iteratorType] :
       llvm::enumerate(tilingInterfaceOp.getLoopIteratorTypes())) {
    if (iteratorType == utils::IteratorType::reduction)
      redDims.push_back(idx);
  }
  bool hasReductionThreads = false;
  for (auto dim : redDims) {
    if (!isConstantIntValue(numThreads[dim], 0) &&
        !isConstantIntValue(numThreads[dim], 1)) {
      hasReductionThreads = true;
      break;
    }
  }

  if (!tileSizes.empty() && tileSizes.size() != numThreads.size())
    return b.notifyMatchFailure(op, "if tile sizes are present it must have as "
                                    "many elements as number of threads");

  if (redDims.front() >= numThreads.size())
    return b.notifyMatchFailure(
        op, "reduction dimension must be mapped to threads");

  // 1. Create the inital tensor value.
  FailureOr<Operation *> identityTensor = nullptr;
  if (hasReductionThreads) {
    identityTensor = op.generateInitialTensorForPartialReduction(
        b, loc, numThreads, redDims);
  }
  if (failed(identityTensor))
    return b.notifyMatchFailure(op,
                                "cannot create a tensor of identity value.");

  // Gather destination tensors.
  SmallVector<Value> dest;
  if (failed(tensor::getOrCreateDestinations(b, loc, op, dest)))
    return b.notifyMatchFailure(op, "failed to get destination tensors");

  Operation *tiledOp = nullptr;

  SmallVector<OpFoldResult> nonZeroNumThreads =
      llvm::to_vector(llvm::make_filter_range(numThreads, [](OpFoldResult ofr) {
        return !isConstantIntValue(ofr, 0);
      }));
  SmallVector<Value> materializedNonZeroNumThreads =
      getValueOrCreateConstantIndexOp(b, loc, nonZeroNumThreads);

  // 2. Create the ForallOp with an empty region.
  scf::ForallOp forallOp = b.create<scf::ForallOp>(
      loc, getAsOpFoldResult(materializedNonZeroNumThreads),
      hasReductionThreads ? (*identityTensor)->getResults() : dest, mapping);

  // 3. Calculate the tile offsets and sizes for the subsequent loop that will
  // be nested under `forallOp`.
  SmallVector<OpFoldResult> tiledOffsets, tiledSizes;
  calculateTileOffsetsAndSizes(b, loc, forallOp, numThreads, iterationDomain,
                               /*omitTileOffsetBoundsCheck =*/false,
                               /*nominalTileSizes=*/tileSizes, tiledOffsets,
                               tiledSizes);

  // 4. Clone the tileable op and update its destination operands to use the
  // output bbArgs of the ForallOp.
  SmallVector<Value> tilingResults;
  ArrayRef<BlockArgument> destBbArgs = forallOp.getRegionIterArgs();
  {
    // 4.a. RAII guard, inserting within forallOp, before terminator.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(forallOp.getTerminator());

    SmallVector<Value> tiledDpsInitOperands;
    for (Value initOperand : destinationStyleOp.getDpsInits()) {
      if (hasReductionThreads) {
        auto *it = llvm::find(dest, initOperand);
        assert(it != dest.end() && "dest operand not found in dest");
        unsigned destNum = std::distance(dest.begin(), it);
        SmallVector<OpFoldResult> strides(numThreads.size(), b.getIndexAttr(1));
        SmallVector<OpFoldResult> outOffsets(numThreads.size(),
                                             b.getIndexAttr(0));
        SmallVector<OpFoldResult> sizes;
        for (auto s :
             cast<RankedTensorType>(destBbArgs[destNum].getType()).getShape()) {
          sizes.emplace_back(getAsIndexOpFoldResult(b.getContext(), (int)s));
        }
        for (auto dim : redDims) {
          sizes[dim] = b.getIndexAttr(1);
        }

        auto nonZeroDimIdx = 0;
        for (auto dim = 0; dim < numThreads.size(); dim++) {
          if (!isConstantIntValue(numThreads[dim], 0)) {
            if (llvm::find(redDims, dim) != redDims.end())
              outOffsets[dim] = forallOp.getInductionVars()[nonZeroDimIdx];
            nonZeroDimIdx++;
          }
        }
        // TODO: use SubsetExtractOpInterface once it is available.
        tiledDpsInitOperands.push_back(b.create<tensor::ExtractSliceOp>(
            loc, cast<RankedTensorType>(initOperand.getType()),
            destBbArgs[destNum], outOffsets, sizes, strides));
      } else {
        tiledDpsInitOperands.push_back(initOperand);
      }
    }

    // 4.b. Clone the op and update init operands.
    // We cannot use a IRMapping here because it can replace
    // different OpOperands with the same value.
    Operation *clonedOp = b.clone(*op.getOperation());
    b.modifyOpInPlace(clonedOp, [&]() {
      for (auto [initOperandPtr, tiledInitValue] : llvm::zip_equal(
               cast<DestinationStyleOpInterface>(clonedOp).getDpsInitsMutable(),
               tiledDpsInitOperands)) {
        initOperandPtr.set(tiledInitValue);
      }
    });

    // 5. Tile the cloned op and delete the clone.
    FailureOr<TilingResult> tilingResult =
        cast<TilingInterface>(clonedOp).getTiledImplementation(b, tiledOffsets,
                                                               tiledSizes);
    if (failed(tilingResult))
      return clonedOp->emitError("Failed to tile op: ");
    if (tilingResult->tiledOps.size() != 1) {
      return clonedOp->emitError("expected a single produced tiled op, got ")
             << tilingResult->tiledOps.size();
    }
    tiledOp = tilingResult->tiledOps.front();
    tilingResults = tilingResult->tiledValues;

    b.eraseOp(clonedOp);
  }

  // 6. Insert the partial reductions back into a new tensor.
  for (auto [index, result, bbArg] : llvm::zip(
           llvm::seq<unsigned>(0, dest.size()), tilingResults, destBbArgs)) {
    // 6.a. Partial subset information is inserted just before the terminator.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(forallOp.getTerminator());

    SmallVector<OpFoldResult> resultOffsets, resultSizes;
    if (failed(tilingInterfaceOp.getResultTilePosition(
            b, index, tiledOffsets, tiledSizes, resultOffsets, resultSizes)))
      return op->emitOpError("output offsets couldn't be calculated");
    SmallVector<OpFoldResult> resultOffsetsRank, resultSizesRank;
    int64_t offIdx = 0;
    int64_t sizeIdx = 0;
    int64_t nonZeroDimIdx = 0;
    for (int64_t i = 0; i < numThreads.size(); ++i) {
      if (llvm::find(redDims, i) != redDims.end()) {
        if (hasReductionThreads) {
          resultOffsetsRank.push_back(
              forallOp.getInductionVars()[nonZeroDimIdx++]);
          resultSizesRank.push_back(b.getIndexAttr(1));
        } else {
          nonZeroDimIdx++;
        }
        continue;
      }
      if (!isConstantIntValue(numThreads[i], 0)) {
        nonZeroDimIdx++;
      }
      resultOffsetsRank.push_back(resultOffsets[offIdx++]);
      resultSizesRank.push_back(resultSizes[sizeIdx++]);
    }
    SmallVector<OpFoldResult> strides(resultSizesRank.size(),
                                      b.getIndexAttr(1));

    // 6.b. Parallel insertions are inserted at the end of the combining
    // terminator.
    b.setInsertionPointToEnd(forallOp.getTerminator().getBody());
    b.create<tensor::ParallelInsertSliceOp>(
        loc, result, bbArg, resultOffsetsRank, resultSizesRank, strides);
  }

  // 7. Merge the partial reductions.
  Operation *mergeOp = nullptr;
  b.setInsertionPointAfter(forallOp);
  if (hasReductionThreads) {
    Operation *mergeOp =
        op.mergeReductions(b, loc, forallOp->getResults(), redDims);
    b.replaceOp(op, mergeOp->getResults());
  } else {
    b.replaceOp(op, forallOp->getResults());
  }

  // 8. Return.
  ForallReductionTilingResult results;
  results.initialOp = *identityTensor;
  results.loops = forallOp;
  results.parallelTiledOp = tiledOp;
  results.mergeOp = mergeOp;
  return results;
}

Value tensorViewRankedTensor(RewriterBase &rewriter,
                             RankedTensorType outTensorType, Value value) {
  Value result, currentValue = value;
  auto loc = currentValue.getLoc();
  auto inTensorType = currentValue.getType().cast<RankedTensorType>();
  auto inShape = inTensorType.getShape();
  auto outShape = outTensorType.getShape();
  auto tensorElementType = inTensorType.getElementType();

  if (inShape == outShape) {
    return currentValue;
  }

  if (outTensorType.getNumDynamicDims() != inTensorType.getNumDynamicDims()) {
    SmallVector<int64_t> alignOutShape(outShape.begin(), outShape.end());
    if (outShape.size() < inShape.size()) {
      SmallVector<int64_t> oneVector(inShape.size() - outShape.size(), 1);
      alignOutShape.insert(alignOutShape.begin(), oneVector.begin(),
                           oneVector.end());
    } else {
      alignOutShape.erase(alignOutShape.begin(),
                          alignOutShape.begin() +
                              (outShape.size() - inShape.size()));
    }
    auto type = RankedTensorType::get(alignOutShape, tensorElementType);
    currentValue = rewriter.create<tensor::CastOp>(loc, type, currentValue);
    if (type == outTensorType) {
      return currentValue;
    }
  }

  if (outShape.size() < inShape.size()) {
    SmallVector<ReassociationIndices> reassocIndices;
    ReassociationIndices firstEntry;
    for (auto i = 0UL; i < inShape.size() - outShape.size() + 1; i++) {
      firstEntry.push_back(i);
    }
    reassocIndices.push_back(firstEntry);
    for (auto i = inShape.size() - outShape.size() + 1; i < inShape.size();
         i++) {
      reassocIndices.push_back({i});
    }
    result = rewriter.create<tensor::CollapseShapeOp>(
        loc, outTensorType, currentValue, reassocIndices);
  } else if (outShape.size() > inShape.size()) {
    SmallVector<ReassociationIndices> reassocIndices;
    ReassociationIndices firstEntry;
    for (auto i = 0; i < outShape.size() - inShape.size() + 1; i++) {
      firstEntry.push_back(i);
    }
    reassocIndices.push_back(firstEntry);
    for (auto i = outShape.size() - inShape.size() + 1; i < outShape.size();
         i++) {
      reassocIndices.push_back({i});
    }
    result = rewriter.create<tensor::ExpandShapeOp>(
        loc, outTensorType, currentValue, reassocIndices);
  } else {
    result = rewriter.create<tensor::CastOp>(loc, outTensorType, currentValue);
  }
  return result;
}

/*
forall([PM, PN]: [MThreads, NThreads) {
  for(PK : KThreads) {
    CSlice = [KThreads, PM * MOuterBlock: (PM + 1) * MOuterBlock,
     PN * NOuterBlock: (PN + 1) * NOuterBlock]
    ASlice = A[PM * MOuterBlock: (PM + 1) * MOuterBlock, PK * KOuterBlock * (PK
+ 1) * KOuterBlock]
    BSlice = B[PK * KOuterBlock * (PK + 1) * KOuterBlock, PN *
NOuterBlock: (PN + 1) * NOuterBlock] CSlice2 = CSlice[PK, PM * MOuterBlock: (PM
+ 1) * MOuterBlock, PN * NOuterBlock: (PN + 1) * NOuterBlock]

    MNumBlock = MOuterBlock / MBlock
    NNumBlock = NOuterBlock / NBlock
    KNumBlock = KOuterBlock / KBlovk
    for([om, on, ok]: [MNumBlock, NNumBlock, KNumBlock]) {
      ASlice2 = ASlice[om * MBlock: (om + 1) * MBlock, ok * KBlock: (ok + 1) *
KBlock]
      BSlice2 = BSlice[0, om * MBlock: (om + 1) * MBlock, ok * KBlock: (ok +
1) * KBlock]
      CSlice3 = CSlice2[0, om * MBlock: (om + 1) * MBlock, on * NBlock:
(on + 1) * NBlock] (init with 0 when ok == 0)
      MNumInnerBlock = MBlock / iim_block_
      ...
      for([im, in]: [MNumInnerBlock, NNumInnerBlock]) {
        ASlice3 = ASlice2[im * iim_block_: (im + 1) * iim_block_, :]
        BSlice3 = BSlice2[0, im * iim_block_: (im + 1) * iim_block_, :]
        CSlice4 = CSlice3[0, im * iim_block_: (im + 1) * iim_block_, in *
iin_block_: (in + 1) * iin_block_] (init with 0 when ok == 0)
        brgemm(bs=KNumInnerBlock, M=iim_block_, N=iin_block_, K=iik_block,
A=ASlice3, B=BSlice3, C=CSlice4, onlyUpdate=(ok!=0));
      }
    }
  }
  C = final_reduce(CSlice)
}
*/
template <typename LinalgOpTy>
struct rewriteToNestedMatmul : public OpRewritePattern<LinalgOpTy> {
  using OpRewritePattern<LinalgOpTy>::OpRewritePattern;
  static_assert(llvm::is_one_of<LinalgOpTy, linalg::MatmulOp,
                                linalg::BatchMatmulOp>::value);

  void getMatmulParallelDims(linalg::LinalgOp linalgOp, unsigned operandIdx,
                             SmallVector<unsigned> &dims) const {
    AffineMap map = linalgOp.getMatchingIndexingMap(
        linalgOp.getDpsInputOperand(operandIdx));
    SmallVector<mlir::utils::IteratorType> iteratorTypes =
        linalgOp.getIteratorTypesArray();

    ArrayRef<AffineExpr> results = map.getResults();
    for (auto dim : results) {
      auto dimExpr = dyn_cast<AffineDimExpr>(dim);
      if (dimExpr && iteratorTypes[dimExpr.getPosition()] ==
                         mlir::utils::IteratorType::parallel) {
        dims.push_back(dimExpr.getPosition());
      }
    }
  }

  unsigned getOprandDim(linalg::LinalgOp &linalgOp, unsigned iteratorPos,
                        unsigned operandIdx) const {
    Value Operand;
    unsigned dimPos;
    linalgOp.mapIterationSpaceDimToOperandDim(iteratorPos, Operand, dimPos);
    return linalgOp.getShape(linalgOp.getDpsInputOperand(operandIdx))[dimPos];
  }

  LogicalResult matchAndRewrite(LinalgOpTy matmulOp,
                                PatternRewriter &rewriter) const override {
    if (matmulOp.hasPureBufferSemantics())
      return failure();
    MatmulConfig cfg = getDefaultMatmulConfig(matmulOp);
    linalg::LinalgOp genericOp;
    bool useBlockedLayout = false;
    // Step 2. Pack to Mkmk(innerMostMBlock, innerMostKBlock) amd
    // NKkn(inermostKBlock, innermostNBlock)
    // Todo: remove this step when layout propogation is ready
    {
      if (useBlockedLayout) {
        SmallVector<OpFoldResult> packSizes(
            matmulOp.getNumLoops(),
            getAsIndexOpFoldResult(rewriter.getContext(), 0));
        packSizes[0] =
            getAsIndexOpFoldResult(rewriter.getContext(), cfg.innerMostMBlock);
        packSizes[1] =
            getAsIndexOpFoldResult(rewriter.getContext(), cfg.innerMostNBlock);
        packSizes[2] =
            getAsIndexOpFoldResult(rewriter.getContext(), cfg.innerMostKBlock);
        auto linalgOp =
            mlir::linalgx::packMatmulOp(rewriter, matmulOp, packSizes);

        if (failed(linalgOp))
          return failure();

        if (linalgOp->hasPureBufferSemantics())
          return failure();
        genericOp = *linalgOp;
      } else {
        genericOp = dyn_cast<linalg::LinalgOp>(matmulOp.getOperation());
      }
    }
    if (genericOp.getOperation()->getParentOfType<scf::ForallOp>())
      return failure();

    // Step 2. Match and remove the init/fill operation
    // Fuse the fill op manually before fusion support this case(fuse it into
    // if-else block)
    bool hasFillOp = false;
    Value fillValue;
    SmallVector<LoopLikeOpInterface> KLoopHandle;
    if (auto op = dyn_cast<linalg::FillOp>(
            genericOp.getDpsInits()[0].getDefiningOp())) {
      hasFillOp = true;
      fillValue = op.getDpsInputs()[0];
      rewriter.replaceOp(op, op.getDpsInits()[0]);
    }

    // Step 3. The processes of transforming matmul to nested matmul
    // 3.0 Get the iteration infomation first
    SmallVector<unsigned> KDimPos, MDimPos, NDimPos;
    genericOp.getReductionDims(KDimPos);
    getMatmulParallelDims(genericOp, 0, MDimPos);
    getMatmulParallelDims(genericOp, 1, NDimPos);

    // 3.1 Tiling reduction parallel loop with scf::forall
    // TODO: move the reduction dim to the front. (M, N, threads) ->
    // (threads, M, N)
    {
      if (genericOp.hasPureBufferSemantics())
        return failure();
      auto KFirstDim = (int)getOprandDim(genericOp, KDimPos[0], 1);
      SmallVector<OpFoldResult> tileSizes(
          genericOp.getNumLoops(),
          getAsIndexOpFoldResult(rewriter.getContext(), 0));
      SmallVector<OpFoldResult> threads(
          genericOp.getNumLoops(),
          getAsIndexOpFoldResult(rewriter.getContext(), 0));
      bool isFirstReductionDim = true;
      for (auto reductionDim : KDimPos) {
        if (isFirstReductionDim) {
          tileSizes[reductionDim] = getAsIndexOpFoldResult(
              rewriter.getContext(),
              useBlockedLayout ? divAndCeil(KFirstDim, cfg.KThreads)
                               : divAndCeil(divAndCeil(KFirstDim, cfg.KBlock),
                                            cfg.KThreads) *
                                     cfg.KBlock);
          threads[reductionDim] = getAsIndexOpFoldResult(
              rewriter.getContext(), cfg.KThreads == 1 ? 0 : cfg.KThreads);
          isFirstReductionDim = false;
        } else {
          tileSizes[reductionDim] = getAsIndexOpFoldResult(
              rewriter.getContext(), cfg.innerMostKBlock);
          threads[reductionDim] =
              getAsIndexOpFoldResult(rewriter.getContext(), 0);
        }
      }

      auto MFirstDim = (int)getOprandDim(genericOp, MDimPos[0], 0);
      auto NFirstDim = (int)getOprandDim(genericOp, NDimPos[0], 1);
      threads[MDimPos[0]] =
          getAsIndexOpFoldResult(rewriter.getContext(), cfg.MThreads);
      threads[NDimPos[0]] =
          getAsIndexOpFoldResult(rewriter.getContext(), cfg.NThreads);
      tileSizes[MDimPos[0]] = getAsIndexOpFoldResult(
          rewriter.getContext(),
          useBlockedLayout
              ? divAndCeil(MFirstDim, cfg.MThreads)
              : divAndCeil(divAndCeil(MFirstDim, cfg.MBlock), cfg.MThreads) *
                    cfg.MBlock);
      tileSizes[NDimPos[0]] = getAsIndexOpFoldResult(
          rewriter.getContext(),
          useBlockedLayout
              ? divAndCeil(NFirstDim, cfg.NThreads)
              : divAndCeil(divAndCeil(NFirstDim, cfg.NBlock), cfg.NThreads) *
                    cfg.NBlock);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(genericOp);
      if (auto partialInterface =
              dyn_cast<PartialReductionOpInterface>(genericOp.getOperation())) {
        auto tilingResult = tileAllUsingForall(
            rewriter,
            cast<PartialReductionOpInterface>(genericOp.getOperation()),
            threads, tileSizes, std::nullopt);
        if (failed(tilingResult))
          return failure();
        genericOp = dyn_cast<linalg::LinalgOp>(tilingResult->parallelTiledOp);
      } else if (auto tilingInterface =
                     cast<TilingInterface>(genericOp.getOperation())) {
        for (auto reductionDim : KDimPos) {
          tileSizes[reductionDim] =
              getAsIndexOpFoldResult(rewriter.getContext(), 0);
        }
        auto tilingResult = linalg::tileToForallOpUsingTileSizes(
            rewriter, tilingInterface, tileSizes, std::nullopt);
        if (failed(tilingResult))
          return failure();
        rewriter.replaceOp(genericOp, tilingResult->tileOp);
        genericOp = dyn_cast<linalg::LinalgOp>(tilingResult->tiledOp);
      }
    }

    // 3.2 Tiling outer loop with scf::for
    {
      if (genericOp.hasPureBufferSemantics())
        return failure();
      scf::SCFTilingOptions tileOption;
      SmallVector<OpFoldResult> TileSizes(
          genericOp.getNumLoops(),
          getAsIndexOpFoldResult(rewriter.getContext(), 0));
      if (useBlockedLayout) {
        TileSizes[MDimPos[0]] = getAsIndexOpFoldResult(
            rewriter.getContext(), (cfg.MBlock - 1) / cfg.innerMostMBlock + 1);
        TileSizes[NDimPos[0]] = getAsIndexOpFoldResult(
            rewriter.getContext(), (cfg.NBlock - 1) / cfg.innerMostNBlock + 1);
        TileSizes[KDimPos[0]] = getAsIndexOpFoldResult(
            rewriter.getContext(), (cfg.KBlock - 1) / cfg.innerMostKBlock + 1);
      } else {
        TileSizes[MDimPos[0]] =
            getAsIndexOpFoldResult(rewriter.getContext(), cfg.MBlock);
        TileSizes[NDimPos[0]] =
            getAsIndexOpFoldResult(rewriter.getContext(), cfg.NBlock);
        TileSizes[KDimPos[0]] =
            getAsIndexOpFoldResult(rewriter.getContext(), cfg.KBlock);
      }

      tileOption.setTileSizes(TileSizes);

      SmallVector<int64_t> interchange(genericOp.getNumLoops(), 0);
      for (auto i = 0UL; i < genericOp.getNumLoops(); i++) {
        interchange[i] = i;
      }
      tileOption.setInterchange(interchange);
      tileOption.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(genericOp);
      auto tilingResult = scf::tileUsingSCF(
          rewriter, cast<TilingInterface>(genericOp.getOperation()),
          tileOption);
      if (failed(tilingResult))
        return failure();
      rewriter.replaceOp(genericOp, tilingResult->replacements);
      genericOp = dyn_cast<linalg::LinalgOp>(tilingResult->tiledOps.back());
    }

    // 3.3 Tile innermost loop
    {
      // set tile size
      scf::SCFTilingOptions tileOption;
      SmallVector<OpFoldResult> TileSizes(
          genericOp.getNumLoops(),
          getAsIndexOpFoldResult(rewriter.getContext(), 0));
      auto iteratorTypes = genericOp.getIteratorTypesArray();
      TileSizes[MDimPos.back()] =
          getAsIndexOpFoldResult(rewriter.getContext(), cfg.innerMostMBlock);
      TileSizes[NDimPos.back()] =
          getAsIndexOpFoldResult(rewriter.getContext(), cfg.innerMostNBlock);
      if (!useBlockedLayout) {
        TileSizes[KDimPos.back()] =
            getAsIndexOpFoldResult(rewriter.getContext(), cfg.KBlock);
      }
      for (auto dim = 0; dim < genericOp.getNumLoops(); dim++) {
        if (dim != MDimPos.back() && dim != NDimPos.back() &&
            iteratorTypes[dim] != mlir::utils::IteratorType::reduction) {
          TileSizes[dim] = getAsIndexOpFoldResult(rewriter.getContext(), 1);
        }
      }
      tileOption.setTileSizes(TileSizes);
      tileOption.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

      // interchange loop order
      SmallVector<int64_t> interchange(genericOp.getNumLoops(), 0);
      for (auto i = 0UL; i < genericOp.getNumLoops(); i++) {
        interchange[i] = i;
      }
      tileOption.setInterchange(interchange);

      // do tiling
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(genericOp);
      auto tilingResult = scf::tileUsingSCF(
          rewriter, cast<TilingInterface>(genericOp.getOperation()),
          tileOption);
      if (failed(tilingResult))
        return failure();
      rewriter.replaceOp(genericOp, tilingResult->replacements);
      genericOp = dyn_cast<linalg::LinalgOp>(tilingResult->tiledOps.back());
    }

    // 3.4 inner loop generation, convert the linalg.generic to brgemm
    {
      // TODO: support the strided brgemm which will use two extra copy on
      // output
      // TODO: support plain in/block out, block in block out and vnni format
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(genericOp);

      // update the extractSlice to static size
      if (isa<tensor::ExtractSliceOp>(
              genericOp.getDpsInputs()[0].getDefiningOp())) {
        tensor::ExtractSliceOp extractSlice = dyn_cast<tensor::ExtractSliceOp>(
            *genericOp.getDpsInputs()[0].getDefiningOp());
        SmallVector<OpFoldResult> mixedOffsets = extractSlice.getMixedOffsets();
        SmallVector<OpFoldResult> mixedSizes = extractSlice.getMixedSizes();
        SmallVector<OpFoldResult> mixedStrides = extractSlice.getMixedStrides();
        SmallVector<int64_t> staticSize =
            useBlockedLayout
                ? SmallVector<int64_t>{1, cfg.KBlock / cfg.innerMostKBlock,
                                       cfg.innerMostMBlock, cfg.innerMostKBlock}
                : SmallVector<int64_t>{cfg.innerMostMBlock,
                                       cfg.KBlock / cfg.innerMostKBlock *
                                           cfg.innerMostKBlock};
        for (auto i = 0; i < mixedSizes.size(); i++) {
          mixedSizes[i] =
              getAsIndexOpFoldResult(rewriter.getContext(), staticSize[i]);
        }
        rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
            extractSlice, extractSlice.getSource(), mixedOffsets, mixedSizes,
            mixedStrides);
      }
      if (isa<tensor::ExtractSliceOp>(
              genericOp.getDpsInputs()[1].getDefiningOp())) {
        tensor::ExtractSliceOp extractSlice = dyn_cast<tensor::ExtractSliceOp>(
            *genericOp.getDpsInputs()[1].getDefiningOp());
        SmallVector<OpFoldResult> mixedOffsets = extractSlice.getMixedOffsets();
        SmallVector<OpFoldResult> mixedSizes = extractSlice.getMixedSizes();
        SmallVector<OpFoldResult> mixedStrides = extractSlice.getMixedStrides();
        SmallVector<int64_t> staticSize =
            useBlockedLayout
                ? SmallVector<int64_t>{1, cfg.KBlock / cfg.innerMostKBlock,
                                       cfg.innerMostKBlock, cfg.innerMostNBlock}
                : SmallVector<int64_t>{cfg.KBlock / cfg.innerMostKBlock *
                                           cfg.innerMostKBlock,
                                       cfg.innerMostNBlock};
        for (auto i = 0; i < mixedSizes.size(); i++) {
          mixedSizes[i] =
              getAsIndexOpFoldResult(rewriter.getContext(), staticSize[i]);
        }
        rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
            extractSlice, extractSlice.getSource(), mixedOffsets, mixedSizes,
            mixedStrides);
      }
      if (isa<tensor::ExtractSliceOp>(
              genericOp.getDpsInits()[0].getDefiningOp())) {
        tensor::ExtractSliceOp extractSlice = dyn_cast<tensor::ExtractSliceOp>(
            *genericOp.getDpsInits()[0].getDefiningOp());
        SmallVector<OpFoldResult> mixedOffsets = extractSlice.getMixedOffsets();
        SmallVector<OpFoldResult> mixedSizes = extractSlice.getMixedSizes();
        SmallVector<OpFoldResult> mixedStrides = extractSlice.getMixedStrides();
        SmallVector<int64_t> staticSize =
            useBlockedLayout ? SmallVector<int64_t>{1, 1, cfg.innerMostMBlock,
                                                    cfg.innerMostNBlock}
                             : SmallVector<int64_t>{cfg.innerMostMBlock,
                                                    cfg.innerMostNBlock};
        for (auto i = 0; i < mixedSizes.size(); i++) {
          mixedSizes[i] =
              getAsIndexOpFoldResult(rewriter.getContext(), staticSize[i]);
        }
        rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
            extractSlice, extractSlice.getSource(), mixedOffsets, mixedSizes,
            mixedStrides);
      }
      auto dataType = dyn_cast<mlir::RankedTensorType>(
          genericOp.getDpsInputs()[0].getType());
      auto weightType = dyn_cast<mlir::RankedTensorType>(
          genericOp.getDpsInputs()[1].getType());
      auto resultType = dyn_cast<mlir::RankedTensorType>(
          genericOp.getDpsInits()[0].getType());

      // View the tensor to brgemm required format
      Value dataOprand = tensorViewRankedTensor(
          rewriter,
          mlir::RankedTensorType::get(
              useBlockedLayout
                  ? SmallVector<int64_t>{cfg.KBlock / cfg.innerMostKBlock,
                                         cfg.innerMostMBlock,
                                         cfg.innerMostKBlock}
                  : SmallVector<int64_t>{1, cfg.innerMostMBlock,
                                         cfg.KBlock / cfg.innerMostKBlock *
                                             cfg.innerMostKBlock},
              dataType.getElementType()),
          genericOp.getDpsInputs()[0]);
      Value weightOprand = tensorViewRankedTensor(
          rewriter,
          mlir::RankedTensorType::get(
              useBlockedLayout
                  ? SmallVector<int64_t>{cfg.KBlock / cfg.innerMostKBlock,
                                         cfg.innerMostKBlock,
                                         cfg.innerMostNBlock}
                  : SmallVector<int64_t>{1,
                                         cfg.KBlock / cfg.innerMostKBlock *
                                             cfg.innerMostKBlock,
                                         cfg.innerMostNBlock},
              weightType.getElementType()),
          genericOp.getDpsInputs()[1]);
      Value resultOprand = tensorViewRankedTensor(
          rewriter,
          mlir::RankedTensorType::get(
              SmallVector<int64_t>{cfg.innerMostMBlock, cfg.innerMostNBlock},
              resultType.getElementType()),
          genericOp.getDpsInits()[0]);

      // Create the brgemm op
      linalg::LinalgOp matmul = rewriter.create<linalg::BatchReduceMatmulOp>(
          resultOprand.getLoc(), resultOprand.getType(),
          ValueRange{dataOprand, weightOprand}, resultOprand);
      Value result = matmul.getOperation()->getResult(0);
      // Insert the result back to the original tensor
      for (Operation *user : genericOp->getResult(0).getUsers()) {
        if (isa<tensor::InsertSliceOp>(user)) {
          tensor::InsertSliceOp insertSlice =
              dyn_cast<tensor::InsertSliceOp>(*user);
          SmallVector<OpFoldResult> mixedOffsets =
              insertSlice.getMixedOffsets();
          SmallVector<OpFoldResult> mixedSizes = insertSlice.getMixedSizes();
          SmallVector<OpFoldResult> mixedStrides =
              insertSlice.getMixedStrides();
          SmallVector<int64_t> staticSize =
              useBlockedLayout ? SmallVector<int64_t>{1, 1, cfg.innerMostMBlock,
                                                      cfg.innerMostNBlock}
                               : SmallVector<int64_t>{cfg.innerMostMBlock,
                                                      cfg.innerMostNBlock};
          for (auto i = 0; i < mixedSizes.size(); i++) {
            mixedSizes[i] =
                getAsIndexOpFoldResult(rewriter.getContext(), staticSize[i]);
          }
          rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
              insertSlice, result, insertSlice.getDest(), mixedOffsets,
              mixedSizes, mixedStrides);
        }
      }
      rewriter.replaceOp(genericOp, matmul.getOperation()->getResult(0));
      genericOp = matmul;
    }

    // 3.5 insert fill back
    {
      // TODO: support partial K in sinsngle threads, control flow may need easy
      // builder support
      auto initOp = genericOp.getDpsInits()[0].getDefiningOp();
      rewriter.setInsertionPointAfter(genericOp);
      auto fillOp = rewriter.create<linalg::FillOp>(initOp->getLoc(), fillValue,
                                                    genericOp.getDpsInits()[0]);
      IRMapping mapping;
      mapping.map(genericOp.getDpsInits()[0], fillOp.getResult(0));
      auto res = rewriter.clone(*(genericOp.getOperation()), mapping);
      rewriter.replaceOp(genericOp, res);
    }

    return success();
  }
};

struct RewriteMatmulToNestedMatmul
    : public tpp::impl::RewriteMatmulToNestedMatmulBase<
          RewriteMatmulToNestedMatmul> {
  using RewriteMatmulToNestedMatmulBase::RewriteMatmulToNestedMatmulBase;

  void runOnOperation() override {
    auto &ctx = getContext();

    RewritePatternSet patterns(&ctx);

    // Step 1. Recover Named Op
    linalg::populateLinalgDeGeneralizationPatterns(patterns);

    // Step 2. Rewrite matmul to nested matmul
    patterns.add<rewriteToNestedMatmul<linalg::MatmulOp>,
                 rewriteToNestedMatmul<linalg::BatchMatmulOp>>(
        patterns.getContext());
    linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
    linalg::ControlDropUnitDims options;
    options.rankReductionStrategy =
        linalg::ControlDropUnitDims::RankReductionStrategy::ExtractInsertSlice;
    linalg::populateFoldUnitExtentDimsPatterns(patterns, options);
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // end namespace

// Example for build a for loop
// {
//     {
//         OpBuilder::InsertionGuard guard(rewriter);
//         rewriter.setInsertionPointAfter(matmulOp);
//         _EasyBuilderBegin_(rewriter, matmulOp.getLoc());
//         scf::ForOp loop1;
//         _SCFFor_(loop1, 0, MThreads, 1,
//         ValueRange{matmulOp.getResult(0)})
//         {
//             auto pm = loop1.getInductionVar();
//             Value lbCst = _BuildOp_(arith::ConstantIndexOp, 1);
//             auto cmpI = _BuildOp_(arith::CmpIOp,
//             arith::CmpIPredicate::slt, lbCst, pm); scf::IfOp ifHandle;
//             _SCFIf_(ifHandle, cmpI, matmulOp.getResult(0).getType())
//             {
//                 Value lbCst = _BuildOp_(arith::ConstantIndexOp, 2);
//                 _SCFYield_(matmulOp.getResult(0));
//             }
//             _SCFElse_
//             {
//                 Value lbCst = _BuildOp_(arith::ConstantIndexOp, 1);
//                 _SCFYield_(matmulOp.getResult(0));
//             }
//             _SCFYield_(ifHandle.getResult(0));
//             loop1.dump();
//         }

//         scf::ForallOp newLoop;
//         _SCFForall_(newLoop, SmallVector<int64_t>{0, 0, 0},
//         SmallVector<int64_t>{MThreads, NThreads, KThreads},
//         SmallVector<int64_t>{1, 1, 1}, std::nullopt, std::nullopt)
//         {
//             [[maybe_unused]] Value lbCst =
//             _BuildOp_(arith::ConstantIndexOp, 1);
//             _SCFInParallel_(newLoop){
//                 Value lbCst = _BuildOp_(arith::ConstantIndexOp, 1);
//             }
//         }
//         newLoop.dump();
//         _EasyBuilderEnd_;
//     }
//     matmulOp.getOperation()->getParentOfType<ModuleOp>().dump();
//     exit(0);
// }

/*
// Example: register external function
{
  auto mod = matmulOp.getOperation()->getParentOfType<ModuleOp>();
  ParserConfig cfg(&ctx, true, nullptr);
  parseSourceString(
"func.func @matmul2() -> tensor<256x512xf32> {\
%3 = tensor.empty() : tensor<256x512xf32>\
return %3 : tensor<256x512xf32>\
}",
      &(mod.getOperation()->getRegion(0).getBlocks().front()), cfg);
  FlatSymbolRefAttr fnName =
      SymbolRefAttr::get(rewriter.getContext(), "matmul2");
  auto loc = matmulOp.getOperation()->getLoc();
  auto libFnType = parseType("() -> tensor<256x512xf32>", &ctx);
  auto oprand = SmallVector<Value>();
  func::CallOp call = rewriter.create<func::CallOp>(
      loc, fnName.getValue(), TypeRange(), oprand);
  call.dump();
  exit(0);
}*/

/*if (cfg.KThreads > 1)
{
    if (genericOp.hasPureBufferSemantics())
        return signalPassFailure();
    SmallVector<OpFoldResult> tileSize(
        genericOp.getNumLoops(),
        getAsIndexOpFoldResult(rewriter.getContext(), 0));
    tileSize[KDimPos[0]] =
        getAsIndexOpFoldResult(rewriter.getContext(), cfg.KThreads );
    bool isFirstReductionDim = true;
    for (auto reductionDim : KDimPos)
    {
        if (isFirstReductionDim)
        {
            tileSize[reductionDim] =
                getAsIndexOpFoldResult(rewriter.getContext(), cfg.KThreads);
            isFirstReductionDim = false;
        }
        else
        {
            tileSize[reductionDim] =
                getAsIndexOpFoldResult(rewriter.getContext(), 1);
        }
    }

    rewriter.setInsertionPoint(genericOp);
    auto tilingResult = scf::tileReductionUsingScf(
        rewriter,
        cast<PartialReductionOpInterface>(genericOp.getOperation()),
        tileSize);
    if (failed(tilingResult))
        return signalPassFailure();
    genericOp =
        dyn_cast<linalg::GenericOp>(tilingResult->parallelTiledOp);
}*/
