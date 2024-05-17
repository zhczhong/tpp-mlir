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

  // cfg.innerMostMBlock = 32;
  // cfg.innerMostNBlock = 32;
  // cfg.innerMostKBlock = 32;
  // cfg.MBlock = 64;
  // cfg.NBlock = 64;
  // cfg.KBlock = 64;
  // cfg.MThreads = 2;
  // cfg.NThreads = 2;
  // cfg.KThreads = 1;
  return cfg;
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

struct OuterLoopGenerationOption {
  enum LoopType { ForOp, ForallOp };
  SmallVector<SmallVector<int>> nestedTileSizes;
  SmallVector<LoopType> loopType;
  SmallVector<SmallVector<int>> loopDim;
};

struct OuterLoopGenerationResult {
  /// Tiled operations that are generated during tiling. The order does not
  /// matter except the last op. The replacements are expected to be the results
  /// of the last op.
  SmallVector<Operation *> tiledOps;
  /// The `scf.for` operations that iterate over the tiles.
  SmallVector<LoopLikeOpInterface> loops;
  /// Values to use as replacements for the untiled op. Is the same size as the
  /// number of results of the untiled op.
  SmallVector<Value> replacements;
};

FailureOr<OuterLoopGenerationResult>
generateOuterLoop(RewriterBase &b, linalg::LinalgOp linalgOp,
                  OuterLoopGenerationOption option) {
  OuterLoopGenerationResult result;
  auto nestedTileSizes = option.nestedTileSizes;
  auto loopType = option.loopType;
  auto loopDim = option.loopDim;

  if (loopType.size() != loopDim.size() ||
      loopDim.size() != nestedTileSizes.size()) {
    return b.notifyMatchFailure(
        linalgOp,
        "loopType, loopDim and nestedTileSizes should have the same size");
  }

  if (linalgOp.hasPureBufferSemantics())
    return b.notifyMatchFailure(
        linalgOp, "currentOp should not has pure buffer semantics");

  linalg::LinalgOp currentOp = linalgOp;
  for (auto iteratorType : llvm::enumerate(loopType)) {
    auto [i, type] = iteratorType;
    auto currentDim = loopDim[i];
    auto currentTileSize = nestedTileSizes[i];
    if (type == OuterLoopGenerationOption::LoopType::ForOp) {
      scf::SCFTilingOptions tileOption;
      SmallVector<OpFoldResult> TileSizes(
          currentOp.getNumLoops(), getAsIndexOpFoldResult(b.getContext(), 0));

      for (auto [d, tile] : llvm::zip(currentDim, currentTileSize)) {
        TileSizes[d] = getAsIndexOpFoldResult(b.getContext(), tile);
      }
      tileOption.setTileSizes(TileSizes);
      tileOption.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPoint(currentOp);
      auto tilingResult = scf::tileUsingSCF(
          b, cast<TilingInterface>(currentOp.getOperation()), tileOption);
      if (failed(tilingResult))
        return failure();
      b.replaceOp(currentOp, tilingResult->replacements);
      currentOp = dyn_cast<linalg::LinalgOp>(tilingResult->tiledOps.back());
    } else if (type == OuterLoopGenerationOption::LoopType::ForallOp) {
      SmallVector<OpFoldResult> tileSizes(
          currentOp.getNumLoops(), getAsIndexOpFoldResult(b.getContext(), 0));
      SmallVector<OpFoldResult> threads(
          currentOp.getNumLoops(), getAsIndexOpFoldResult(b.getContext(), 0));
      SmallVector<unsigned> reductionDims;
      currentOp.getReductionDims(reductionDims);
      for (auto [d, tile] : llvm::zip(currentDim, currentTileSize)) {
        if (llvm::find(reductionDims, d) != reductionDims.end() &&
            !dyn_cast<PartialReductionOpInterface>(currentOp.getOperation()))
          tileSizes[d] = getAsIndexOpFoldResult(b.getContext(), 0);
        else
          tileSizes[d] = getAsIndexOpFoldResult(b.getContext(), tile);
      }

      SmallVector<OpFoldResult> numThreads;
      SmallVector<Range> loopRanges =
          cast<TilingInterface>(currentOp.getOperation()).getIterationDomain(b);
      unsigned nLoops = loopRanges.size();
      numThreads.reserve(nLoops);
      AffineExpr s0, s1;
      bindSymbols(b.getContext(), s0, s1);
      AffineExpr divExpr = s0.ceilDiv(s1);
      for (const auto &it : llvm::zip(tileSizes, loopRanges)) {
        OpFoldResult numTiles = std::get<0>(it);
        if (!isConstantIntValue(numTiles, 0))
          numTiles = makeComposedFoldedAffineApply(
              b, currentOp.getLoc(), divExpr,
              {std::get<1>(it).size, std::get<0>(it)});
        numThreads.push_back(numTiles);
      }

      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPoint(currentOp);
      if (auto partialInterface =
              dyn_cast<PartialReductionOpInterface>(currentOp.getOperation())) {
        auto tilingResult = linalg::tileAllUsingForall(
            b, cast<PartialReductionOpInterface>(currentOp.getOperation()),
            numThreads, tileSizes, std::nullopt);
        if (failed(tilingResult))
          return failure();
        currentOp = dyn_cast<linalg::LinalgOp>(tilingResult->parallelTiledOp);
      } else if (auto tilingInterface =
                     cast<TilingInterface>(currentOp.getOperation())) {
        auto tilingResult = linalg::tileToForallOpUsingTileSizes(
            b, tilingInterface, tileSizes, std::nullopt);
        if (failed(tilingResult))
          return failure();
        b.replaceOp(currentOp, tilingResult->tileOp);
        currentOp = dyn_cast<linalg::LinalgOp>(tilingResult->tiledOp);
      }
    }
  }
  result.tiledOps.emplace_back(currentOp);
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
  static_assert(
      llvm::is_one_of<LinalgOpTy, linalg::MatmulOp, linalg::BatchMatmulOp,
                      linalg::GenericOp>::value);

  void getMatmulParallelDims(linalg::LinalgOp linalgOp, unsigned operandIdx,
                             SmallVectorImpl<unsigned> &dims) const {
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
    static int cnt = 0;
    if (matmulOp.hasPureBufferSemantics())
      return failure();
    MatmulConfig cfg = getDefaultMatmulConfig(matmulOp);
    linalg::LinalgOp genericOp;
    genericOp = dyn_cast<linalg::LinalgOp>(matmulOp.getOperation());
    if (genericOp.getOperation()->getParentOfType<scf::ForallOp>())
      return failure();

    // Step 2. Match and remove the init/fill operation
    // Fuse the fill op manually before fusion support this case(fuse it into
    // if-else block)
    // bool hasFillOp = false;
    // Value fillValue;
    // SmallVector<LoopLikeOpInterface> KLoopHandle;
    // if (auto op = dyn_cast<linalg::FillOp>(
    //         genericOp.getDpsInits()[0].getDefiningOp())) {
    //   hasFillOp = true;
    //   fillValue = op.getDpsInputs()[0];
    //   rewriter.replaceOp(op, op.getDpsInits()[0]);
    // }

    // Step 3. The processes of transforming matmul to nested matmul
    // 3.0 Get the iteration infomation first
    SmallVector<unsigned> KDimPos, MDimPos, NDimPos;
    genericOp.getReductionDims(KDimPos);
    getMatmulParallelDims(genericOp, 0, MDimPos);
    getMatmulParallelDims(genericOp, 1, NDimPos);
    bool useBlockedLayout = KDimPos.size() > 1;

    // 3.1 Outer Loop Generation
    // TODO: move the reduction dim to the front. (M, N, threads) ->
    // (threads, M, N)
    {
      OuterLoopGenerationOption option;
      auto iteratorTypes = genericOp.getIteratorTypesArray();
      auto KFirstDim = (int)getOprandDim(genericOp, KDimPos[0], 1);
      auto MFirstDim = (int)getOprandDim(genericOp, MDimPos[0], 0);
      auto NFirstDim = (int)getOprandDim(genericOp, NDimPos[0], 1);
      auto KParallelBlockSize =
          useBlockedLayout
              ? divAndCeil(KFirstDim, cfg.KThreads)
              : divAndCeil(divAndCeil(KFirstDim, cfg.KBlock), cfg.KThreads) *
                    cfg.KBlock;
      auto MParallelBlockSize =
          useBlockedLayout
              ? divAndCeil(MFirstDim, cfg.MThreads)
              : divAndCeil(divAndCeil(MFirstDim, cfg.MBlock), cfg.MThreads) *
                    cfg.MBlock;
      auto NParallelBlockSize =
          useBlockedLayout
              ? divAndCeil(NFirstDim, cfg.NThreads)
              : divAndCeil(divAndCeil(NFirstDim, cfg.NBlock), cfg.NThreads) *
                    cfg.NBlock;
      auto KOuterBlockSize = useBlockedLayout
                                 ? (cfg.KBlock - 1) / cfg.innerMostKBlock + 1
                                 : cfg.KBlock;
      auto MOuterBlockSize = useBlockedLayout
                                 ? (cfg.MBlock - 1) / cfg.innerMostMBlock + 1
                                 : cfg.MBlock;
      auto NOuterBlockSize = useBlockedLayout
                                 ? (cfg.NBlock - 1) / cfg.innerMostNBlock + 1
                                 : cfg.NBlock;

      // Outer
      option.nestedTileSizes.emplace_back(SmallVector<int>{
          MParallelBlockSize, NParallelBlockSize, KParallelBlockSize});
      option.loopType.emplace_back(
          OuterLoopGenerationOption::LoopType::ForallOp);
      option.loopDim.emplace_back(
          SmallVector<int>{MDimPos[0], NDimPos[0], KDimPos[0]});
      // Middle
      for (auto [tile, dim] :
           llvm::zip(SmallVector<int>{MOuterBlockSize, NOuterBlockSize,
                                      KOuterBlockSize},
                     SmallVector<int>{MDimPos[0], NDimPos[0], KDimPos[0]})) {
        option.nestedTileSizes.emplace_back(SmallVector<int>{tile});
        option.loopType.emplace_back(
            OuterLoopGenerationOption::LoopType::ForOp);
        option.loopDim.emplace_back(SmallVector<int>{dim});
      }
      // Inner
      if (!useBlockedLayout) {
        option.nestedTileSizes.emplace_back(SmallVector<int>{cfg.KBlock});
        option.loopType.emplace_back(
            OuterLoopGenerationOption::LoopType::ForOp);
        option.loopDim.emplace_back(SmallVector<int>{KDimPos.back()});
      }
      for (auto dim = 0; dim < genericOp.getNumLoops(); dim++) {
        if (dim != MDimPos.back() && dim != NDimPos.back() &&
            iteratorTypes[dim] != mlir::utils::IteratorType::reduction) {
          option.nestedTileSizes.emplace_back(SmallVector<int>{1});
          option.loopType.emplace_back(
              OuterLoopGenerationOption::LoopType::ForOp);
          option.loopDim.emplace_back(SmallVector<int>{dim});
        }
      }
      auto tilingResult = generateOuterLoop(rewriter, genericOp, option);
      genericOp = dyn_cast<linalg::LinalgOp>(tilingResult->tiledOps.back());
    }

    // 3.2 inner loop generation, convert the linalg.generic to brgemm
    if (KDimPos.size() == 1) {
      genericOp.getOperation()->getParentOfType<func::FuncOp>().dump();
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
    // {
    //   // TODO: support partial K in sinsngle threads, control flow may need
    //   easy
    //   // builder support
    //   auto initOp = genericOp.getDpsInits()[0].getDefiningOp();
    //   rewriter.setInsertionPointAfter(genericOp);
    //   auto fillOp = rewriter.create<linalg::FillOp>(initOp->getLoc(),
    //   fillValue,
    //                                                 genericOp.getDpsInits()[0]);
    //   IRMapping mapping;
    //   mapping.map(genericOp.getDpsInits()[0], fillOp.getResult(0));
    //   auto res = rewriter.clone(*(genericOp.getOperation()), mapping);
    //   rewriter.replaceOp(genericOp, res);
    // }
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
                 rewriteToNestedMatmul<linalg::BatchMatmulOp>,
                 rewriteToNestedMatmul<linalg::GenericOp>>(
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
