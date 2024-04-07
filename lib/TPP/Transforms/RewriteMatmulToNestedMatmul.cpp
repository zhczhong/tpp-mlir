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

MatmulConfig getDefaultMatmulConfig(linalg::MatmulOp &linalgOp) {
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
  cfg.MThreads = 2;
  cfg.NThreads = 2;
  cfg.KThreads = 2;

  // Block
  cfg.MBlock = divAndCeil((int)MNumBlock, cfg.MThreads) * cfg.innerMostMBlock;
  cfg.NBlock = divAndCeil((int)NNumBlock, cfg.NThreads) * cfg.innerMostNBlock;
  cfg.KBlock = divAndCeil((int)KNumBlock, cfg.KThreads) * cfg.innerMostKBlock;
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
struct rewriteToNestedMatmul : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

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

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (matmulOp.hasPureBufferSemantics())
      return failure();
    MatmulConfig cfg = getDefaultMatmulConfig(matmulOp);
    linalg::LinalgOp genericOp;
    bool useBlockedLayout = false;
    // Step 2. Pack to Mkmk(innerMostMBlock, innerMostKBlock) amd
    // NKkn(inermostKBlock, innermostNBlock)
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

    // Step 3. The processes of transforming matmul to nested matmul
    // 3.0 Get the iteration infomation first
    SmallVector<unsigned> KDimPos, MDimPos, NDimPos;
    genericOp.getReductionDims(KDimPos);
    getMatmulParallelDims(genericOp, 0, MDimPos);
    getMatmulParallelDims(genericOp, 1, NDimPos);

    // 3.1 Parallel Loop with scf::forall
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(genericOp);
      SmallVector<OpFoldResult> tiles(
          genericOp.getNumLoops(),
          getAsIndexOpFoldResult(rewriter.getContext(), 0));
      auto MFirstDim = (int)getOprandDim(genericOp, MDimPos[0], 0);
      auto NFirstDim = (int)getOprandDim(genericOp, NDimPos[0], 1);
      tiles[MDimPos[0]] = getAsIndexOpFoldResult(
          rewriter.getContext(),
          useBlockedLayout
              ? divAndCeil(MFirstDim, cfg.MThreads)
              : divAndCeil(divAndCeil(MFirstDim, cfg.MBlock), cfg.MThreads) *
                    cfg.MBlock);
      tiles[NDimPos[0]] = getAsIndexOpFoldResult(
          rewriter.getContext(),
          useBlockedLayout
              ? divAndCeil(NFirstDim, cfg.NThreads)
              : divAndCeil(divAndCeil(NFirstDim, cfg.NBlock), cfg.NThreads) *
                    cfg.NBlock);
      auto tilingResult = linalg::tileToForallOpUsingTileSizes(
          rewriter, cast<TilingInterface>(genericOp.getOperation()), tiles,
          /*mapping=*/std::nullopt);
      if (failed(tilingResult))
        return failure();
      rewriter.replaceOp(genericOp, tilingResult->tileOp->getResults());
      genericOp = dyn_cast<linalg::LinalgOp>(tilingResult->tiledOp);
    }

    // 3.2 Tiling reduction parallel loop with scf::forall
    // TODO: move the reduction dim to the front. (M, N, threads) ->
    // (threads, M, N)
    {
      // TODO: support more than one reduction dim
      if (cfg.KThreads > 1 && !useBlockedLayout) {
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
            threads[reductionDim] =
                getAsIndexOpFoldResult(rewriter.getContext(), cfg.KThreads);
            isFirstReductionDim = false;
          } else {
            tileSizes[reductionDim] =
                getAsIndexOpFoldResult(rewriter.getContext(), 1);
            threads[reductionDim] =
                getAsIndexOpFoldResult(rewriter.getContext(), 1);
          }
        }

        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(genericOp);
        auto tilingResult = linalg::tileReductionUsingForall(
            rewriter,
            cast<PartialReductionOpInterface>(genericOp.getOperation()),
            threads, tileSizes);
        if (failed(tilingResult))
          return failure();
        genericOp = dyn_cast<linalg::LinalgOp>(tilingResult->parallelTiledOp);
      }
    }

    // 3.3 Tiling outer loop with scf::for
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

    // 3.4 Tile innermost loop
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

    // 3.5 inner loop generation, convert the linalg.generic to brgemm
    {
      // TODO: support the strided brgemm which will use two extra copy on
      // output
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(genericOp);
      auto dataType = dyn_cast<mlir::RankedTensorType>(
          genericOp.getDpsInputs()[0].getType());
      auto weightType = dyn_cast<mlir::RankedTensorType>(
          genericOp.getDpsInputs()[1].getType());
      auto resultType = dyn_cast<mlir::RankedTensorType>(
          genericOp.getDpsInits()[0].getType());
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
      linalg::LinalgOp matmul = rewriter.create<linalg::BatchReduceMatmulOp>(
          resultOprand.getLoc(), resultOprand.getType(),
          ValueRange{dataOprand, weightOprand}, resultOprand);
      Value result = tensorViewRankedTensor(
          rewriter, resultType, matmul.getOperation()->getResult(0));
      rewriter.replaceOp(genericOp, result);
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
    patterns.add<rewriteToNestedMatmul>(patterns.getContext());

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
