//===- RewriteConvToMatmul.cpp -----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/EasyBuilder.h"
#include "TPP/Transforms/Utils/TransformUtils.h"
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

[[maybe_unused]] MatmulConfig
getMatmulConfig(const linalg::MatmulOp &matmulOp) {
  return {32, 32, 32, 4, 14, 1, 8, 8, 8};
}

/*
for([PM, PN]: [MThreads, NThreads) {
  CSlice = [KThreads, PM * MOuterBlock: (PM + 1) * MOuterBlock, PN *
NOuterBlock: (PN + 1) * NOuterBlock] (init with 0) for(PK : KThreads) {
    MOuterBlock = 256 / MThreads
    ...
    ASlice = A[PM * MOuterBlock: (PM + 1) * MOuterBlock, PK * KOuterBlock * (PK
+ 1) * KOuterBlock] BSlice = B[PK * KOuterBlock * (PK + 1) * KOuterBlock, PN *
NOuterBlock: (PN + 1) * NOuterBlock] CSlice_2 = CSlice[PK, PM * MOuterBlock: (PM
+ 1) * MOuterBlock, PN * NOuterBlock: (PN + 1) * NOuterBlock] CSlice_2 =
matmul(ASlice, BSlice)
  }
  final_reduce(CSlice)
}

for([PM, PN]: [MThreads, NThreads) {
  CSlice = [KThreads, PM * MOuterBlock: (PM + 1) * MOuterBlock, PN *
NOuterBlock: (PN + 1) * NOuterBlock] for(PK : KThreads) { MOuterBlock = 256 /
MThreads
    ...
    ASlice = A[PM * MOuterBlock: (PM + 1) * MOuterBlock, PK * KOuterBlock * (PK
+ 1) * KOuterBlock] BSlice = B[PK * KOuterBlock * (PK + 1) * KOuterBlock, PN *
NOuterBlock: (PN + 1) * NOuterBlock] CSlice2 = CSlice[PK, PM * MOuterBlock: (PM
+ 1) * MOuterBlock, PN * NOuterBlock: (PN + 1) * NOuterBlock]

    MNumBlock = MOuterBlock / MBlock
    NNumBlock = NOuterBlock / NBlock
    KNumBlock = KOuterBlock / KBlovk
    for([om, on, ok]: [MNumBlock, NNumBlock, KNumBlock]) {
      ASlice2 = ASlice[om * MBlock: (om + 1) * MBlock, ok * KBlock: (ok + 1) *
KBlock] BSlice2 = BSlice[0, om * MBlock: (om + 1) * MBlock, ok * KBlock: (ok +
1) * KBlock] CSlice3 = CSlice2[0, om * MBlock: (om + 1) * MBlock, on * NBlock:
(on + 1) * NBlock] (init with 0 when ok == 0) MNumInnerBlock = MBlock /
iim_block_
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
  final_reduce(CSlice)
}
*/
struct RewriteMatmulToNestedMatmul
    : public tpp::impl::RewriteMatmulToNestedMatmulBase<
          RewriteMatmulToNestedMatmul> {
  using RewriteMatmulToNestedMatmulBase::RewriteMatmulToNestedMatmulBase;

  void runOnOperation() override {
    auto &ctx = getContext();
    IRRewriter rewriter(&ctx);
    // Step 1. Recover Named Op
    {
      // Attempt to recover named ops.
      RewritePatternSet patterns(&ctx);
      linalg::populateLinalgDeGeneralizationPatterns(patterns);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    // Pack matmul, will be replaced by blocked tensor dialect pass
    auto MBlock = 32, NBlock = 32, KBlock = 32;
    auto MThreads = 3, NThreads = 3, KThreads = 1;
    auto innerMostMBlock = 8, innerMostNBlock = 8, innerMostKBlock = 8;

    // Step 2. Pack to Mkmk(innerMostMBlock, innerMostKBlock) amd
    // NKkn(inermostKBlock, innermostNBlock)
    getOperation()->walk([&](linalg::MatmulOp matmulOp) {
      if (matmulOp.hasPureBufferSemantics())
        return signalPassFailure();
      SmallVector<OpFoldResult> packSizes(
          matmulOp.getNumLoops(),
          getAsIndexOpFoldResult(rewriter.getContext(), 0));
      packSizes[0] =
          getAsIndexOpFoldResult(rewriter.getContext(), innerMostMBlock);
      packSizes[1] =
          getAsIndexOpFoldResult(rewriter.getContext(), innerMostNBlock);
      packSizes[2] =
          getAsIndexOpFoldResult(rewriter.getContext(), innerMostKBlock);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(matmulOp);
      [[maybe_unused]] auto res =
          mlir::linalgx::packMatmulOp(rewriter, matmulOp, packSizes);
    });

    // Step 3. The processes of transforming matmul to nested matmul
    getOperation()->walk([&](linalg::GenericOp linalgOp) {
      // // TODO: check whether the op is matmul or packed matmul op
      // auto M = linalgOp.getShape(linalgOp.getDpsInputOperand(0))[0],
      //      N = linalgOp.getShape(linalgOp.getDpsInputOperand(1))[1],
      //      K = linalgOp.getShape(linalgOp.getDpsInputOperand(1))[0];
      if (linalgOp.hasPureBufferSemantics())
        return signalPassFailure();
      auto matmulOp = linalgOp;
      // 3.1 Parallel Loop with scf::forall
      {
        SmallVector<OpFoldResult> numThreads(
            matmulOp.getNumLoops(),
            getAsIndexOpFoldResult(rewriter.getContext(), 0));
        numThreads[0] = getAsIndexOpFoldResult(rewriter.getContext(), MThreads);
        numThreads[1] = getAsIndexOpFoldResult(rewriter.getContext(), NThreads);
        numThreads[2] = getAsIndexOpFoldResult(rewriter.getContext(), 0);
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(matmulOp);
        auto tilingResult = linalg::tileToForallOp(
            rewriter, cast<TilingInterface>(matmulOp.getOperation()),
            numThreads,
            /*mapping=*/std::nullopt);
        if (failed(tilingResult))
          return signalPassFailure();
        rewriter.replaceOp(matmulOp, tilingResult->tileOp->getResults());
        matmulOp = dyn_cast<linalg::GenericOp>(tilingResult->tiledOp);
      }

      // 3.2 Tiling reduction parallel loop with scf::forall
      // TODO: move the reduction dim to the front. (M, N, threads) ->
      // (threads, M, N)
      {
        if (KThreads > 1) {
          if (matmulOp.hasPureBufferSemantics())
            return signalPassFailure();
          SmallVector<OpFoldResult> tileSize(
              matmulOp.getNumLoops(),
              getAsIndexOpFoldResult(rewriter.getContext(), 0));
          tileSize[2] = getAsIndexOpFoldResult(rewriter.getContext(), KThreads);
          tileSize[5] = getAsIndexOpFoldResult(rewriter.getContext(), 1);

          rewriter.setInsertionPoint(matmulOp);
          auto tilingResult = scf::tileReductionUsingScf(
              rewriter,
              cast<PartialReductionOpInterface>(matmulOp.getOperation()),
              tileSize);
          if (failed(tilingResult))
            return signalPassFailure();
          matmulOp = dyn_cast<linalg::GenericOp>(tilingResult->parallelTiledOp);
        }
      }

      // 3.3 Tiling outer loop with scf::for
      {
        if (matmulOp.hasPureBufferSemantics())
          return signalPassFailure();
        scf::SCFTilingOptions tileOption;
        SmallVector<OpFoldResult> TileSizes(
            matmulOp.getNumLoops(),
            getAsIndexOpFoldResult(rewriter.getContext(), 0));
        TileSizes[0] = getAsIndexOpFoldResult(
            rewriter.getContext(), (MBlock - 1) / innerMostMBlock + 1);
        TileSizes[1] = getAsIndexOpFoldResult(
            rewriter.getContext(), (NBlock - 1) / innerMostNBlock + 1);
        TileSizes[2] = getAsIndexOpFoldResult(
            rewriter.getContext(), (KBlock - 1) / innerMostKBlock + 1);
        tileOption.setTileSizes(TileSizes);

        SmallVector<int64_t> interchange(matmulOp.getNumLoops(), 0);
        for (auto i = 0UL; i < matmulOp.getNumLoops(); i++) {
          interchange[i] = i;
        }
        tileOption.setInterchange(interchange);

        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(matmulOp);
        auto tilingResult = scf::tileUsingSCF(
            rewriter, cast<TilingInterface>(matmulOp.getOperation()),
            tileOption);
        if (failed(tilingResult))
          return signalPassFailure();
        rewriter.replaceOp(matmulOp, tilingResult->replacements);
        matmulOp = dyn_cast<linalg::GenericOp>(tilingResult->tiledOps.back());
      }

      // 3.4 Tile innermost loop
      {
        scf::SCFTilingOptions tileOption;
        SmallVector<OpFoldResult> TileSizes(
            matmulOp.getNumLoops(),
            getAsIndexOpFoldResult(rewriter.getContext(), 0));
        TileSizes[0] = getAsIndexOpFoldResult(rewriter.getContext(), 1);
        TileSizes[1] = getAsIndexOpFoldResult(rewriter.getContext(), 1);
        tileOption.setTileSizes(TileSizes);

        SmallVector<int64_t> interchange(matmulOp.getNumLoops(), 0);
        for (auto i = 0UL; i < matmulOp.getNumLoops(); i++) {
          interchange[i] = i;
        }
        tileOption.setInterchange(interchange);

        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(matmulOp);
        auto tilingResult = scf::tileUsingSCF(
            rewriter, cast<TilingInterface>(matmulOp.getOperation()),
            tileOption);
        if (failed(tilingResult))
          return signalPassFailure();
        rewriter.replaceOp(matmulOp, tilingResult->replacements);
        matmulOp = dyn_cast<linalg::GenericOp>(tilingResult->tiledOps.back());
      }

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
      auto mod = matmulOp.getOperation()->getParentOfType<ModuleOp>();
      mod.dump();
    });

    {
      // Step 4:
      // - replace extract/insert slice with ranked reduced extract/insert slice
      // and expand shape ops.
      RewritePatternSet patterns(&ctx);
      linalg::ControlDropUnitDims options;
      options.rankReductionStrategy = linalg::ControlDropUnitDims::
          RankReductionStrategy::ExtractInsertSlice;
      linalg::populateFoldUnitExtentDimsPatterns(patterns, options);
      linalg::populateLinalgDeGeneralizationPatterns(patterns);
      linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
      tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
      ctx.getOrLoadDialect<tensor::TensorDialect>()
          ->getCanonicalizationPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

} // end namespace
