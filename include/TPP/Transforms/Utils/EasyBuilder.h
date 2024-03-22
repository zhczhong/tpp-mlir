//===- Builder Utils - Helper for builder patterns ------------------------===//
// Utilities to help build MLIR
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TRANSFORMS_UTILS_EASYBUILDER_H
#define TPP_TRANSFORMS_UTILS_EASYBUILDER_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace mlir {

/**
 * This class builds a for-loop node with RAII of C++.
 * It provides an iterator which can only iterate once.
 * Users should use range-based-for to iterate on this iterator
 * The iterator returns the "var expr" of the for-loop to generate
 * e.g.
 * for (auto i: range(0, 100, "i")) {
 *  buf[i] = buf[i] + 1;
 * }
 *
 * At the end of the scope, this class will push a for-loop node in current
 * build
 * */
struct SCFForRangeSimulator {
  scf::ForOp loopOp_;
  IRRewriter::InsertPoint insertPoint_;
  OpBuilder *rewriter_;
  struct ForRangeIterator {
    scf::ForOp loopOp_;
    bool consumed_;
    scf::ForOp operator*() const { return loopOp_; }

    ForRangeIterator &operator++() {
      consumed_ = true;
      return *this;
    }

    bool operator!=(ForRangeIterator &other) const {
      return consumed_ != other.consumed_;
    }

    ForRangeIterator(scf::ForOp loopOp_)
        : loopOp_(std::move(loopOp_)), consumed_(false) {}
    ForRangeIterator() : consumed_(true) {}
  };

  ForRangeIterator begin() const { return ForRangeIterator(loopOp_); }

  ForRangeIterator end() { return ForRangeIterator(); }

  SCFForRangeSimulator(OpBuilder *rewriter, Location loc, scf::ForOp &bindLoop,
                       int lb, int ub, int step = 1,
                       ValueRange iterArgs = std::nullopt)
      : rewriter_(rewriter) {
    insertPoint_ = rewriter->saveInsertionPoint();
    Value lbCst = rewriter->create<arith::ConstantIndexOp>(loc, lb);
    Value ubCst = rewriter->create<arith::ConstantIndexOp>(loc, ub);
    Value stepCst = rewriter->create<arith::ConstantIndexOp>(loc, step);
    scf::ForOp loopOp =
        rewriter->create<scf::ForOp>(loc, lbCst, ubCst, stepCst, iterArgs);
    rewriter->setInsertionPointToStart(loopOp.getBody());
    loopOp_ = loopOp;
    bindLoop = loopOp;
    rewriter_ = rewriter;
  }

  SCFForRangeSimulator(OpBuilder *rewriter, Location loc, scf::ForOp &bindLoop,
                       Value lb, Value ub, Value step,
                       ValueRange iterArgs = std::nullopt) {
    insertPoint_ = rewriter->saveInsertionPoint();
    scf::ForOp loopOp =
        rewriter->create<scf::ForOp>(loc, lb, ub, step, iterArgs);
    rewriter->setInsertionPointToStart(loopOp.getBody());
    loopOp_ = loopOp;
    bindLoop = loopOp;
    rewriter_ = rewriter;
  }

  SCFForRangeSimulator(const SCFForRangeSimulator &other) = delete;
  SCFForRangeSimulator(SCFForRangeSimulator &&other)
      : loopOp_(std::move(other.loopOp_)),
        insertPoint_(std::move(other.insertPoint_)),
        rewriter_(std::move(other.rewriter_)) {}

  ~SCFForRangeSimulator() { rewriter_->restoreInsertionPoint(insertPoint_); }
};

struct SCFForallRangeSimulator {
  scf::ForallOp loopOp_;
  IRRewriter::InsertPoint insertPoint_;
  OpBuilder *rewriter_;
  struct ForallRangeIterator {
    scf::ForallOp loopOp_;
    bool consumed_;
    scf::ForallOp operator*() const { return loopOp_; }

    ForallRangeIterator &operator++() {
      consumed_ = true;
      return *this;
    }

    bool operator!=(ForallRangeIterator &other) const {
      return consumed_ != other.consumed_;
    }

    ForallRangeIterator(scf::ForallOp loopOp_)
        : loopOp_(std::move(loopOp_)), consumed_(false) {}
    ForallRangeIterator() : consumed_(true) {}
  };

  ForallRangeIterator begin() const { return ForallRangeIterator(loopOp_); }

  ForallRangeIterator end() { return ForallRangeIterator(); }

  SCFForallRangeSimulator(OpBuilder *rewriter, Location loc,
                          scf::ForallOp &bindLoop, ArrayRef<int64_t> lb,
                          ArrayRef<int64_t> ub, ArrayRef<int64_t> step,
                          ValueRange sharedOut = std::nullopt,
                          std::optional<ArrayAttr> mapping = std::nullopt)
      : rewriter_(rewriter) {
    insertPoint_ = rewriter->saveInsertionPoint();
    auto lbCst = getAsIndexOpFoldResult(rewriter->getContext(), lb);
    auto ubCst = getAsIndexOpFoldResult(rewriter->getContext(), ub);
    auto stepCst = getAsIndexOpFoldResult(rewriter->getContext(), step);
    scf::ForallOp loopOp = rewriter->create<scf::ForallOp>(
        loc, lbCst, ubCst, stepCst, sharedOut, mapping);
    rewriter->setInsertionPointToStart(loopOp.getBody());
    loopOp_ = loopOp;
    bindLoop = loopOp;
    rewriter_ = rewriter;
  }

  SCFForallRangeSimulator(OpBuilder *rewriter, Location loc,
                          scf::ForallOp &bindLoop, ArrayRef<OpFoldResult> lb,
                          ArrayRef<OpFoldResult> ub,
                          ArrayRef<OpFoldResult> step,
                          ValueRange sharedOut = std::nullopt,
                          std::optional<ArrayAttr> mapping = std::nullopt) {
    insertPoint_ = rewriter->saveInsertionPoint();
    scf::ForallOp loopOp =
        rewriter->create<scf::ForallOp>(loc, lb, ub, step, sharedOut, mapping);
    rewriter->setInsertionPointToStart(loopOp.getBody());
    loopOp_ = loopOp;
    bindLoop = loopOp;
  }

  SCFForallRangeSimulator(const SCFForallRangeSimulator &other) = delete;
  SCFForallRangeSimulator(SCFForallRangeSimulator &&other)
      : loopOp_(std::move(other.loopOp_)),
        insertPoint_(std::move(other.insertPoint_)),
        rewriter_(std::move(other.rewriter_)) {}

  ~SCFForallRangeSimulator() { rewriter_->restoreInsertionPoint(insertPoint_); }
};

/**
 *  This class builds a if-then-else node with RAII of C++.
 *  SCFIfSimulator will generate an iterator with "block_num" inside.
 *  Calling "++" on SCFIfIterator will increase "block_num" by one.
 *  Using "*" on SCFIfIterator will generate a std::pair<scope_mgr_t,int>, where
 *  the "scope_mgr_t" is the scope guard. When the "then/else" scope is done,
 *  the "scope_mgr_t" will register the generated basic_block_t to
 *  SCFIfSimulator.{true_block, false_block}. The second element of
 "*SCFIfIterator"
 *  is the current "block_num", which can help decide whether we are in
 *  "then block" or "else block". When range-based-for scope is done,
 *  SCFIfSimulator will be destructed and make and if-else node with registerd
 *  true/false blocks. The "_if_" macro wraps the underlying range-based-for.
 *  It expands like:
    {
    SCFIfSimulator _simu = SCFIfSimulator(builder::get_current(), cond);
    SCFIfIterator itr = _simu.begin()
    for (;itr != _simu.end(); itr++) {
        std::pair<scope_mgr_t,int> _scope = *itr;
        if (scope.second == 0) {
        // true block
        } else {
        // false block
        }
        // _scope is destoryed here. Will register "true block" or "false block"
        // in _simu
    }
    // _simu is destoryed here. The if-then-else is generated
    }
*/
struct SCFIfSimulator {
  scf::IfOp ifOp_;
  IRRewriter::InsertPoint insertPoint_;
  OpBuilder *rewriter_;

  struct SCFIfIterator {
    int blockNum_;
    SCFIfSimulator *scope_;
    int operator*() {
      if (blockNum_ == 0) {
        scope_->rewriter_->setInsertionPointToStart(scope_->ifOp_.thenBlock());
      } else {
        scope_->rewriter_->setInsertionPointToStart(scope_->ifOp_.elseBlock());
      }
      return blockNum_;
    }

    SCFIfIterator &operator++() {
      blockNum_++;
      return *this;
    }

    bool operator!=(SCFIfIterator &other) const {
      return blockNum_ != other.blockNum_;
    }

    SCFIfIterator(SCFIfSimulator *ifScope) : blockNum_(0), scope_(ifScope) {}
    SCFIfIterator() : blockNum_(2), scope_(nullptr) {}
  };

  SCFIfIterator begin() { return SCFIfIterator(this); }

  SCFIfIterator end() { return SCFIfIterator(); }

  SCFIfSimulator(OpBuilder *rewriter, Location loc, Value cond,
                 TypeRange resultTypes = std::nullopt)
      : rewriter_(rewriter) {
    insertPoint_ = rewriter_->saveInsertionPoint();
    ifOp_ = rewriter_->create<scf::IfOp>(loc, resultTypes, cond, true);
  }

  SCFIfSimulator(OpBuilder *rewriter, Location loc, scf::IfOp &handle,
                 Value cond, TypeRange resultTypes = std::nullopt)
      : rewriter_(rewriter) {
    insertPoint_ = rewriter_->saveInsertionPoint();
    ifOp_ = rewriter_->create<scf::IfOp>(loc, resultTypes, cond, true);
    handle = ifOp_;
  }

  SCFIfSimulator(SCFIfIterator &other) = delete;
  ~SCFIfSimulator() { rewriter_->restoreInsertionPoint(insertPoint_); }
};

#define _SCFFor_(loopHandle, ...)                                              \
  for ([[maybe_unused]] auto _loop_iter_ :                                     \
       SCFForRangeSimulator(&rewriter_, loc_, loopHandle, __VA_ARGS__))

#define _SCFForall_(loopHandle, ...)                                           \
  for ([[maybe_unused]] auto _loop_iter_ :                                     \
       SCFForallRangeSimulator(&rewriter_, loc_, loopHandle, __VA_ARGS__))

#define _SCFInParallel_(loopHandle)                                            \
  auto term = loopHandle.getTerminator();                                      \
  rewriter_.setInsertionPointToStart(term.getBody());

#define _SCFIf_(...)                                                           \
  for (auto &&__ifIter__ : SCFIfSimulator(&rewriter_, loc_, __VA_ARGS__))      \
    if (__ifIter__ == 0)

#define _SCFElse_ else

#define _SCFYield_(value) rewriter_.create<scf::YieldOp>(loc_, (value));

#define _EasyBuilderBegin_(rewriter, location)                                 \
  {                                                                            \
    OpBuilder::InsertionGuard guard(rewriter);                                 \
    OpBuilder &rewriter_ = rewriter;                                           \
    Location loc_ = (location);

#define _EasyBuilderEnd_ }

#define _BuildOp_(name, ...) rewriter_.create<name>(loc_, __VA_ARGS__);

#define _Type_(typeName) parseType(#typeName, rewriter_.getContext())

#define _Index_(value) rewriter_.create<arith::ConstantIndexOp>(loc_, value)

} // namespace mlir

#endif