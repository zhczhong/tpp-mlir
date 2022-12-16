//===- PerfOps.cpp - Perf dialect ops ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Perf/PerfOps.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "TPP/Dialect/Perf/PerfOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "TPP/Dialect/Perf/PerfOpsTypes.cpp.inc"

using namespace mlir;
using namespace mlir::perf;

//===----------------------------------------------------------------------===//
// StopTimerOp
//===----------------------------------------------------------------------===//

LogicalResult StopTimerOp::verify() {
  auto timerSrc = getTimer().getDefiningOp();
  if (!timerSrc || !isa<StartTimerOp>(timerSrc))
    return emitOpError("invalid timer input");

  // Any timer can only be stopped once. It is unusable afterwards.
  int numStopTimers = 0;
  for (auto user : timerSrc->getUsers()) {
    if (isa<StopTimerOp>(*user))
      ++numStopTimers;
  }
  if (numStopTimers != 1)
    return emitOpError("timer stopped multiple times");

  return success();
}

//===----------------------------------------------------------------------===//
// BenchOp
//===----------------------------------------------------------------------===//

void BenchOp::build(OpBuilder &builder, OperationState &result, Value numIters,
                    Value deltas, ValueRange args) {
  result.addOperands({numIters, deltas});
  result.addOperands(args);

  // Results have to match the input arguments
  for (Value v : args)
    result.addTypes(v.getType());

  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();

  // Create the default terminator if the arguments are not provided.
  // Otherwise, leave this to the caller because we don't know which values to
  // return from the body.
  if (args.empty()) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    builder.create<perf::YieldOp>(result.location);
  }
}

YieldOp BenchOp::getYieldOp() {
  return cast<perf::YieldOp>(getRegion().front().getTerminator());
}