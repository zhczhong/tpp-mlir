#ifndef TPP_RUN_MLIRBENCH_H
#define TPP_RUN_MLIRBENCH_H

//===- MLIRBench.h - MLIR Benchmark Producer ------------------------------===//
//
// Producer for benchmark wrapper methods. Upon selecting a kernel to run, maps
// the arguments, random initialize them and call the kernel as many times as
// requested, taking measurements and printing the result in the end.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class ModuleOp;
class MemRefType;
class Operation;
class Value;
namespace func {
class FuncOp;
} // namespace func

/// MLIRBench - Creates wrapper for calling kernel methods.
///
/// Note: This class is a mix between a utility class and a driver
/// because I still don't know which way we want to go. For now, the
/// inteface is a bit weird, but it will get better once we clear the
/// API design, with time.
class MLIRBench {
  /// MLIR OpBulder
  OpBuilder builder;

  /// Unknown location, since all this code is auto-generated
  Location unkLoc;

  /// Main module
  ModuleOp module;

  /// Kernel function, if found
  func::FuncOp kernel;

  /// Values of the kernel arguments (no need to declare every time)
  llvm::SmallVector<Value> kernelArgs;

  /// Main wrapper function, calls kernel
  func::FuncOp main;

  /// Local cache of the main name
  llvm::StringRef mainName;

  /// Global variables for all arguments (in order)
  llvm::SmallVector<llvm::StringRef> globals;

  /// Create a random global based on the memref type
  llvm::StringRef createGlobal(MemRefType);

  struct {
    func::FuncOp alloc;
    func::FuncOp start;
    func::FuncOp stop;
    func::FuncOp average;
    func::FuncOp deviation;
  } timer;

  /// Get a global memref by name
  MemRefType getGlobalType(llvm::StringRef);

  // Create a random constant dense tensor
  Value createConstTensor(TensorType);

  /// Gets module's main block
  Block &getModuleBlock();

  /// Gets main wrappers's block
  Block &getMainBlock();

public:
  /// Creates context, builder
  MLIRBench(Operation *op);

  /// Finds the kernel method, checks correct name and shape
  LogicalResult findKernel(llvm::StringRef);

  /// Check if the kernel is already an entry point
  /// Find the kernel first with findKernel.
  LogicalResult checkKernelSignature();

  /// Renames the kernel to _name, so that we can create the wrapper
  LogicalResult renameKernel();

  /// Create and initialize the kernel input arguments
  /// The values are cached locally in a kernel argument list, in order
  LogicalResult createKernelArgs();

  /// Create all globals for the kernel method initializers
  /// Populates the list with the names, in order
  LogicalResult createGlobals(llvm::SmallVector<llvm::StringRef> &);

  /// Create main wrapper function, sets insertion point
  LogicalResult createMainWrapper();

  /// Creates and returns a call to the kernel.
  Operation *callKernel();

  /// Returns the result of a kernel call, which is either
  /// the return value (if any) or the last argument (outs).
  Value getKernelResult(Operation *kernelCall);

  /// Create a benchmarking region around the kernel call
  /// Returns the memref containing measured time deltas
  Value createTimerLoop(unsigned);

  /// Get the timer average/deviation
  Value getTimerStats(Value);

  /// Prints a float value (used for mean/dev)
  void printVector(Value);

  /// Prints the shaped type (tensor/memref) as a vector read + print
  LogicalResult printShapedType(Value);

  /// Prints the result of a kernel call
  LogicalResult printResult(Operation *kernelCall);

  /// Terminates the function, issuing a return, lower to LLVM
  LogicalResult finalize();

  /// Reports error on the current module's location
  LogicalResult emitError(llvm::Twine);
};

} // namespace mlir

#endif
