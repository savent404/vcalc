#pragma once

#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace valc {
class VcalcDialect : public mlir::Dialect {
public:
    explicit VcalcDialect(mlir::MLIRContext* context);
    static llvm::StringRef getDialectNamespace() { return "vcalc"; }
    void initialize();
};

class PrintOp : public mlir::Op<PrintOp,
                    mlir::OpTrait::OneOperand,
                    mlir::OpTrait::ZeroResults> {
public:
    using Op::Op;
    static llvm::StringRef getOperationName() { return "vcalc.print"; }
    mlir::LogicalResult verifyInvariants() { return mlir::success(); }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state,
        mlir::DenseElementsAttr value);
};
} // namespace valc
