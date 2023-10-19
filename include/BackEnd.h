#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "ast.hpp"
#include "valc_dialect.hpp"

class BackEnd {
public:
    BackEnd();

    int emitMain(ast::BlockNode& root);

protected:
    // AST -> MLIR
    void parseBlock(ast::BlockNodePtr node);
    void parseStat(ast::StateNodePtr node);
    void parseExpr(ast::ExprNodePtr node);

    void setupPrintf();
    void printConstInt(int value);
    void printNewline();

private:
    mlir::MLIRContext context;
    mlir::ModuleOp module;
    std::shared_ptr<mlir::OpBuilder> builder;

    mlir::Location loc;
};
