#include "Interpreter.h"

std::any Interpreter::visitFile(SCalcParser::FileContext *ctx) {
    for (auto f_stmt: ctx->full_statement()) {
        visit(f_stmt);
    }
    return nullptr;
}

std::any Interpreter::visitDecl(SCalcParser::DeclContext * ctx) {
    scope[ctx->name->getText()] = std::any_cast<int32_t>(visit(ctx->exp));
    return nullptr;
}

std::any Interpreter::visitAssn(SCalcParser::AssnContext * ctx) {
    scope[ctx->name->getText()] = std::any_cast<int32_t>(visit(ctx->exp));
    return nullptr;
}

std::any Interpreter::visitCond(SCalcParser::CondContext * ctx) {
    if (std::any_cast<int32_t>(visit(ctx->exp))) {
        for (auto ps : ctx->part_statement()) {
            visit(ps);
        }
    }
    return nullptr;
}

std::any Interpreter::visitLoop(SCalcParser::LoopContext * ctx) {
    while (std::any_cast<int32_t>(visit(ctx->exp))) {
        for (auto ps : ctx->part_statement()) {
            visit(ps);
        }
    }
    return nullptr;
}

std::any Interpreter::visitPrnt(SCalcParser::PrntContext * ctx) {
    prints += std::to_string(std::any_cast<int32_t>(visit(ctx->exp))) + "\n";
    return nullptr;
}

std::any Interpreter::visitParens(SCalcParser::ParensContext * ctx) {
    return std::any_cast<int32_t>(visit(ctx->expression()));
}

std::any Interpreter::visitOp(SCalcParser::OpContext * ctx){
    size_t op = ctx->op->getType();
    auto left = std::any_cast<int32_t>(visit(ctx->expression(0)));
    auto right = std::any_cast<int32_t>(visit(ctx->expression(1)));

    if (op == scalc::SCalcParser::ADD){
        left = left + right;
    } else if (op == scalc::SCalcParser::SUB){
        left = left - right;
    } else if (op == scalc::SCalcParser::MUL){
        left = left * right;
    } else if (op == scalc::SCalcParser::DIV){
        left = left / right;
    } else if (op == scalc::SCalcParser::GT){
        left = left > right;
    } else if (op == scalc::SCalcParser::LT){
        left = left < right;
    } else if (op == scalc::SCalcParser::EE){
        left = left == right;
    } else if (op == scalc::SCalcParser::NE){
        left = left != right;
    }

    return (int32_t)left;
}

std::any Interpreter::visitLit(SCalcParser::LitContext * ctx){
    return (int32_t)std::stoi(ctx->INT_LIT()->getText());
}

std::any Interpreter::visitId(SCalcParser::IdContext * ctx){
    return (int32_t)scope[ctx->ID()->getText()];
}