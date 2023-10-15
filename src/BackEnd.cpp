#include "Backend.h"

std::any Backend::visitFile(SCalcParser::FileContext * ctx) {
    std::string output = "";
    for (auto f_stmt : ctx->full_statement()) {
        output += std::any_cast<std::string>(visit(f_stmt));
    }
    if (output == ""){return (std::string)"";}
    return (std::string)(strings->header() + output + strings->exit());
}

std::any Backend::visitDecl(SCalcParser::DeclContext * ctx) {
    std::string name = ctx->name->getText();
    strings->declare(&name);
    auto expr = std::any_cast<std::string>(visit(ctx->exp));
    return (std::string)(expr + strings->assign(&name));
}

std::any Backend::visitAssn(SCalcParser::AssnContext * ctx) {
    std::string name = ctx->name->getText();
    auto expr = std::any_cast<std::string>(visit(ctx->exp));
    return (std::string)(expr + strings->assign(&name));
}

std::any Backend::visitCond(SCalcParser::CondContext * ctx) {
    auto expr = std::any_cast<std::string>(visit(ctx->exp));
    std::string body = "";
    for (auto ps : ctx->part_statement()) {
        body += std::any_cast<std::string>(visit(ps));
    }
    return (std::string)(expr + strings->cond(&body));
}

std::any Backend::visitLoop(SCalcParser::LoopContext * ctx) {
    auto expr = std::any_cast<std::string>(visit(ctx->exp));
    std::string body = "";
    for (auto ps : ctx->part_statement()) {
        body += std::any_cast<std::string>(visit(ps));
    }
    return (std::string)strings->loop(&expr, &body);
}

std::any Backend::visitPrnt(SCalcParser::PrntContext * ctx) {
    auto expr = std::any_cast<std::string>(visit(ctx->exp));
    return (std::string)(expr + strings->print());
}

std::any Backend::visitParens(SCalcParser::ParensContext * ctx) {
    return (std::string)std::any_cast<std::string>(visit(ctx->expression()));
}

std::any Backend::visitOp(SCalcParser::OpContext * ctx) {
    auto left = std::any_cast<std::string>(visit(ctx->expression(0)));
    auto right = std::any_cast<std::string>(visit(ctx->expression(1)));
    std::string op = ctx->op->getText();
    return (std::string)(left + right + strings->operation(&op));
}

std::any Backend::visitLit(SCalcParser::LitContext * ctx) {
    std::string lit = ctx->INT_LIT()->getText();
    return (std::string)strings->literal(&lit);
}

std::any Backend::visitId(SCalcParser::IdContext * ctx) {
    std::string id = ctx->getText();
    return (std::string)strings->get_value_from_id(&id);
}