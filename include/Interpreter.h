#ifndef SCALC_EJS_UALBERTA_INTERPRETER_H
#define SCALC_EJS_UALBERTA_INTERPRETER_H

#include <cstdint>
#include <fstream>
#include <map>
#include "SCalcBaseVisitor.h"

using namespace scalc;

class Interpreter : public SCalcBaseVisitor {
private:
    std::map<std::string, int32_t> scope;
    std::string prints = "";
public:
    explicit Interpreter()=default;
    std::any visitFile(SCalcParser::FileContext * ctx) override;
    std::any visitDecl(SCalcParser::DeclContext * ctx) override;
    std::any visitAssn(SCalcParser::AssnContext * ctx) override;
    std::any visitCond(SCalcParser::CondContext * ctx) override;
    std::any visitLoop(SCalcParser::LoopContext * ctx) override;
    std::any visitPrnt(SCalcParser::PrntContext * ctx) override;
    std::any visitParens(SCalcParser::ParensContext * ctx) override;
    std::any visitOp(SCalcParser::OpContext * ctx) override;
    std::any visitLit(SCalcParser::LitContext * ctx) override;
    std::any visitId(SCalcParser::IdContext * ctx) override;
    std::string get_output(){return prints;};
};

#endif //SCALC_EJS_UALBERTA_INTERPRETER_H
