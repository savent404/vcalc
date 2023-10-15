#ifndef SCALC_BACKEND_H
#define SCALC_BACKEND_H

#include "SCalcBaseVisitor.h"
#include "StringInterface.h"

using namespace scalc;

class Backend : public SCalcBaseVisitor{
private:
    StringInterface * strings;

public:
    explicit Backend(StringInterface * interface){ strings = interface; }
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
};


#endif //SCALC_BACKEND_H
