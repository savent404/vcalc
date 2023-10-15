#ifndef SCALC_X86_H
#define SCALC_X86_H

#include "StringInterface.h"

class X86 : public StringInterface {
private:
    std::string decls = "";
public:
    X86()= default;
    std::string exit() override;
    std::string header() override;
    std::string declare(std::string * id) override;
    std::string assign(std::string * id) override;
    std::string cond(std::string * iftrue) override;
    std::string loop(std::string * test, std::string * body) override;
    std::string print() override;
    std::string get_value_from_id(std::string * id) override;
    std::string operation(std::string * op) override;
    std::string literal(std::string * lit) override;
};


#endif //SCALC_X86_H
