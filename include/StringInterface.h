#ifndef SCALC_STRINGINTERFACE_H
#define SCALC_STRINGINTERFACE_H

#include <string>

class StringInterface {
public:
    StringInterface()= default;
    virtual std::string exit(){return "";};
    virtual std::string header(){return "";};
    virtual std::string declare(std::string * id){return "";};
    virtual std::string assign(std::string * id){return "";};
    virtual std::string cond(std::string * iftrue){return "";};
    virtual std::string loop(std::string * test, std::string * body){return "";};
    virtual std::string print(){return "";};
    virtual std::string get_value_from_id(std::string * id){return "";};
    virtual std::string operation(std::string * op){return "";};
    virtual std::string literal(std::string * lit){return "";};
};

#endif //SCALC_STRINGINTERFACE_H
