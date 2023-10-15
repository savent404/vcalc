#include "X86.h"

#include "X86.h"
#include "nlohmann/json.hpp"
#include "inja.hpp"

using JSON = nlohmann::json;

std::string X86::exit() {
    return "mov eax, 0\n"
           "ret\n";
}

std::string X86::header() {
    // vars is a string made up of concatenated variable declarations returned from X86::declare().
    // Must be called after all declarations are complete.
    JSON data;
    data["vars"] = decls;
    return inja::render(
            "global main\n"
            "extern printf\n"
            "section .data\n"
            "_format_string: DB \"%d\", 0xA, 0x0\n"
            "{{ vars }}\n"
            "section .text\n"
            "main:\n",
            data);
}

std::string X86::declare(std::string * id) {
    JSON data;
    data["id"] = *id;
    decls += inja::render(
            "_{{ id }}: DD 0\n",
            data);
    return "";
}

std::string X86::assign(std::string * id) {
    //val should be a register.
    JSON data;
    data["id"] = *id;
    return inja::render(
            "mov eax, _{{ id }}\n"
            "pop ecx\n"
            "mov [eax], ecx\n",
            data);
}

std::string X86::cond(std::string * iftrue) {
    static uint32_t cond_num = 0;
    JSON data;
    data["cond_num"] = cond_num++;
    data["iftrue"] = *iftrue;
    return inja::render(
            "pop eax\n"
            "cmp eax, 0\n"
            "je cond{{ cond_num }}false\n"
            "{{ iftrue }}\n"
            "cond{{ cond_num }}false: nop\n",
            data);
}

std::string X86::loop(std::string * test, std::string * body) {
    static uint32_t loop_num = 0;
    JSON data;
    data["loop_num"] = loop_num++;
    data["test"] = *test;
    data["body"] = *body;
    return inja::render(
            "loop{{ loop_num }}begin:\n"
            "{{ test }}\n"
            "pop eax\n"
            "cmp eax, 0\n"
            "je loop{{ loop_num }}end\n"
            "{{ body }}\n"
            "jmp loop{{ loop_num }}begin\n"
            "loop{{ loop_num }}end: nop\n",
            data);
}

std::string X86::print() {
    return "push _format_string\n"
           "call printf\n"
           "add esp, 8\n";
}

std::string X86::get_value_from_id(std::string * id) {
    JSON data;
    data["id"] = *id;
    return inja::render(
            "mov eax, _{{ id }}\n"
            "mov eax, [eax]\n"
            "push eax\n",
            data);
}

std::string X86::operation(std::string * op) {
    static int32_t id = 0;
    JSON data;
    data["r1"] = "eax";
    data["r2"] = "ecx";
    data["id"] = id;
    std::string operation = "";
    if (*op == "*") {
        operation = "imul {{ r1 }}, {{ r2 }}\n";
    } else if (*op == "/") {
        operation = "idiv {{ r2 }}\n";
    } else if (*op == "+") {
        operation = "add {{ r1 }}, {{ r2 }}\n";
    } else if (*op == "-") {
        operation = "sub {{ r1 }}, {{ r2 }}\n";
    } else if (*op == ">") {
        operation = "cmp {{ r1 }}, {{ r2 }}\n"
                    "jg gt{{ id }}\n"
                    "mov {{ r1 }}, 0\n"
                    "jmp ngt{{ id }}\n"
                    "gt{{ id }}: mov {{ r1 }}, 1\n"
                    "ngt{{ id }}:\n";
        ++id;
    } else if (*op == "<") {
        operation = "cmp {{ r1 }}, {{ r2 }}\n"
                    "jl lt{{ id }}\n"
                    "mov {{ r1 }}, 0\n"
                    "jmp nlt{{ id }}\n"
                    "lt{{ id }}: mov {{ r1 }}, 1\n"
                    "nlt{{ id }}:\n";
        ++id;
    } else if (*op == "==") {
        operation = "cmp {{ r1 }}, {{ r2 }}\n"
                    "je eq{{ id }}\n"
                    "mov {{ r1 }}, 0\n"
                    "jmp neq{{ id }}\n"
                    "eq{{ id }}: mov {{ r1 }}, 1\n"
                    "neq{{ id }}:\n";
        ++id;
    } else if (*op == "!=") {
        operation = "cmp {{ r1 }}, {{ r2 }}\n"
                    "jne neq{{ id }}\n"
                    "mov {{ r1 }}, 0\n"
                    "jmp eq{{ id }}\n"
                    "neq{{ id }}: mov {{ r1 }}, 1\n"
                    "eq{{ id }}:\n";
        ++id;
    } else {
        return "";
    }

    data["operation"] = inja::render(operation, data);
    return inja::render(
            "pop {{ r2 }}\n"
            "pop {{ r1 }}\n"
            "{{ operation }}"
            "push {{ r1 }}\n",
            data);
}

std::string X86::literal(std::string * lit){
    JSON data;
    data["lit"] = *lit;
    return inja::render("push {{ lit }}\n", data);
}