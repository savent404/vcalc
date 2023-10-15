#include "RISCV.h"
#include "nlohmann/json.hpp"
#include "inja.hpp"

using JSON = nlohmann::json;

std::string RISCV::exit() {
    return "li a7, 10\n"
           "ecall\n";
}

std::string RISCV::header() {
    // vars is a string made up of concatenated variable declarations returned from RISCV::declare().
    // Must be called after all declarations are complete.
    JSON data;
    data["vars"] = decls;
    return inja::render(
            ".data\n"
            "{{ vars }}\n"
            ".text\n"
            "main:\n",
            data);
}

std::string RISCV::declare(std::string * id) {
    JSON data;
    data["id"] = *id;
    decls += inja::render(
            "_{{ id }}: .word 0\n",
            data);
    return "";
}

std::string RISCV::assign(std::string * id) {
    //val should be a register.
    JSON data;
    data["id"] = *id;
    return inja::render(
            "lw t1, (sp)\n"
            "addi sp, sp, 4\n"
            "la t0, _{{ id }}\n"
            "sw t1, (t0)\n",
            data);
}

std::string RISCV::cond(std::string * iftrue) {
    static uint32_t cond_num = 0;
    JSON data;
    data["cond_num"] = cond_num++;
    data["iftrue"] = *iftrue;
    return inja::render(
            "lw t1, (sp)\n"
            "addi sp, sp, 4\n"
            "beq t1, zero, .cond{{ cond_num }}false\n"
            "{{ iftrue }}\n"
            ".cond{{ cond_num }}false: nop\n",
            data);
}

std::string RISCV::loop(std::string * test, std::string * body) {
    static uint32_t loop_num = 0;
    JSON data;
    data["loop_num"] = loop_num++;
    data["test"] = *test;
    data["body"] = *body;
    return inja::render(
            ".loop{{ loop_num }}begin:\n"
            "{{ test }}\n"
            "lw t1, (sp)\n"
            "addi sp, sp, 4\n"
            "beq t1, zero, .loop{{ loop_num }}end\n"
            "{{ body }}\n"
            "b .loop{{ loop_num }}begin\n"
            ".loop{{ loop_num }}end: nop\n",
            data);
}

std::string RISCV::print() {
    return "li a7, 1\n"
           "lw a0, (sp)\n"
           "addi sp, sp, 4\n"
           "ecall\n"
           "li a7, 11\n"
           "li a0, 10\n"
           "ecall\n";
}

std::string RISCV::get_value_from_id(std::string * id) {
    JSON data;
    data["id"] = *id;
    return inja::render(
            "la t0, _{{ id }}\n"
            "lw t0, (t0)\n"
            "addi sp, sp, -4\n"
            "sw t0, (sp)\n",
            data);
}

std::string RISCV::operation(std::string * op) {
    JSON data;
    data["r1"] = "t0";
    data["r2"] = "t1";
    std::string operation = "";
    if (*op == "*") {
        operation = "mul {{ r1 }}, {{ r1 }}, {{ r2 }}\n";
    } else if (*op == "/") {
        operation = "div {{ r1 }}, {{ r1 }}, {{ r2 }}\n";
    } else if (*op == "+") {
        operation = "add {{ r1 }}, {{ r1 }}, {{ r2 }}\n";
    } else if (*op == "-") {
        operation = "sub {{ r1 }}, {{ r1 }}, {{ r2 }}\n";
    } else if (*op == ">") {
        operation = "sgt {{ r1 }}, {{ r1 }}, {{ r2 }}\n";
    } else if (*op == "<") {
        operation = "slt {{ r1 }}, {{ r1 }}, {{ r2 }}\n";
    } else if (*op == "==") {
        operation = "sub {{ r1 }}, {{ r1 }}, {{ r2 }}\n"
                     "seqz {{ r1 }}, {{ r1 }}\n";
    } else if (*op == "!=") {
        operation = "sub {{ r1 }}, {{ r1 }}, {{ r2 }}\n"
                     "snez {{ r1 }}, {{ r1 }}\n";
    } else {
        return "";
    }

    data["operation"] = inja::render(operation, data);
    return inja::render(
            "lw {{ r2 }}, (sp)\n"
            "addi sp, sp, 4\n"
            "lw {{ r1 }}, (sp)\n"
            "{{ operation }}"
            "sw {{ r1 }}, (sp)\n",
            data);
}

std::string RISCV::literal(std::string * lit){
    JSON data;
    data["lit"] = *lit;
    return inja::render("addi sp, sp, -4\n"
                        "li t1, {{ lit }}\n"
                        "sw t1, (sp)\n",
                        data);
}