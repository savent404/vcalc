#include "ARM.h"
#include "nlohmann/json.hpp"
#include "inja.hpp"

using JSON = nlohmann::json;

std::string ARM::exit() {
    return "pop {ip, lr}\n"
           "mov r0, #0\n"
           "bx lr\n";
}

std::string ARM::header() {
    // vars is a string made up of concatenated variable declarations returned from ARM::declare().
    // Must be called after all declarations are complete.
    JSON data;
    data["vars"] = decls;
    return inja::render(
            ".arch armv7-a\n"
            ".data\n"
            "_format_string: .asciz \"%d\\n\"\n"
            "{{ vars }}\n"
            ".text\n"
            ".globl main\n"
            "main:\n"
            "push {ip, lr}\n",
            data);
}

std::string ARM::declare(std::string * id) {
    JSON data;
    data["id"] = *id;
    decls += inja::render(
            "_{{ id }}: .word 0\n",
            data);
    return "";
}

std::string ARM::assign(std::string * id) {
    //val should be a register.
    JSON data;
    data["id"] = *id;
    return inja::render(
            "pop {r0}\n"
            "ldr r1, =_{{ id }}\n"
            "str r0, [r1]\n",
            data);
}

std::string ARM::cond(std::string * iftrue) {
    static uint32_t cond_num = 0;
    JSON data;
    data["cond_num"] = cond_num++;
    data["iftrue"] = *iftrue;
    return inja::render(
            "pop {r0}\n"
            "cmp r0, #0\n"
            "beq .cond{{ cond_num }}false\n"
            "{{ iftrue }}\n"
            ".cond{{ cond_num }}false: nop\n",
            data);
}

std::string ARM::loop(std::string * test, std::string * body) {
    static uint32_t loop_num = 0;
    JSON data;
    data["loop_num"] = loop_num++;
    data["test"] = *test;
    data["body"] = *body;
    return inja::render(
            ".loop{{ loop_num }}begin:\n"
            "{{ test }}\n"
            "pop {r0}\n"
            "cmp r0, #0\n"
            "beq .loop{{ loop_num }}end\n"
            "{{ body }}\n"
            "b .loop{{ loop_num }}begin\n"
            ".loop{{ loop_num }}end: nop\n",
            data);
}

std::string ARM::print() {
    return "ldr r0, =_format_string\n"
           "pop {r1}\n"
           "bl printf\n";
}

std::string ARM::get_value_from_id(std::string * id) {
    JSON data;
    data["id"] = *id;
    return inja::render(
            "ldr r0, =_{{ id }}\n"
            "ldr r0, [r0]\n"
            "push {r0}\n",
            data);
}

std::string ARM::operation(std::string * op) {
    static uint32_t id = 0;
    JSON data;
    data["r1"] = "r0";
    data["r2"] = "r1";
    data["id"] = id;
    std::string operation = "";
    if (*op == "*") {
        operation = "mul {{ r1 }}, {{ r1 }}, {{ r2 }}\n";
    } else if (*op == "/") {
        operation = "bl __aeabi_idiv(PLT)\n";
    } else if (*op == "+") {
        operation = "add {{ r1 }}, {{ r1 }}, {{ r2 }}\n";
    } else if (*op == "-") {
        operation = "sub {{ r1 }}, {{ r1 }}, {{ r2 }}\n";
    } else if (*op == ">") {
        operation = "cmp {{ r1 }}, {{ r2 }}\n"
                    "bgt .gt{{ id }}\n"
                    "mov {{ r1 }}, #0\n"
                    "b .ngt{{ id }}\n"
                    ".gt{{ id }}: mov {{ r1 }}, #1\n"
                    ".ngt{{ id }}:\n";
        ++id;
    } else if (*op == "<") {
        operation = "cmp {{ r1 }}, {{ r2 }}\n"
                    "blt .lt{{ id }}\n"
                    "mov {{ r1 }}, #0\n"
                    "b .nlt{{ id }}\n"
                    ".lt{{ id }}: mov {{ r1 }}, #1\n"
                    ".nlt{{ id }}:\n";
        ++id;
    } else if (*op == "==") {
        operation = "cmp {{ r1 }}, {{ r2 }}\n"
                    "beq .eq{{ id }}\n"
                    "mov {{ r1 }}, #0\n"
                    "b .neq{{ id }}\n"
                    ".eq{{ id }}: mov {{ r1 }}, #1\n"
                    ".neq{{ id }}:\n";
        ++id;
    } else if (*op == "!=") {
        operation = "cmp {{ r1 }}, {{ r2 }}\n"
                    "bne .ne{{ id }}\n"
                    "mov {{ r1 }}, #0\n"
                    "b .eq{{ id }}\n"
                    ".ne{{ id }}: mov {{ r1 }}, #1\n"
                    ".eq{{ id }}:\n";
        ++id;
    } else {
        return "";
    }

    data["operation"] = inja::render(operation, data);
    return inja::render(
            "pop { {{ r2 }} }\n"
            "pop { {{ r1 }} }\n"
            "{{ operation }}"
            "push { {{ r1 }} }\n",
            data);
}

std::string ARM::literal(std::string * lit){
    JSON data;
    data["lit"] = *lit;
    return inja::render("mov r0, #{{ lit }}\n"
                        "push {r0}\n",
                        data);
}