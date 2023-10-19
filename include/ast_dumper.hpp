#pragma once

#include "ast.hpp"
#include <iostream>

namespace ast {

struct AstDumper {
private:
    int indent = 0;
    void printIndent()
    {
        for (int i = 0; i < indent; i++) {
            std::cout << ". ";
        }
    }
    void Callee(std::function<void()> func)
    {
        indent++;
        func();
        indent--;
    }

public:
    void dump(BlockNodePtr block)
    {
        printIndent();
        std::cout << "BlockNode {" << std::endl;
        Callee([&]() {
            for (auto& stat : block->body) {
                dump(stat);
            }
        });
        printIndent();
        std::cout << "}" << std::endl;
    }

    void dump(StateNodePtr stat)
    {
        switch (stat->getStatKind()) {
        case StatKind::VarAssign:
            printIndent();
            std::cout << "VarAssignNode {" << std::endl;
            Callee([&] {
                auto node = reinterpret_cast<VarAssignNodePtr>(stat);
                printIndent();
                std::cout << "varassign $var=$expr" << std::endl;
                // dump var
                printIndent();
                std::cout << "$var: {" << std::endl;
                Callee([&] {
                    dump(node->var);
                });
                printIndent();
                std::cout << "}" << std::endl;

                // dump var's expr
                printIndent();
                std::cout << "$expr: {" << std::endl;
                Callee([&] {
                    dump(node->expr);
                });
                printIndent();
                std::cout << "}" << std::endl;
            });

            printIndent();
            std::cout << "}" << std::endl;
            break;
        case StatKind::VarDeclare:
            printIndent();
            std::cout << "VarDeclareNode {" << std::endl;
            Callee([&] {
                auto node = reinterpret_cast<VarDeclareNodePtr>(stat);
                printIndent();
                std::cout << "vardef $var=$expr" << std::endl;
                // dump var
                printIndent();
                std::cout << "$var: {" << std::endl;
                Callee([&] {
                    dump(node->value);
                });
                printIndent();
                std::cout << "}" << std::endl;

                // dump var's expr
                printIndent();
                std::cout << "$expr: {" << std::endl;
                Callee([&] {
                    dump(node->expr);
                });
                printIndent();
                std::cout << "}" << std::endl;
            });
            printIndent();
            std::cout << "}" << std::endl;
            break;
        case StatKind::Print:
            printIndent();
            std::cout << "PrintNode {" << std::endl;
            Callee([&] {
                auto node = reinterpret_cast<PrintNodePtr>(stat);
                printIndent();
                std::cout << "print $expr" << std::endl;
                printIndent();
                std::cout << "$expr: {" << std::endl;
                Callee([&] {
                    dump(node->expr);
                });
                printIndent();
                std::cout << "}" << std::endl;
            });
            printIndent();
            std::cout << "}" << std::endl;
            break;
        case StatKind::If:
            printIndent();
            std::cout << "IfNode {" << std::endl;
            Callee([&] {
                auto node = reinterpret_cast<IfNodePtr>(stat);
                printIndent();
                std::cout << "if $expr {" << std::endl;
                Callee([&] {
                    dump(node->expr);
                });
                printIndent();
                std::cout << "}" << std::endl;
                printIndent();
                std::cout << "then {" << std::endl;
                Callee([&] {
                    dump(node->body);
                });
                printIndent();
                std::cout << "}" << std::endl;
            });
            printIndent();
            std::cout << "}" << std::endl;
            break;
        case StatKind::Loop:
            printIndent();
            std::cout << "LoopNode {" << std::endl;
            Callee([&] {
                auto node = reinterpret_cast<LoopNodePtr>(stat);
                printIndent();
                std::cout << "loop $expr {" << std::endl;
                Callee([&] {
                    dump(node->expr);
                });
                printIndent();
                std::cout << "}" << std::endl;
                printIndent();
                std::cout << "do {" << std::endl;
                Callee([&] {
                    dump(node->body);
                });
                printIndent();
                std::cout << "}" << std::endl;
            });
            printIndent();
            std::cout << "}" << std::endl;
            break;
        }
    }

    void dump(ExprNodePtr expr)
    {
        switch (expr->getExprKind()) {
        case ExprKind::Value:
            dump(reinterpret_cast<ValueNodePtr>(expr));
            break;
        case ExprKind::Binary:
            const std::string binaryStr[] = {
                [static_cast<int>(BinaryKind::Add)] = "+",
                [static_cast<int>(BinaryKind::Sub)] = "-",
                [static_cast<int>(BinaryKind::Mul)] = "*",
                [static_cast<int>(BinaryKind::Div)] = "/",
                [static_cast<int>(BinaryKind::And)] = "and",
                [static_cast<int>(BinaryKind::Or)] = "or",
                [static_cast<int>(BinaryKind::Range)] = "..",
                [static_cast<int>(BinaryKind::Eq)] = "==",
                [static_cast<int>(BinaryKind::Ne)] = "!=",
                [static_cast<int>(BinaryKind::Lt)] = "<",
                [static_cast<int>(BinaryKind::Le)] = "<=",
                [static_cast<int>(BinaryKind::Gt)] = ">",
                [static_cast<int>(BinaryKind::Ge)] = ">=",
            };
            printIndent();
            std::cout << "BinaryNode {" << std::endl;
            Callee([&] {
                auto binaryExpr = reinterpret_cast<BinaryNodePtr>(expr);
                auto binaryKind = binaryExpr->getBinaryKind();
                printIndent();
                std::cout << "def $l $op $r" << std::endl;
                printIndent();
                std::cout << "op: " << binaryStr[static_cast<int>(binaryKind)] << std::endl;
                printIndent();
                std::cout << "$l: {" << std::endl;
                Callee([&] {
                    dump(binaryExpr->lhs);
                });
                printIndent();
                std::cout << "}" << std::endl;
                printIndent();
                std::cout << "$r: {" << std::endl;
                Callee([&] {
                    dump(binaryExpr->rhs);
                });
                printIndent();
                std::cout << "}" << std::endl;
            });
            printIndent();
            std::cout << "}" << std::endl;
            break;
        }
    }
    void dump(ValueNodePtr value)
    {
        switch (value->getValueKind()) {
        case ValueKind::Var:
            printIndent();
            std::cout << "VarNode {" << std::endl;
            Callee([&] {
                printIndent();
                std::cout << "name: " << value->name << std::endl;
            });
            printIndent();
            std::cout << "}" << std::endl;
            break;
        case ValueKind::Const:
            printIndent();
            std::cout << "ConstNode {" << std::endl;
            Callee([&] {
                printIndent();
                std::cout << "value: " << value->const_val << std::endl;
            });
            printIndent();
            std::cout << "}" << std::endl;
        }
    }
};

}
