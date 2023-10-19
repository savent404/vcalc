#pragma once

#include "VCalcBaseVisitor.h"
#include "VCalcLexer.h"
#include "VCalcParser.h"
#include "VCalcVisitor.h"

#include "ast.hpp"

namespace ast {

struct IdGenerator {
    size_t nextId = 1;
    size_t genId()
    {
        return nextId++;
    }
};

struct TokenManager {
    std::map<size_t, NodePtr> maps;
    size_t nextTokenId = 1;
    void addNode(NodePtr node)
    {
        node->tokenId = nextTokenId++;
        maps[node->tokenId] = node;
    }

    template <typename T>
    T getNode(size_t tokenId)
    {
        assert(tokenId > 0);
        return dynamic_cast<T>(maps[tokenId]);
    }
};

struct VCalAstGenerator : public vcalc::VCalcBaseVisitor {
    BlockNodePtr rootNode;
    BlockNodePtr currentBlock;
    VarScopePtr rootScope;
    VarScopePtr currentScope;
    TokenManager tokens;

    VCalAstGenerator()
    {
        rootNode = new BlockNode();
        currentBlock = rootNode;
        rootScope = new VarScope { rootNode, nullptr };
        currentScope = rootScope;
    }

    void insertStat(StateNodePtr node)
    {
        currentBlock->addStat(node);
    }

    virtual std::any aggregateResult(std::any last, std::any nextResult) override
    {
        if (nextResult.has_value()) {
            return nextResult;
        } else {
            return last;
        }
    }

    virtual std::any visitStat(vcalc::VCalcParser::StatContext* ctx) override
    {
        auto any = visitChildren(ctx);
        auto token = any.has_value() ? std::any_cast<size_t>(any) : 0;
        auto node = tokens.getNode<StateNodePtr>(token);
        assert(node->getNodeKind() == Node::Type::Stat);
        insertStat(node);
        return node->tokenId;
    }

    virtual std::any visitPrint(vcalc::VCalcParser::PrintContext* ctx) override
    {
        auto any = visit(ctx->exp());
        auto token = std::any_cast<size_t>(any);
        auto exp = tokens.getNode<ExprNodePtr>(token);
        assert(exp->getNodeKind() == Node::Type::Expr);
        auto node = new PrintNode { exp };
        tokens.addNode(node);
        return node->tokenId;
    }

    void createVarScopeAndVisit(NodePtr binding, std::function<void(void)> fn)
    {
        auto scope = new VarScope { binding, currentScope };
        auto old_scope = currentScope;

        currentScope = scope;
        fn();
        currentScope = old_scope;
    }

    BlockNodePtr createBlockAndVisit(std::function<void(void)> fn)
    {
        // create new block
        auto block = new BlockNode();
        tokens.addNode(block);

        // push block and scope, use new one as current block
        auto block_backup = currentBlock;
        currentBlock = block;

        // create new scope and visit fn()
        createVarScopeAndVisit(block, [&]() {
            fn();
        });
        // pop block and scope, recover old one as current block
        currentBlock = block_backup;

        return block;
    }

    virtual std::any visitIf(vcalc::VCalcParser::IfContext* ctx) override
    {
        auto any = visit(ctx->exp());
        auto token = std::any_cast<size_t>(any);
        auto exp = tokens.getNode<ExprNodePtr>(token);

        auto block = createBlockAndVisit([&]() {
            any = visit(ctx->block());
            token = std::any_cast<size_t>(any);
        });

        if (exp->getValueType() != ValueType::Boolean) {
            exp = new ConvertNode { ValueType::Boolean, nullptr, exp };
            tokens.addNode(exp);
        }

        auto node = new IfNode { exp, block };
        tokens.addNode(node);
        return node->tokenId;
    }

    virtual std::any visitLoop(vcalc::VCalcParser::LoopContext* ctx) override
    {
        auto any = visit(ctx->exp());
        auto token = std::any_cast<size_t>(any);
        auto exp = tokens.getNode<ExprNodePtr>(token);

        auto block = createBlockAndVisit([&]() {
            any = visit(ctx->block());
            token = std::any_cast<size_t>(any);
        });

        if (exp->getValueType() != ValueType::Boolean) {
            exp = new ConvertNode { ValueType::Boolean, nullptr, exp };
            tokens.addNode(exp);
        }

        auto node = new LoopNode { exp, block };
        tokens.addNode(node);
        return node->tokenId;
    }

    virtual std::any visitAssign(vcalc::VCalcParser::AssignContext* ctx) override
    {
        auto any = visit(ctx->var());
        auto token = std::any_cast<size_t>(any);
        auto var = tokens.getNode<ValueNodePtr>(token);
        assert(var->getValueKind() == ValueKind::Var);
        any = visit(ctx->exp());
        token = std::any_cast<size_t>(any);
        auto exp = tokens.getNode<ExprNodePtr>(token);
        auto node = new VarAssignNode { var, exp };
        tokens.addNode(node);
        return node->tokenId;
    }

    virtual std::any visitVardef(vcalc::VCalcParser::VardefContext* ctx) override
    {
        auto any = visit(ctx->var());
        auto token = std::any_cast<size_t>(any);
        auto var = tokens.getNode<ValueNodePtr>(token);

        // figure out var's ValueType
        auto str = ctx->getText();
        auto fn_beginWith = [](std::string str, std::string prefix) {
            return str.substr(0, prefix.size()) == prefix;
        };

        if (fn_beginWith(str, "int")) {
            var->valueType = ValueType::Int;
        } else if (fn_beginWith(str, "vector")) {
            var->valueType = ValueType::Vector;
        } else {
            throw std::runtime_error("visitVardef: unknown var type");
        }

        // add var to current scope, check if var already defined
        if (currentScope->find(var->name)) {
            throw std::runtime_error("visitVardef: var already defined");
        }
        currentScope->add(var);

        any = visit(ctx->exp());
        token = std::any_cast<size_t>(any);
        auto exp = tokens.getNode<ExprNodePtr>(token);
        auto node = new VarDeclareNode { var, exp };
        tokens.addNode(node);
        return node->tokenId;
    }

    virtual std::any visitExp(vcalc::VCalcParser::ExpContext* ctx) override
    {
        ExprNodePtr node = nullptr;
        if (ctx->varOrExp()) {
            auto any = visit(ctx->varOrExp());
            auto token = std::any_cast<size_t>(any);
            auto exp = tokens.getNode<ExprNodePtr>(token);
            assert(exp->getNodeKind() == Node::Type::Expr);
            return exp->tokenId;
        } else if (ctx->operatorGenerator()) {
            throw std::runtime_error("visitExp: not implemented");
        } else if (ctx->operatorFilter()) {
            throw std::runtime_error("visitExp: not implemented");
        } else if (ctx->operatorRange()) {
            auto lhs_token = std::any_cast<size_t>(visit(ctx->exp(0)));
            auto rhs_token = std::any_cast<size_t>(visit(ctx->exp(1)));
            auto lhs = tokens.getNode<ExprNodePtr>(lhs_token);
            auto rhs = tokens.getNode<ExprNodePtr>(rhs_token);
            auto fn_convert2Int = [&](ExprNodePtr node) {
                auto type = node->getValueType();
                if (type == ValueType::Boolean) {
                    node = new ConvertNode { ValueType::Int, nullptr, node };
                    tokens.addNode(node);
                } else if (type == ValueType::Int) {
                    // do nothing
                } else {
                    throw std::runtime_error("visitExp: range expect two int as index");
                }
                return node;
            };
            node = new ValueNode { fn_convert2Int(lhs), fn_convert2Int(rhs) };
        } else if (ctx->operatorMulDiv()) {
            node = createBinaryNodeHelper(getBinaryKind(ctx->operatorMulDiv()->getText()), ctx);
        } else if (ctx->operatorAddSub()) {
            node = createBinaryNodeHelper(getBinaryKind(ctx->operatorAddSub()->getText()), ctx);
        } else if (ctx->operatorAnd()) {
            node = createBinaryNodeHelper(BinaryKind::And, ctx);
        } else if (ctx->operatorOr()) {
            node = createBinaryNodeHelper(BinaryKind::Or, ctx);
        } else if (ctx->operatorComparison()) {
            node = createBinaryNodeHelper(getBinaryKind(ctx->operatorComparison()->getText()), ctx);
        } else {
            throw std::runtime_error("visitExp: unknown exp type");
        }

        if (node) {
            tokens.addNode(node);
            return node->tokenId;
        } else {
            throw std::runtime_error("visitExp: unknown exp type");
        }
    }

    virtual std::any visitVar(vcalc::VCalcParser::VarContext* ctx) override
    {
        ValueNodePtr node = nullptr;
        if (ctx->NAME()) {

            // we also need new node for var 'a' even already defined 'a' in VarScope
            // It be comes 'a.1' 'a.2' 'a.3' ... in backend to identify different state
            node = new ValueNode(ctx->NAME()->getText());
            // Find out var's valueType
            auto var = currentScope->find(node->name);
            if (var) {
                node->valueType = var->valueType;
            } else {
                // NOTE: only leave Unknow if var not defined
                // visitVardef will figure out var's valueType
            }
        } else {
            node = new ValueNode(std::stoi(ctx->INT()->getText()));
        }
        tokens.addNode(node);
        return node->tokenId;
    }

private:
    ExprNodePtr createBinaryNodeHelper(BinaryKind kind, vcalc::VCalcParser::ExpContext* ctx)
    {
        auto lhs_token = std::any_cast<size_t>(visit(ctx->exp(0)));
        auto rhs_token = std::any_cast<size_t>(visit(ctx->exp(1)));
        auto lhs = tokens.getNode<ExprNodePtr>(lhs_token);
        auto rhs = tokens.getNode<ExprNodePtr>(rhs_token);
        auto node = new BinaryNode { kind, lhs, rhs };
        ExprNodePtr out = node;

        auto node_valueType = node->getValueType();
        auto lhs_valueType = lhs->getValueType();
        auto rhs_valueType = rhs->getValueType();

        assert(node_valueType != ValueType::Unknow);
        assert(lhs_valueType != ValueType::Unknow);
        assert(rhs_valueType != ValueType::Unknow);

        if (lhs_valueType != rhs_valueType) {
            auto res = valueTypeCmp(lhs_valueType, rhs_valueType);
            // add type convert node(ConvertNode) for the little one
            // replace the binaryNode's lhs or rhs then
            if (res > 0) {
                node->rhs = new ConvertNode { lhs->getValueType(), lhs->vecAttr, rhs };
                tokens.addNode(node->rhs);
            } else {
                node->lhs = new ConvertNode { rhs->getValueType(), rhs->vecAttr, lhs };
                tokens.addNode(node->lhs);
            }
        } else {
            // lhs and rhs is the same type, but not same as node
            if (node_valueType != lhs_valueType) {
                // force node's type as lhs and rhs's type
                // add type convert node(ConvertNode) for node
                node->valueType = lhs_valueType;
                // NOVE: no vector attr for convert node
                auto upper_node = new ConvertNode { node_valueType, nullptr, node };
                // can't touch node anymore, so we need record tokens
                tokens.addNode(node);
                // return convert node as tree root
                out = upper_node;
            }
        }
        return out;
    }

    static int valueTypeCmp(ValueType lhs, ValueType rhs)
    {
        int val[4] = {
            0, 1, 2, 1000
        };
        int v1 = val[static_cast<int>(lhs)];
        int v2 = val[static_cast<int>(rhs)];
        if (v1 == v2)
            return 0;
        return v1 > v2 ? 1 : -1;
    }

    BinaryKind getBinaryKind(std::string op)
    {
        if (op == "..") {
            return BinaryKind::Range;
        } else if (op == "+") {
            return BinaryKind::Add;
        } else if (op == "-") {
            return BinaryKind::Sub;
        } else if (op == "*") {
            return BinaryKind::Mul;
        } else if (op == "/") {
            return BinaryKind::Div;
        } else if (op == "and") {
            return BinaryKind::And;
        } else if (op == "or") {
            return BinaryKind::Or;
        } else if (op == "!=") {
            return BinaryKind::Ne;
        } else if (op == "==") {
            return BinaryKind::Eq;
        } else if (op == "<") {
            return BinaryKind::Lt;
        } else if (op == "<=") {
            return BinaryKind::Le;
        } else if (op == ">") {
            return BinaryKind::Gt;
        } else if (op == ">=") {
            return BinaryKind::Ge;
        } else {
            throw std::runtime_error("getBinaryKind: unknown op");
        }
    }
};

}
