#pragma once

#include <algorithm>
#include <assert.h>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ast {

struct Node;
struct BlockNode;
struct StateNode;
struct ExprNode;
struct ValueNode;
struct BinaryNode;
struct VarDeclareNode;
struct VarAssignNode;
struct PrintNode;
struct IfNode;
struct LoopNode;
struct VarScope;

using NodePtr = Node*;
using BlockNodePtr = BlockNode*;
using StateNodePtr = StateNode*;
using ExprNodePtr = ExprNode*;
using ValueNodePtr = ValueNode*;
using BinaryNodePtr = BinaryNode*;
using VarDeclareNodePtr = VarDeclareNode*;
using VarAssignNodePtr = VarAssignNode*;
using PrintNodePtr = PrintNode*;
using IfNodePtr = IfNode*;
using LoopNodePtr = LoopNode*;
using VarScopePtr = VarScope*;

enum class ValueType {
    Boolean = 0,
    Int,
    Vector,
    Unknow,
};

struct Node {
    enum class Type {
        Block,
        Stat,
        Expr,
    };
    Type nodeType;
    size_t tokenId = 0;
    Node(Type type)
        : nodeType(type)
    {
    }
    virtual ~Node() = default;
    Type getNodeKind() { return nodeType; }
};

struct StateNode : public Node {
    enum class Type {
        VarDeclare,
        VarAssign,
        Print,
        If,
        Loop,
    };
    Type statType;
    explicit StateNode(Type type)
        : Node(Node::Type::Stat)
        , statType(type)
    {
    }
    Type getStatKind() const { return statType; }
};

struct VarDeclareNode : public StateNode {
    ValueNodePtr value;
    ExprNodePtr expr;

    explicit VarDeclareNode(ValueNodePtr value, ExprNodePtr expr)
        : StateNode(Type::VarDeclare)
        , value(std::move(value))
        , expr(std::move(expr))
    {
    }
};

struct VarAssignNode : public StateNode {
    ValueNodePtr var;
    ExprNodePtr expr;

    explicit VarAssignNode(ValueNodePtr var, ExprNodePtr expr)
        : StateNode(Type::VarAssign)
        , var(std::move(var))
        , expr(std::move(expr))
    {
    }
};

struct PrintNode : public StateNode {
    ExprNodePtr expr;
    explicit PrintNode(ExprNodePtr expr)
        : StateNode(Type::Print)
        , expr(std::move(expr))
    {
    }
};

struct IfNode : public StateNode {
    ExprNodePtr expr;
    BlockNodePtr body;
    explicit IfNode(ExprNodePtr expr, BlockNodePtr body)
        : StateNode(Type::If)
        , expr(std::move(expr))
        , body(std::move(body))
    {
    }
};

struct LoopNode : public StateNode {
    ExprNodePtr expr;
    BlockNodePtr body;
    explicit LoopNode(ExprNodePtr expr, BlockNodePtr body)
        : StateNode(Type::Loop)
        , expr(std::move(expr))
        , body(std::move(body))
    {
    }
};

struct ExprNode : public Node {
    enum class Type {
        Value,
        Binary,
    };
    Type exprType;
    ValueType valueType;

    explicit ExprNode(Type exprType, ValueType valueType)
        : Node(Node::Type::Expr)
        , exprType(exprType)
        , valueType(valueType)
    {
    }
    Type getExprKind() const { return exprType; }
    ValueType getValueType() const { return valueType; }
    std::string getValueTypeStr() const
    {
        switch (valueType) {
        case ValueType::Boolean:
            return "boolean";
        case ValueType::Int:
            return "int";
        case ValueType::Vector:
            return "vector";
        case ValueType::Unknow:
            return "unknow";
        }
        return "unknow";
    }
};

struct ValueNode : public ExprNode {
    enum class ValueKind {
        Var,
        Const,
    };
    ValueKind valueKind;
    std::string name;
    int const_val;

    ValueNode(int const_val)
        : ExprNode(Type::Value, ValueType::Int)
        , valueKind(ValueKind::Const)
        , const_val(const_val)
    {
    }

    // NOVE: var's ValueType can be set by Vardef or query from parent scope
    ValueNode(std::string name)
        : ExprNode(Type::Value, ValueType::Unknow)
        , valueKind(ValueKind::Var)
        , name(name)
    {
    }

    ValueKind getValueKind() const { return valueKind; }
};

struct BinaryNode : public ExprNode {
    enum class BinaryKind : int {
        Range, // x..y, generate a vector. x,y must be integer
        Add,
        Sub,
        Mul,
        Div,
        // Logical operators start
        And,
        Or,
        Ne, // Not equal
        Eq, // Equal
        Gt, // Greater than
        Lt, // Less than
        Ge, // Greater than or equal
        Le, // Less than or equal
    };
    BinaryKind binaryKind;
    ExprNodePtr lhs;
    ExprNodePtr rhs;
    explicit BinaryNode(BinaryKind binaryKind, ExprNodePtr lhs, ExprNodePtr rhs)
        : ExprNode(Type::Binary, ValueType::Unknow)
        , binaryKind(binaryKind)
        , lhs(lhs)
        , rhs(rhs)
    {
        bool has_vector = lhs->getValueType() == ValueType::Vector || rhs->getValueType() == ValueType::Vector;
        if (binaryKind == BinaryKind::Range) {
            valueType = ValueType::Vector;
        } else if (binaryKind >= BinaryKind::And && !has_vector) {
            valueType = ValueType::Boolean;
        }
        if (valueType != ValueType::Unknow) {
            return;
        }
        /**
         * create a matrix to figure out the result type of binary expression
         * Like this:
         * 0 1 2 3
         * 1 1 2 3
         * 2 2 2 3
         * 3 3 3 3
         */
        ValueType lhs_type = lhs->getValueType(), rhs_type = rhs->getValueType();
        ValueType result[4][4] = {
            { ValueType::Boolean, ValueType::Int, ValueType::Vector, ValueType::Unknow },
            { ValueType::Int, ValueType::Int, ValueType::Vector, ValueType::Unknow },
            { ValueType::Vector, ValueType::Vector, ValueType::Vector, ValueType::Unknow },
            { ValueType::Unknow, ValueType::Unknow, ValueType::Unknow, ValueType::Unknow },
        };
        valueType = result[static_cast<int>(lhs_type)][static_cast<int>(rhs_type)];
    }
    BinaryKind getBinaryKind() const { return binaryKind; }
};

struct BlockNode : public Node {
    std::vector<StateNodePtr> body;
    BlockNode()
        : Node(Node::Type::Block)
    {
    }
    virtual ~BlockNode()
    {
        for (auto& node : body) {
            delete node;
        }
    }
    void addStat(StateNodePtr node) { body.push_back(node); }
    auto begin() { return body.begin(); }
    auto end() { return body.end(); }
};

struct VarScope {
    BlockNodePtr block; // The block that this scope belongs to (not the block that this scope defines)
    VarScopePtr parent;
    std::map<std::string, ValueNodePtr> vars;

    VarScope(BlockNodePtr block, VarScopePtr parent)
        : block(block)
        , parent(parent)
    {
    }

    void add(ValueNodePtr var)
    {
        assert(var->getValueKind() == ValueNode::ValueKind::Var);
        vars[var->name] = var;
    }

    ValueNodePtr find(std::string name)
    {
        auto it = vars.find(name);
        if (it != vars.end()) {
            return it->second;
        }
        if (parent) {
            return parent->find(name);
        }
        return nullptr;
    }
};

using NodeKind = Node::Type;
using StatKind = StateNode::Type;
using ExprKind = ExprNode::Type;
using ValueKind = ValueNode::ValueKind;
using BinaryKind = BinaryNode::BinaryKind;

} // namespace ast
