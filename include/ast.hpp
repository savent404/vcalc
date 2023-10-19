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
struct ConvertNode;
struct VecAttr;

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
using ConvertNodePtr = ConvertNode*;
using VecAttrPtr = VecAttr*;

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

struct VecAttr {
    ExprNodePtr start;
    ExprNodePtr end;
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
        Convert,
    };
    Type exprType;
    ValueType valueType;
    VecAttrPtr vecAttr;

    explicit ExprNode(Type exprType, ValueType valueType, VecAttrPtr attr)
        : Node(Node::Type::Expr)
        , exprType(exprType)
        , valueType(valueType)
        , vecAttr(attr)
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

struct ConvertNode : public ExprNode {
    enum class CvtKind {
        Bool2Int,
        Bool2Vec,
        Int2Bool,
        Int2Vec,
        Invalid,
    };
    CvtKind unaryKind;
    ExprNodePtr exp;

    ConvertNode(ValueType type, VecAttrPtr attr, ExprNodePtr exp)
        : ExprNode(Type::Convert, type, attr)
        , unaryKind(CvtKind::Bool2Int)
        , exp(std::move(exp))
    {
        CvtKind matrix[4][4] = {
            {
                /* 0,0 */ CvtKind::Invalid,
                /* 0,1 */ CvtKind::Int2Bool,
                /* 0,2 */ CvtKind::Invalid,
                /* 0,3 */ CvtKind::Invalid,
            },
            {
                /* 1,0 */ CvtKind::Bool2Int,
                /* 1,1 */ CvtKind::Invalid,
                /* 1,2 */ CvtKind::Invalid,
                /* 1,3 */ CvtKind::Invalid,
            },
            {
                /* 2,0 */ CvtKind::Bool2Vec,
                /* 2,1 */ CvtKind::Int2Vec,
                /* 2,2 */ CvtKind::Invalid,
                /* 2,3 */ CvtKind::Invalid,
            },
            {
                /* 3,0 */ CvtKind::Invalid,
                /* 3,1 */ CvtKind::Invalid,
                /* 3,2 */ CvtKind::Invalid,
                /* 3,3 */ CvtKind::Invalid,
            }

        };
        int v1 = static_cast<int>(type);
        int v2 = static_cast<int>(exp->getValueType());

        unaryKind = matrix[v1][v2];
        assert(unaryKind != CvtKind::Invalid);
    }

    CvtKind getCvtKind() const { return unaryKind; }
    const std::string getCvtKindStr() const
    {
        switch (unaryKind) {
        case CvtKind::Bool2Int:
            return "bool->int";
        case CvtKind::Bool2Vec:
            return "bool->vec";
        case CvtKind::Int2Bool:
            return "int->bool";
        case CvtKind::Int2Vec:
            return "int->vec";
        case CvtKind::Invalid:
            return "invalid";
        }
        return "invalid";
    }
};

struct ValueNode : public ExprNode {
    enum class ValueKind {
        Var,
        Const,
    };
    ValueKind valueKind;
    std::string name; // works when valueKind == ValueKind::Var
    int const_val; // works when valueKind == ValueKind::Const

    ValueNode(int const_val)
        : ExprNode(Type::Value, ValueType::Int, nullptr)
        , valueKind(ValueKind::Const)
        , const_val(const_val)
    {
    }

    // NOVE: var's ValueType can be set by Vardef or query from parent scope
    ValueNode(std::string name)
        : ExprNode(Type::Value, ValueType::Unknow, nullptr)
        , valueKind(ValueKind::Var)
        , name(name)
    {
    }

    ValueNode(ExprNodePtr start, ExprNodePtr end)
        : ExprNode(Type::Value, ValueType::Vector, nullptr)
    {
        // estimate if vector is const
        if (start->getExprKind() == end->getExprKind() && start->getExprKind() == ExprNode::Type::Value) {
            auto valueStart = reinterpret_cast<ValueNodePtr>(start);
            auto valueEnd = reinterpret_cast<ValueNodePtr>(end);
            if (valueStart->getValueKind() == valueEnd->getValueKind()
                && valueStart->getValueKind() == ValueNode::ValueKind::Const) {
                valueKind = ValueKind::Const;
            }
        } else {
            valueKind = ValueKind::Var;
        }

        auto attr = new VecAttr();
        attr->start = start;
        attr->end = end;
        vecAttr = attr;
    }

    ValueKind getValueKind() const { return valueKind; }

    bool isImplicitVar() const
    {
        return valueKind == ValueKind::Var && name.empty();
    }
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

    // TODO: can't figure out the vector size in compile time, need a runtime operation to get the size
    explicit BinaryNode(BinaryKind binaryKind, ExprNodePtr lhs, ExprNodePtr rhs)
        : ExprNode(Type::Binary, ValueType::Unknow, nullptr)
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
    NodePtr node; // The node that this scope belongs to (not the node that this scope defines)
    VarScopePtr parent;
    std::map<std::string, ValueNodePtr> vars;

    VarScope(NodePtr node, VarScopePtr parent)
        : node(node)
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
