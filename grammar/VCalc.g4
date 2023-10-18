// reference: grammar Lua
grammar VCalc;
file
    : block EOF
    ;

block
    : stat*
    ;

stat
    : ';'
    | vardef ';'
    | loop ';'
    | if ';'
    | print ';'
    | assign ';'
    ;

assign: var '=' exp;

vardef
    :   'int' var ('=' exp)?
    |   'vector' var ('=' exp)?
    ;

loop
    : 'loop' '(' exp ')' block 'pool'
    ;

if
    : 'if' '(' exp ')' block 'fi'
    ;

print
    : 'print' '(' exp ')'
    ;

exp
    : varOrExp
    | '[' var 'in' exp operatorGenerator exp ']'
    | '[' var 'in' exp operatorFilter exp ']'
    | exp '[' exp ']'
    | exp operatorRange exp
    | exp operatorMulDiv exp
    | exp operatorAddSub exp
    | exp operatorComparison exp
    | exp operatorAnd exp
    | exp operatorOr exp
    ;

varOrExp
    : var | '(' exp ')'
    ;

var
    : INT
    | NAME
    ;

operatorGenerator
    : '|';

operatorFilter
    : '&';

operatorRange
    : '..';

operatorOr
	: 'or';

operatorAnd
	: 'and';

operatorComparison
	: '<' | '>' | '<=' | '>=' | '~=' | '==' | '!=' ;

operatorAddSub
	: '+' | '-';

operatorMulDiv
	: '*' | '/' ;

// LEXER

NAME
    : [a-zA-Z_][a-zA-Z_0-9]*
    ;

INT
    : Digit+
    ;

fragment
Digit
    : [0-9]
    ;

WS
    : [ \t\u000C\r\n]+ -> skip
    ;