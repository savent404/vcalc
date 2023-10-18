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
    ;

explist
    : (exp ',')* exp
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
    : number
    | varOrExp var*
    | exp operatorMulDivMod exp
    | exp operatorAddSub exp
    | exp operatorComparison exp
    | exp operatorAnd exp
    | exp operatorOr exp
    ;

varOrExp
    : var | '(' exp ')'
    ;

var
    : NAME
    ;

operatorOr
	: 'or';

operatorAnd
	: 'and';

operatorComparison
	: '<' | '>' | '<=' | '>=' | '~=' | '==' | '!=' ;

operatorAddSub
	: '+' | '-';

operatorMulDivMod
	: '*' | '/' | '%' | '//';

number
    : INT
    ;

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