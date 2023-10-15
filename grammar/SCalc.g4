grammar SCalc;

fragment DIGIT: [0-9];
fragment CHAR: [a-zA-Z];

LP: '(';
RP: ')';
MUL: '*';
DIV: '/';
ADD: '+';
SUB: '-';

LT: '<';
GT: '>';
EE: '==';
NE: '!=';

TYPE: 'int';
EQ: '=';
IF: 'if';
FI: 'fi';
LOOP: 'loop';
POOL: 'pool';
PRINT: 'print';
SC: ';';

INT_LIT: DIGIT+;
ID: CHAR+ (CHAR | DIGIT)*;
WS: [ \t\r\n]+ -> skip;                 // Skip whitespace

file: full_statement* EOF;

full_statement: declaration
              | part_statement
              ;

declaration:    type=TYPE name=ID EQ exp=expression SC              #decl;
part_statement: name=ID EQ exp=expression SC                        #assn
              | IF LP exp=expression RP part_statement* FI SC       #cond
              | LOOP LP exp=expression RP part_statement* POOL SC   #loop
              | PRINT LP exp=expression RP SC                       #prnt
              ;

expression: LP expression RP                        #parens
          | expression op=(MUL|DIV) expression      #op
          | expression op=(ADD|SUB) expression      #op
          | expression op=(LT|GT) expression        #op
          | expression op=(EE|NE) expression        #op
          | INT_LIT                                 #lit
          | ID                                      #id
          ;