grammar VCalc;

fragment DIGIT: [0-9];
fragment CHAR: [a-zA-Z];

LB: '[';
RB: ']';
BAR: '|';
AMP: '&';
TO: '..';

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

INT: 'int';
VEC: 'vector';
TYPE: INT | VEC;
EQ: '=';
IN: 'in';
IF: 'if';
FI: 'fi';
LOOP: 'loop';
POOL: 'pool';
PRINT: 'print';
SC: ';';

INT_LIT: DIGIT+;
ID: CHAR+ (CHAR | DIGIT)*;
WS: [ \t\r\n]+ -> skip;                 // Skip whitespace

file: statement* EOF;

statement: type=TYPE name=ID EQ exp=expression SC       #decl
         | name=ID EQ exp=expression SC                 #assn
         | IF LP exp=expression RP statement* FI SC     #cond
         | LOOP LP exp=expression RP statement* POOL SC #loop
         | PRINT LP exp=expression RP SC                #prnt
         ;

expression: LP exp=expression RP                                #parens
          | LB id=ID IN rnge=expression BAR exp=expression RB   #generator
          | LB id=ID IN rnge=expression AMP pred=expression RB  #filter
          | val=expression LB idx=expression RB                 #index
          | from=expression TO to=expression                    #range
          | l=expression op=(MUL|DIV) r=expression              #op
          | l=expression op=(ADD|SUB) r=expression              #op
          | l=expression op=(LT|GT) r=expression                #op
          | l=expression op=(EE|NE) r=expression                #op
          | INT_LIT                                             #lit
          | ID                                                  #id
          ;