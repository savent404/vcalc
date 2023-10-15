#include "SCalcLexer.h"
#include "SCalcParser.h"

#include "ANTLRFileStream.h"
#include "CommonTokenStream.h"
#include "tree/ParseTree.h"

#include <iostream>
#include <string>

#include "Interpreter.h"
#include "Backend.h"
#include "RISCV.h"
#include "X86.h"
#include "ARM.h"

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cout << "Missing required argument.\n"
              << "Required arguments: <mode> <input file path> <output file path>\n";
    return 1;
  }

  antlr4::ANTLRFileStream afs;
  afs.loadFromFile(argv[2]);
  scalc::SCalcLexer lexer(&afs);
  antlr4::CommonTokenStream tokens(&lexer);
  scalc::SCalcParser parser(&tokens);
  antlr4::tree::ParseTree * tree = parser.file();
  std::ofstream ostream(argv[3]);
  std::string mode(argv[1]);

  if (mode == "interpreter") {
    Interpreter i;
    i.visit(tree);
    ostream << i.get_output();
  } else if (mode == "riscv") {
      RISCV rv;
      Backend b(&rv);
      ostream << std::any_cast<std::string>(b.visit(tree));
  } else if (mode == "x86") {
      X86 x;
      Backend b(&x);
      ostream << std::any_cast<std::string>(b.visit(tree));
  } else if (mode == "arm") {
      ARM a;
      Backend b(&a);
      ostream << std::any_cast<std::string>(b.visit(tree));
  }

  return 0;
}
