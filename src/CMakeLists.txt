# Gather our source files in this directory.
set(
  scalc_src_files
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
        Interpreter.cpp
        ../include/Interpreter.h
        Backend.cpp
        ../include/Backend.h
        ../include/StringInterface.h
        ../include/StringInterface.h
        RISCV.cpp
        ../include/RISCV.h
        X86.cpp
        ../include/X86.h
        ARM.cpp
        ../include/ARM.h
)

# Build our executable from the source files.
add_executable(scalc ${scalc_src_files}
        ../include/StringInterface.h)
target_include_directories(scalc PUBLIC ${ANTLR_GEN_DIR})

# Ensure that the antlr4-runtime is available.
add_dependencies(scalc antlr)

# Add the antlr runtime and parser as libraries to link.
target_link_libraries(scalc parser antlr4-runtime)

# Symbolic link our executable to the base directory so we don't have to go searching for it.
symlink_to_bin("scalc")
