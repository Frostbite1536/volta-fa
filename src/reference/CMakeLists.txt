add_library(volta_fa_ref STATIC attention_reference.cpp attention_tiled_model.cpp)
target_include_directories(volta_fa_ref PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Keep the CPU error sweep practical even in the default (untyped) build.
  # No fast-math: the explicit FP32 rounding points are part of the model.
  target_compile_options(volta_fa_ref PRIVATE -O2 -ffp-contract=off)
endif()
