foreach(name test_fp16 test_reference test_tiled_model error_budget)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE volta_fa_ref)
endforeach()
add_test(NAME fp16 COMMAND test_fp16)
foreach(case tiny uniform double gqa causal masked softcap large validation)
  add_test(NAME reference_${case} COMMAND test_reference ${case})
endforeach()
foreach(case matrix masked rounding validation)
  add_test(NAME tiled_${case} COMMAND test_tiled_model ${case})
endforeach()
add_test(NAME error_budget COMMAND error_budget)


# Needs the CUDA runtime (top-level block); skipped on CPU-only builds.
if(TARGET volta_fa_p2p)
  add_executable(test_p2p test_p2p.cpp)
  target_link_libraries(test_p2p PRIVATE volta_fa_p2p)
  foreach(case probe env staged_sizes stream_order_before buffer_reuse peer)
    add_test(NAME p2p_${case} COMMAND test_p2p ${case})
    set_tests_properties(p2p_${case} PROPERTIES SKIP_RETURN_CODE 77)
  endforeach()
endif()

# bench/analyze.py must parse real test-backend-ops SQL output (fixture: CPU run at bench/LLAMA_COMMIT).
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME analyze_tbo
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/bench/analyze.py tbo
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tbo-perf-cpu.sql ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tbo-perf-cpu.sql)
  set_tests_properties(analyze_tbo PROPERTIES PASS_REGULAR_EXPRESSION "type_V=q8_0 \\| [0-9.]+ ms \\| [0-9.]+ ms \\| 1\\.00×")
  # llama-bench rounds (bench/patches/run.sh): a comma-joined group is summarised as median (min-max, n).
  add_test(NAME analyze_lb_rounds
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/bench/analyze.py lb
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lb-r1.jsonl,${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lb-r2.jsonl
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lb-r3.jsonl --labels a,b)
  set_tests_properties(analyze_lb_rounds PROPERTIES PASS_REGULAR_EXPRESSION "105\\.00 \\(100\\.00–110\\.00, n=2\\) \\| 130\\.00 ± 1\\.00 \\| \\+23\\.8%")
endif()
