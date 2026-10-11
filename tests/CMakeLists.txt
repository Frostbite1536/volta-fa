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

  # Synthetic QMM fixtures; arithmetic is documented in qmm-dump.txt and qmm-r1.sql.
  set(qmm_script ${PROJECT_SOURCE_DIR}/bin/qmatmul-table.py)
  set(qmm_dump ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/qmm-dump.txt)
  set(qmm_r1 ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/qmm-r1.sql)
  set(qmm_rounds ${qmm_r1},${CMAKE_CURRENT_SOURCE_DIR}/fixtures/qmm-r2.sql)
  add_test(NAME qmm_shapes COMMAND ${Python3_EXECUTABLE} ${qmm_script} shapes ${qmm_dump})
  # Two layer weights: 2*576 B = 1152 B = 0.001099 MiB; total known bytes=3136.
  set_tests_properties(qmm_shapes PROPERTIES PASS_REGULAR_EXPRESSION
    "ffn_up.weight \\| Q4_K \\| 256 \\| 4 \\| 2 \\| 0\\.001099.*Q4_K \\| 5120 \\| 0\\.002747.*Total matmul weights: 3136 bytes.*output.weight: 576 bytes.*unknown type: MYSTERY")
  add_test(NAME qmm_perf_rounds COMMAND ${Python3_EXECUTABLE} ${qmm_script} perf ${qmm_rounds} --labels rounds)
  # 576/6/1000=0.096 GB/s; 2304/12/1000=0.192 GB/s; max spread=(8-4)/6.
  set_tests_properties(qmm_perf_rounds PROPERTIES PASS_REGULAR_EXPRESSION
    "Q4_K \\| 4 \\| 256 \\| 6\\.000 \\| 12\\.000 \\| 24\\.000.*Q4_K \\| 16 \\| 256 \\| 12\\.000 \\| 24\\.000 \\| 48\\.000.*Q4_K \\| 4 \\| 256 \\| 0\\.096 \\| 0\\.048 \\| 0\\.024.*Q4_K \\| 16 \\| 256 \\| 0\\.192 \\| 0\\.096 \\| 0\\.048.*Max round spread .*66\\.67%")
  add_test(NAME qmm_perf_groups
    COMMAND ${Python3_EXECUTABLE} ${qmm_script} perf ${qmm_r1} ${qmm_rounds} --labels single,rounds)
  set_tests_properties(qmm_perf_groups PROPERTIES PASS_REGULAR_EXPRESSION
    "### single.*Q4_K \\| 4 \\| 256 \\| 4\\.000 \\| 8\\.000 \\| 24\\.000.*### rounds.*Q4_K \\| 4 \\| 256 \\| 6\\.000 \\| 12\\.000 \\| 24\\.000")
  add_test(NAME qmm_step COMMAND ${Python3_EXECUTABLE} ${qmm_script} step ${qmm_dump} ${qmm_rounds} --tokens 1,2,4,8,16)
  # Median bandwidth at n=1 is (0.096+0.192)/2=0.144 GB/s -> 1152/144=8 us.
  # Exact time=2*6=12 us; total=20 us; coverage=1152/2560=45%. n=2,4 scale.
  set_tests_properties(qmm_step PROPERTIES PASS_REGULAR_EXPRESSION
    "output.weight is excluded.*1 \\| 12\\.000 \\| 8\\.000 \\| 20\\.000 \\| 1152/2560 \\| 45\\.00% \\| 2.*2 \\| 24\\.000 \\| 16\\.000 \\| 40\\.000.*4 \\| 48\\.000 \\| 32\\.000 \\| 80\\.000.*8 \\| 0\\.000 \\| 0\\.000 \\| 0\\.000 \\| 0/2560 \\| 0\\.00% \\| 5.*16 \\| 0\\.000 \\| 0\\.000 \\| 0\\.000 \\| 0/2560 \\| 0\\.00% \\| 5")
  add_test(NAME qmm_step_missing COMMAND ${Python3_EXECUTABLE} ${qmm_script} step ${qmm_dump} ${qmm_rounds} --tokens 1)
  set_tests_properties(qmm_step_missing PROPERTIES PASS_REGULAR_EXPRESSION
    "1 \\| F16 \\| 4 \\| 32 \\| 1 \\| not measured.*1 \\| MYSTERY \\| 4 \\| 32 \\| 1 \\| not measured.*1 \\| Q4_K \\| 8 \\| 256 \\| 1 \\| estimated \\| 0\\.144 \\| 8\\.000")
  add_test(NAME qmm_step_with_output
    COMMAND ${Python3_EXECUTABLE} ${qmm_script} step ${qmm_dump} ${qmm_rounds} --tokens 1 --with-output)
  # Including output adds 6 us and 576 exact bytes: 1728/3136=55.1020408%.
  set_tests_properties(qmm_step_with_output PROPERTIES PASS_REGULAR_EXPRESSION
    "output.weight is included.*1 \\| 18\\.000 \\| 8\\.000 \\| 26\\.000 \\| 1728/3136 \\| 55\\.10% \\| 2")
endif()
