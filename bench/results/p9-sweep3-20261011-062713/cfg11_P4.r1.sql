CREATE TABLE IF NOT EXISTS test_backend_ops (
  test_time TEXT,
  build_commit TEXT,
  backend_name TEXT,
  op_name TEXT,
  op_params TEXT,
  test_mode TEXT,
  supported INTEGER,
  passed INTEGER,
  error_message TEXT,
  time_us REAL,
  flops REAL,
  bandwidth_gb_s REAL,
  memory_kb INTEGER,
  n_runs INTEGER,
  device_description TEXT,
  backend_reg_name TEXT
);

[New LWP 2970505]
[New LWP 2970504]
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/usr/lib/x86_64-linux-gnu/libthread_db.so.1".
0x000075e9fc47f913 in wait4 () from /usr/lib/x86_64-linux-gnu/libc.so.6
#0  0x000075e9fc47f913 in wait4 () from /usr/lib/x86_64-linux-gnu/libc.so.6
#1  0x000055e4f2effa23 in ggml_print_backtrace ()
#2  0x000055e4f2effbc7 in ggml_abort ()
#3  0x000055e4f1fa9047 in ggml_cuda_error(char const*, char const*, char const*, int, char const*) ()
#4  0x000055e4f1ffb2f4 in void mul_mat_q_volta_launch<(ggml_type)12, 1>(ggml_backend_cuda_context&, char const*, __half const*, __half const*, float*, int, int, int, long, long) ()
#5  0x000055e4f1ffbc81 in ggml_cuda_mul_mat_q_volta(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*) ()
#6  0x000055e4f1fc2bb1 in ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) ()
#7  0x000055e4f2f1cc35 in ggml_backend_graph_compute ()
#8  0x000055e4f1d5f749 in main ()
[Inferior 1 (process 2970503) detached]
