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

INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-10-09T20:24:18Z', '79e2e74eb', 'CPU', 'FLASH_ATTN_EXT', 'hsk=64,hsv=64,nh=8,nr23=[8,1],kv=7680,nb=1,mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=f16,type_V=f16,permute=[0,1,2,3],kv_view=1,v_is_view_of_k=0,n_kv_max=0', 'perf', '1', '1', '', '37615.109375', '3345174906.858821', '0.000000', '28847', '64', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-10-09T20:24:22Z', '79e2e74eb', 'CPU', 'FLASH_ATTN_EXT', 'hsk=64,hsv=64,nh=8,nr23=[8,1],kv=7680,nb=1,mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=q4_0,type_V=q4_0,permute=[0,1,2,3],kv_view=1,v_is_view_of_k=0,n_kv_max=0', 'perf', '1', '1', '', '49578.765625', '2537963953.191906', '0.000000', '8147', '64', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-10-09T20:24:27Z', '79e2e74eb', 'CPU', 'FLASH_ATTN_EXT', 'hsk=64,hsv=64,nh=8,nr23=[8,1],kv=7680,nb=1,mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=[0,1,2,3],kv_view=1,v_is_view_of_k=0,n_kv_max=0', 'perf', '1', '1', '', '57898.171875', '2173283126.653124', '0.000000', '15347', '64', '', '');
