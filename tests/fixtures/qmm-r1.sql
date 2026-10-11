-- Synthetic timings (microseconds), not benchmark data.
-- N=4, K=256: 576 B; medians over r1/r2 at n=1,2,4 are 6,12,24 us.
-- N=16, K=256: 2304 B; medians are 12,24,48 us.
-- n=1 bandwidths: 0.096,0.192 GB/s; median=0.144 -> missing N=8 costs 8 us.
-- Default step: 2*6+8=20 us; coverage=1152/2560=45%.
-- With output: 3*6+8=26 us; coverage=1728/3136=55.1020408%.
-- n=2 and n=4 times scale by 2 and 4. Max round spread=(8-4)/6=66.6667%.
CREATE TABLE test_backend_ops (op_name TEXT, op_params TEXT, test_mode TEXT, supported INTEGER, time_us REAL);
INSERT INTO test_backend_ops VALUES ('MUL_MAT','type_a=q4_K,type_b=f32,m=4,n=1,k=256,bs=[1,1],nr=[1,1]','perf',1,4);
INSERT INTO test_backend_ops VALUES ('MUL_MAT','type_a=q4_K,type_b=f32,m=4,n=2,k=256,bs=[1,1],nr=[1,1]','perf',1,8);
INSERT INTO test_backend_ops VALUES ('MUL_MAT','type_a=q4_K,type_b=f32,m=4,n=4,k=256,bs=[1,1],nr=[1,1]','perf',1,24);
INSERT INTO test_backend_ops VALUES ('MUL_MAT','type_a=q4_K,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1]','perf',1,8);
INSERT INTO test_backend_ops VALUES ('MUL_MAT','type_a=q4_K,type_b=f32,m=16,n=2,k=256,bs=[1,1],nr=[1,1]','perf',1,16);
INSERT INTO test_backend_ops VALUES ('MUL_MAT','type_a=q4_K,type_b=f32,m=16,n=4,k=256,bs=[1,1],nr=[1,1]','perf',1,32);
INSERT INTO test_backend_ops VALUES ('MUL_MAT','type_a=q8_0,type_b=f32,m=4,n=1,k=32,bs=[1,1],nr=[1,1]','perf',0,0);
INSERT INTO test_backend_ops VALUES ('ADD','type=f32','perf',1,999);
INSERT INTO test_backend_ops VALUES ('MUL_MAT','type_a=f16,type_b=f32,m=4,n=1,k=32,bs=[1,1],nr=[1,1]','test',1,999);
