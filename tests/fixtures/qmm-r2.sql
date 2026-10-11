-- Synthetic second round. Unsupported n=4 sample must not enter the N=4 median.
CREATE TABLE test_backend_ops (op_name TEXT, op_params TEXT, test_mode TEXT, supported INTEGER, time_us REAL);
INSERT INTO test_backend_ops VALUES ('MUL_MAT','type_a=q4_K,type_b=f32,m=4,n=1,k=256,bs=[1,1],nr=[1,1]','perf',1,8);
INSERT INTO test_backend_ops VALUES ('MUL_MAT','type_a=q4_K,type_b=f32,m=4,n=2,k=256,bs=[1,1],nr=[1,1]','perf',1,16);
INSERT INTO test_backend_ops VALUES ('MUL_MAT','type_a=q4_K,type_b=f32,m=4,n=4,k=256,bs=[1,1],nr=[1,1]','perf',0,999);
INSERT INTO test_backend_ops VALUES ('MUL_MAT','type_a=q4_K,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1]','perf',1,16);
INSERT INTO test_backend_ops VALUES ('MUL_MAT','type_a=q4_K,type_b=f32,m=16,n=2,k=256,bs=[1,1],nr=[1,1]','perf',1,32);
INSERT INTO test_backend_ops VALUES ('MUL_MAT','type_a=q4_K,type_b=f32,m=16,n=4,k=256,bs=[1,1],nr=[1,1]','perf',1,64);
