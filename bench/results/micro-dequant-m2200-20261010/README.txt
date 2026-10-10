Laptop GPU (Quadro M2200, sm_52, 8 SMs, ~88 GB/s peak), NOT a V100: shows the mechanism and bit-exactness only.
Built: nvcc 12.8 -O3 -arch=sm_52 bench/micro/dequant_nc.cu; run: dequant_nc 20
