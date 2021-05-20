using CUDA
using BenchmarkTools

N = 10000000
dim = 2

A = randn(Float32, dim, dim)
B = randn(Float32, dim, N)
C = zeros(Float32, dim, N)

AC = CUDA.randn(Float32, dim, dim)
BC = CUDA.randn(Float32, dim, N)
CC = CUDA.zeros(Float32, dim, N)


function cpu_matmul(C, A, B)
    C .= A * B
end

@btime cpu_matmul(C, A, B)

function gpu(CC, AC, BC)
    CC .= AC * BC
end

@btime CUDA.@sync gpu(CC, AC, BC)


#=

Comparing matrix multiplication time on CPU & GPU

Using a 2080Ti & Ryzen 9 3900

17.928 ms (2 allocations: 76.29 MiB)
 2.124 ms (56 allocations: 1.34 KiB)

=#





