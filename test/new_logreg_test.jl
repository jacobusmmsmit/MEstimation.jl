using Test

# @testset "logistic_regression using the new obj implementation" begin
#     using MEstimation
#     using Random
#     using Distributions
#     using LinearAlgebra
#     using Optim

#     include("new_logistic_regression.jl")

#     const GCACHE = Dict{DataType,Any}()

#     Random.seed!(123);
#     n = 500;
#     p = 10;
#     x = rand(n, p);
#     x[:, 1] = fill(1.0, n);
#     true_betas = rand(p) * 0.02;
#     my_data = simulate(true_betas, x, fill(1, n));

#     logistic_template = objective_function_template(
#         new_logistic_regression.nobs,
#         new_logistic_regression.observation,
#         new_logistic_regression.loglik
#     )    
    
# end

using Random
using Distributions
using LinearAlgebra
using Optim
using ReverseDiff
using ForwardDiff

include("new_logistic_regression.jl")
include("../src/objective_functions.jl")


Random.seed!(123);
n = 50;
p = 10;
x = rand(n, p);
x[:, 1] = fill(1.0, n);
true_betas = rand(p) * 0.02;
my_data = new_logistic_regression.simulate(true_betas, x, fill(1, n));

logistic_template = objective_function_template(
    new_logistic_regression.nobs,
    new_logistic_regression.observation,
    new_logistic_regression.loglik
)

out_obj_quants = obj_quantities(true_betas, my_data, logistic_template)

@testset "dimensions of obj_quantities are correct" begin
    @inferred Vector{<: Array{Float64,N} where N} obj_quantities(true_betas, my_data, logistic_template)
    @test size(out_obj_quants[1]) == (p, p)
    @test size(out_obj_quants[2]) == (p, p)
    @test size(out_obj_quants[3]) == (p, p)
    @test size(out_obj_quants[4]) == (p,)
end