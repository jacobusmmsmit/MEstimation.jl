## Ioannis Kosmdiis
## 4 Feb 2021

using ForwardDiff
using ReverseDiff
using Random
using BenchmarkTools

module logit

using Distributions

struct logistic_data
    y::Vector
    x::Array{Float64}
    m::Vector
end

function nobs(data::logistic_data)
    nx = size(data.x)[1]
    ny = length(data.y)
    nm = length(data.m)
    if (nx != ny)
        error("number of rows in of x is not equal to the length of y")
    elseif (nx != nm)
       error("number of rows in of x is not equal to the length of m")
    elseif (ny != nm)
       error("length of y is not equal to the length of m")
    end
    nx
end

function simulate(theta::Vector,
                  x::Matrix{Float64},
                  m::Vector{Int64})
    n = size(x)[2]
    y = rand.(Binomial.(m, cdf.(Logistic(), x * theta)))
    logistic_data(y, x, m);
end

function loglik1(theta::Vector,
                 data::logistic_data,
                 i::Int64)
    eta = sum(data.x[i, :] .* theta)
    mu = exp.(eta) ./ (1 .+ exp.(eta))
    data.y[i] .* log.(mu) + (data.m[i] - data.y[i]) .* log.(1 .- mu)
end


function observation(data::logistic_data, i::Int64)
    vcat(data.y[i], data.m[i], data.x[i,:])
end

function loglik2(theta::AbstractArray,
                 data::AbstractArray)
    y = data[1]
    m = data[2]
    x = data[3:end]
    eta = sum(x .* theta)
    mu = exp.(eta) ./ (1 .+ exp.(eta))
    y .* log.(mu) + (m - y) .* log.(1 .- mu)
end

end


Random.seed!(123);
n = 5000;
p = 50;
x = rand(n, p);
x[:, 1] = fill(1.0, n);
true_betas = rand(p) * 0.02;
my_data = logit.simulate(true_betas, x, fill(1, n));

## Gradient per observation as required by MEstimation
function gR(theta::Vector{T}, obs::Vector{S}) where {T <: Real,S <: Real}
    x = (theta, obs)
    f = (x0, x1) -> logit.loglik2(x0, x1)
    if !(haskey(CACHE, T))
        tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, x))
        CACHE[T] = (tape, (zeros(T, length(theta)), zeros(T, length(obs))))
    end
    tape, y = CACHE[T]
    return ReverseDiff.gradient!(y, tape, x)[1]
end

function gR2(theta::Vector{T}, obs::Vector{S}) where {T <: Real,S <: Real}
    f = eta -> logit.loglik2(eta, obs)
    if !(haskey(CACHE, T))
        tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, theta))
        CACHE[T] = (tape, zeros(T, length(theta)))
    end
    tape, y = CACHE[T]
    return ReverseDiff.gradient!(y, tape, theta)
end

## Same thing using forward diff
function gF(theta::Vector, data::logit.logistic_data, i::Int64)
    obs = logit.observation(data, i)
    ForwardDiff.gradient(th -> logit.loglik2(th, obs), theta)
end


const CACHE = Dict{DataType,Any}()

function runR(pars, j::Int64) 
    ForwardDiff.jacobian(theta -> gR(theta, logit.observation(my_data, j)), pars)
end

function runR2(pars, j::Int64)
    ForwardDiff.jacobian(theta -> gR2(theta, logit.observation(my_data, j)), pars)
end

function runF(pars, j::Int64) 
    ForwardDiff.jacobian(theta -> gF(theta, my_data, j), pars)
end

function runFhessian(pars, j::Int64) 
    ForwardDiff.hessian(theta -> logit.loglik1(theta, my_data, j), pars)
end


CACHE
runR(true_betas, 1)
runR2(true_betas, 1)
CACHE
runF(true_betas, 1)
runFhessian(true_betas, 1)

## The log-likelihood hessian across all observations
ra = 1:logit.nobs(my_data)
a0 = @benchmark sum(map(j -> runR2(true_betas, j), ra))
a1 = @benchmark sum(map(j -> runR(true_betas, j), ra))
a2 = @benchmark sum(map(j -> runF(true_betas, j), ra))
a3 = @benchmark sum(map(j -> runFhessian(true_betas, j), ra))

## Forward is 15-25 times slower than reverse implementation above
ratio(mean(a2), mean(a1))
ratio(mean(a3), mean(a1))
# julia> ratio(mean(a2), mean(a1))
# BenchmarkTools.TrialRatio: 
#   time:             16.721108277780537
#   gctime:           29.415474243662555
#   memory:           26.19886681971946
#   allocs:           1.0385193063358555

# julia> ratio(mean(a3), mean(a1))
# BenchmarkTools.TrialRatio: 
#   time:             26.0195816865898
#   gctime:           29.945178720705652
#   memory:           32.704893760988654
#   allocs:           3.469006754781931

## Use in optimization
using NLsolve

function fR(theta::Vector)
    gr = zeros(length(theta))
    for j in 1:n
        obs = logit.observation(my_data, j)
        gr += gR(theta, obs)
    end
    gr
end

function fF(theta::Vector)
    gr = zeros(length(theta))
    for j in 1:n
        gr += gF(theta, my_data, j)
    end
    gr
end

a1 = @benchmark nlsolve(fR, true_betas)
a2 = @benchmark nlsolve(fF, true_betas)

## Optimization with forward is 3x slower and reuires 10 times the memory
ratio(mean(a2), mean(a1))
# julia> ratio(mean(a2), mean(a1))
# BenchmarkTools.TrialRatio: 
#   time:             3.0670491962167503
#   gctime:           14.19187050750875
#   memory:           10.935421740548241
#   allocs:           1.1018641415575794
