module new_logistic_regression

using Distributions

struct data
    y::Vector
    x::Array{Float64}
    m::Vector
end

## Logistic regression nobs
function nobs(data::data)
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
    return nx
end

function observation(data::data, i::Int64)
    vcat(data.y[i], data.m[i], data.x[i,:])
end

function loglik(theta::AbstractArray,
                obs::AbstractArray)
    y = obs[1]
    m = obs[2]
    x = obs[3:end]
    eta = sum(x .* theta)
    mu = exp.(eta) ./ (1 .+ exp.(eta))
    return y .* log.(mu) + (m - y) .* log.(1 .- mu)
end

function logistic_ef(theta::AbstractArray,
                     obs::AbstractArray)
    y = obs[1]
    m = obs[2]
    x = obs[3:end]
    eta = sum(x .* theta)
    mu = exp.(eta) ./ (1 .+ exp.(eta))
    return x * (y - m * mu)
end

function simulate(theta::Vector,
                  x::Matrix{Float64},
                  m::Vector{Int64})
    n = size(x)[2]
    y = rand.(Binomial.(m, cdf.(Logistic(), x * theta)))
    data(y, x, m);
end


end