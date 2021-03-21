module MEstimation # general estimating equations with or without bias-reducting adjustments

using NLsolve
using Optim
using FiniteDiff
using ForwardDiff
using LinearAlgebra
using Distributions
using ReverseDiff
# using InvertedIndices

import Base: show, print
import StatsBase: fit, aic, vcov, coef, coeftable, stderror, CoefTable

export objective_function
export objective_function_template

export estimating_function
export get_estimating_function
export estimating_function_template

export aic
export tic
export vcov
export coef
export coeftable
export fit
export stderror

export slice
# export profile

const JCACHE = Dict{DataType, Any}()
const GCACHE = Dict{DataType, Any}()
# look at objective_functions.jl
# take a test and try to rewrite the test using the new design (look for all the new things I will need to do)
# remove most of the tests, keep only logreg using objectives 
include("estimating_functions.jl")
include("objective_functions.jl")
include("fit.jl")
include("result_methods.jl")
include("slice.jl")
# include("profile.jl")

end # module
