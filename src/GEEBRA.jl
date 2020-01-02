module GEEBRA

using NLsolve
using Optim
using ForwardDiff
using LinearAlgebra

export objective_function, estimating_function, get_estimating_function, solve_estimating_equation, geebra_ef_template, geebra_obj_template, optimize_objective, optimize_objective2

include("estimating_functions.jl")
include("objective_functions.jl")

end # module






# function optimize_objective2(theta::Vector,
#                              data::Any,
#                              template::geebra_obj_template,
#                              br::Bool = false;
#                              method = LBFGS(),
#                              optim_options = Optim.Options())
#     ders = obj_derivatives(data, template)
#     quants = b -> obj_quantities2(b, data, template, ders, br)
#     obj = b -> -objective_function(b, data, template, false)
#     optimize(be -> obj(be) + quants(be)[2], theta, method, optim_options)
# end

# function obj_quantities2(theta::Vector,
#                          data::Any,
#                          template::geebra_obj_template,
#                          derivatives::Any,
#                          penalty::Bool = false)
#     npsi = derivatives[1]
#     nj = derivatives[2]
#     p = length(theta)
#     n_obs = template.nobs(data)
#     psi = Matrix(undef, n_obs, p)
#     njmats = Vector(undef, n_obs)
#     for i in 1:n_obs
#         psi[i, :] =  npsi(theta, i)
#         njmats[i] = nj(theta, i)
#     end
#     jmat_inv = inv(-sum(njmats))
#     emat = psi' * psi
#     emat = convert(Array{Float64, 2}, emat)
#     vcov = jmat_inv * (emat * jmat_inv)
#     if (penalty)        
#         penalty = - tr(jmat_inv * emat) / 2
#         [vcov, penalty]
#     else
#         [vcov]
#     end
# end

# function obj_derivatives(data::Any,
#                          template::geebra_obj_template)
#     result
#     npsi(eta::Vector, i::Int) = ForwardDiff.gradient(beta -> template.obj_contribution(beta, data, i), eta)
#     nj(eta::Vector, i::Int) = ForwardDiff.hessian(beta -> template.obj_contribution(beta, data, i), eta)
#     [npsi, nj]
# end                        
