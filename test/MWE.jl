using Test
using Random
using Distributions
using MEstimation
using NLsolve

include("new_logistic_regression.jl")

Random.seed!(123);
n = 50;
p = 10;
x = rand(n, p);
x[:, 1] = fill(1.0, n);
true_betas = rand(p) * 0.02;
my_data = new_logistic_regression.simulate(true_betas, x, fill(1, n));

logistic_template = MEstimation.objective_function_template(
    new_logistic_regression.nobs,
    new_logistic_regression.observation,
    new_logistic_regression.loglik
)   

logistic_obj_template = MEstimation.objective_function_template(new_logistic_regression.nobs,
                                                    new_logistic_regression.observation,
                                                    new_logistic_regression.loglik)

obj_br_implicit = MEstimation.fit(logistic_obj_template, my_data, true_betas, estimation_method="RBM")

obj_br_explicit = MEstimation.fit(logistic_obj_template, my_data, true_betas, estimation_method="RBM", br_method="explicit_trace")


@test isapprox(coef(obj_br_implicit), coef(obj_br_explicit))