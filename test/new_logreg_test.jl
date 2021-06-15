using Test
using Random
using Distributions
using MEstimation



# out_obj_quants = MEstimation.obj_quantities(true_betas, my_data, logistic_template)

# @testset "dimensions of obj_quantities are correct" begin
#     @inferred Vector{<: Array{Float64,N} where N} MEstimation.obj_quantities(true_betas, my_data, logistic_template)
#     @test size(out_obj_quants[1]) == (p, p)
#     @test size(out_obj_quants[2]) == (p, p)
#     @test size(out_obj_quants[3]) == (p, p)
#     @test size(out_obj_quants[4]) == (p,)
# end

@testset "agreement between obj and ef implementations" begin
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
    logistic_ef_template = MEstimation.estimating_function_template(new_logistic_regression.nobs,
                                                        new_logistic_regression.observation,
                                                        new_logistic_regression.logistic_ef)

    ## Get ef template directly from obj template
    logistic_ef2_template = MEstimation.estimating_function_template(logistic_obj_template)
    
    o1_ml = MEstimation.fit(logistic_obj_template, my_data, true_betas, estimation_method="M")
    e1_ml = MEstimation.fit(logistic_ef_template, my_data, true_betas, estimation_method="M")
    e2_ml = MEstimation.fit(logistic_ef2_template, my_data, true_betas, estimation_method="M")

    o1_br = MEstimation.fit(logistic_obj_template, my_data, true_betas, estimation_method="RBM") # broken
    e1_br = MEstimation.fit(logistic_ef_template, my_data, true_betas, estimation_method="RBM")
    e2_br = MEstimation.fit(logistic_ef2_template, my_data, true_betas, estimation_method="RBM")

    o1_br1 = MEstimation.fit(logistic_obj_template, my_data, true_betas, estimation_method="RBM", br_method="explicit_trace")
    e1_br1 = MEstimation.fit(logistic_ef_template, my_data, true_betas, estimation_method="RBM", br_method="explicit_trace")
    e2_br1 = MEstimation.fit(logistic_ef2_template, my_data, true_betas, estimation_method="RBM", br_method="explicit_trace")

    o1_br2 = MEstimation.fit(logistic_obj_template, my_data, coef(o1_ml), estimation_method="RBM", br_method="explicit_trace")
    e1_br2 = MEstimation.fit(logistic_ef_template, my_data, coef(o1_ml), estimation_method="RBM", br_method="explicit_trace")
    e2_br2 = MEstimation.fit(logistic_ef2_template, my_data, coef(o1_ml), estimation_method="RBM", br_method="explicit_trace")

    
    @test isapprox(coef(o1_ml), coef(e1_ml), atol=1e-05)
    @test isapprox(coef(o1_ml), coef(e2_ml), atol=1e-05)
    @test isapprox(coef(o1_br), coef(e1_br), atol=1e-05) # Fails
    @test isapprox(coef(o1_br), coef(e2_br), atol=1e-05) # Fails
    @test isapprox(coef(o1_br1), coef(e1_br1), atol=1e-05)
    @test isapprox(coef(o1_br1), coef(e2_br1), atol=1e-05)
    @test isapprox(coef(o1_br2), coef(e1_br2), atol=1e-05)
    @test isapprox(coef(o1_br2), coef(e2_br2), atol=1e-05)
    @test isapprox(coef(o1_br1), coef(e1_br2), atol=1e-05)

    @test isapprox(aic(o1_ml),
                   -2 * (objective_function(coef(o1_ml), my_data, logistic_obj_template, false) - p))

    @test isapprox(aic(o1_br),
                   -2 * (objective_function(coef(o1_br), my_data, logistic_obj_template) - p))

    quants_ml = MEstimation.obj_quantities(coef(o1_ml), my_data, logistic_obj_template, true)
    quants_br = MEstimation.obj_quantities(coef(o1_br), my_data, logistic_obj_template, true)

    @test isapprox(tic(o1_ml),
                   -2 * (objective_function(coef(o1_ml), my_data, logistic_obj_template) + 2 * quants_ml[1]))
    
    @test isapprox(tic(o1_br),
                   -2 * (objective_function(coef(o1_br), my_data, logistic_obj_template) + 2 * quants_br[1]))
    
    @test isapprox(vcov(o1_ml), vcov(e1_ml))
    @test isapprox(vcov(o1_br), vcov(e1_br))  # Fails
    @test isapprox(vcov(o1_br), vcov(e2_br)) # Fails
    @test isapprox(vcov(e1_br), vcov(e2_br)) # e1_br and e2_br agree, but o1_br disagrees
    @test isapprox(vcov(o1_br1), vcov(e1_br1))
    @test isapprox(vcov(o1_br2), vcov(e1_br2))
    @test isapprox(vcov(o1_br2), vcov(e2_br2))

    @test isapprox(coeftable(o1_ml).cols, coeftable(e1_ml).cols)
    @test isapprox(coeftable(o1_br).cols, coeftable(e1_br).cols) # Fails
    @test isapprox(coeftable(o1_br1).cols, coeftable(e1_br1).cols)
    @test isapprox(coeftable(o1_br2).cols, coeftable(e1_br2).cols) 
    
end