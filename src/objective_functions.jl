struct objective_template
    nobs::Function
    obj_contribution::Function
end

## Objective
function objective_function(theta::Vector,
                            data::Any,
                            template::objective_template,
                            br::Bool = false)
    p = length(theta)
    n_obs = template.nobs(data)
    contributions = Vector(undef, n_obs)
    for i in 1:n_obs
        contributions[i] = template.obj_contribution(theta, data, i)
    end
    if (br)
        quants = obj_quantities(theta, data, template, br)
        sum(contributions) + quants[2]
    else
        sum(contributions)
    end
end

## optimize objective
function fit(theta::Vector,
             data::Any,
             template::objective_template,
             br::Bool = false;
             method = LBFGS(),
             optim_options = Optim.Options())
    obj = beta -> -objective_function(beta, data, template, br)
    out = optimize(obj, theta, method, optim_options)
    geebra_results(out, out.minimizer, data, template, br, true)
end

function obj_quantities(theta::Vector,
                        data::Any,
                        template::objective_template,
                        penalty::Bool = false)
    npsi(eta::Vector, i::Int) = ForwardDiff.gradient(beta -> template.obj_contribution(beta, data, i), eta)
    nj(eta::Vector, i::Int) = ForwardDiff.hessian(beta -> template.obj_contribution(beta, data, i), eta)
    p = length(theta)
    n_obs = template.nobs(data)
    psi = Matrix(undef, n_obs, p)
    njmats = Vector(undef, n_obs)
    for i in 1:n_obs
        psi[i, :] =  npsi(theta, i)
        njmats[i] = nj(theta, i)
    end
    jmat_inv = inv(-sum(njmats))
    emat = psi' * psi
    emat = convert(Array{Float64, 2}, emat)
    vcov = jmat_inv * (emat * jmat_inv)
    if (penalty)        
        penalty = - tr(jmat_inv * emat) / 2
        [vcov, penalty]
    else
        [vcov]
    end
end