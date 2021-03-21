"""
    estimating_function_template(nobs::Function, 
                                 ef_contribution::Function)

[Composite type](https://docs.julialang.org/en/v1/manual/types/#Composite-Types-1) for defining an `estimating_function_template`.

Arguments
===
+ `nobs`: a function of `data` that computes the number of observations of the particular data type,
+ `observation`: a function of `data` and an indexing value `i` which extracts the `i`th observation from the data
+ `ef_contribution`: a function of the parameters `theta`, the `data` and the observation index `i` that returns a vector of length `length(theta)`.

Result
===
An `estimating_function_template` object with fields `nobs` and `obj_contributions`.
"""
struct estimating_function_template
    nobs::Function
    observation::Function
    ef_contribution::Function
end

""" 
    estimating_function(theta::Vector,
                        data::Any,
                        template::estimating_function_template,
                        br::Bool = false,
                        concentrate::Vector{Int64} = Vector{Int64}())

Evaluate a vector of estimating functions at `theta` by adding up all contributions in `data`, according to an [`estimating_function_template`](@ref).

Arguments
===
+ `theta`: a `Vector` of parameter values at which to evaluate the estimating functions
+ `data`: typically an object of [composite type](https://docs.julialang.org/en/v1/manual/types/#Composite-Types-1) with all the data required to compute the `estimating_function`.
+ `template`: an [`estimating_function_template`](@ref) object.
+ `br`: a `Bool`. If `false` (default), the estimating functions is constructed by adding up all contributions in 
`data`, according to [`estimating_function_template`](@ref), before it is evaluated at `theta`. If `true` then the empirical bias-reducing adjustments in [Kosmidis & Lunardon, 2020](http://arxiv.org/abs/2001.03786) are computed and added to the estimating functions.
+ `concentrate`: a `Vector{Int64}`; if specified, empirical bias-reducing adjustments are added only to the subset of estimating functions indexed by `concentrate`. The default is to add empirical bias-reducing adjustments to all estimating functions.

Result
===
A `Vector`.

Details
===
`data` can be used to pass additional constants other than the actual data to the objective.
"""
function estimating_function(theta::Vector,
                             data::Any,
                             template::estimating_function_template,
                             br::Bool=false,
                             concentrate::Vector{Int64}=Vector{Int64}())
    p = length(theta)
    n_obs = template.nobs(data)
    ef = zeros(p)
    for i in 1:n_obs
        obs_i = template.observation(data, i)
        ef += template.ef_contribution(theta, obs_i)
    end
    if (br)
        quants = ef_quantities(theta, data, template, br, concentrate)
        ef + quants[1]
    else
        ef
    end
end

""" 
    get_estimating_function(data::Any,
                            template::estimating_function_template,
                            br::Bool = false,
                            concentrate::Vector{Int64} = Vector{Int64}(),
                            regularizer::Any = Vector{Int64}())

Construct the estimating functions by adding up all contributions in the `data` according to [`estimating_function_template`](@ref).

Arguments
===
+ `data`: typically an object of [composite type](https://docs.julialang.org/en/v1/manual/types/#Composite-Types-1) with all the data required to compute the `estimating_function`.
+ `template`: an [`estimating_function_template`](@ref) object.
+ `br`: a `Bool`. If `false` (default), the estimating functions is constructed by adding up all contributions in 
`data`, according to [`estimating_function_template`](@ref), before it is evaluated at `theta`. If `true` then the empirical bias-reducing adjustments in [Kosmidis & Lunardon, 2020](http://arxiv.org/abs/2001.03786) are computed and added to the estimating functions.
+ `concentrate`: a `Vector{Int64}`; if specified, empirical bias-reducing adjustments are added only to the subset of estimating functions indexed by `concentrate`. The default is to add empirical bias-reducing adjustments to all estimating functions.
+ `regularizer`: a function of `theta` and `data` returning a `Vector` of dimension equal to the number of the estimating functions, which is added to the (bias-reducing) estimating function; the default value will result in no regularization.

Result
===
An in-place function that stores the value of the estimating functions inferred from `template`, in a preallocated vector passed as its first argument, ready to be used withing `NLsolve.nlsolve`. This is the in-place version of [`estimating_function`](@ref) with the extra `regularizer` argument.
"""
function get_estimating_function(data::Any,
                                 template::estimating_function_template,
                                 br::Bool=false,
                                 concentrate::Vector{Int64}=Vector{Int64}(),
                                 regularizer::Any=Vector{Int64}())
    ## regulizer here has different type and default than fit, because
    ## the dimension of theta cannot be inferred
    has_regularizer = typeof(regularizer) <: Function
    function (F, theta::Vector)
        if has_regularizer
            out = estimating_function(theta, data, template, br, concentrate) + regularizer(theta, data)
        else
            out = estimating_function(theta, data, template, br, concentrate)
        end
        for i in 1:length(out)
            F[i] = out[i]
        end
    end    
end

function ef_quantities(theta::Vector,
                       data::Any,
                       template::estimating_function_template,
                       adjustment::Bool=false,
                       concentrate::Vector{Int64}=Vector{Int64}())
    estfun = (pars, obs) -> template.ef_contribution(pars, obs)
    
    function ja_obs(eta::Vector{T}, obs) where {T <: Any}
        x = (eta, obs)
        f = (params, obs_i) -> estfun(params, obs_i)
        if !(haskey(JCACHE, T))
            jtape = ReverseDiff.compile(ReverseDiff.JacobianTape(f, x))
            JCACHE[T] = (
                jtape,
                (
                    zeros(T, (length(eta), length(eta))),
                    zeros(T, (length(obs), length(eta)))
                )
            )
        end
        jtape, y = JCACHE[T]
        return ReverseDiff.jacobian!(y, jtape, x)[1]
    end

    p = length(theta)
    n_obs = template.nobs(data)
    psi = zeros(p)
    emat = zeros(p, p)
    jmat = zeros(p, p)

    if adjustment
        u = (theta, obs) -> ForwardDiff.jacobian(eta -> ja_obs(eta, obs), theta) 
        psi2 = Vector(undef, p)
        for j in 1:p
            psi2[j] = zeros(p, p)
        end
        umat = zeros(p * p, p)
        for i in 1:n_obs
            obs = template.observation(data, i)
            cpsi = estfun(theta, obs)
            psi += cpsi
            emat += cpsi * cpsi'
            jaco = ja_obs(theta, obs)
            jmat += -jaco
            umat += u(theta, obs)
            for j in 1:p
                psi2[j] += jaco[j, :] * cpsi'
            end
        end
    else
        for i in 1:n_obs
            obs = template.observation(data, i)
            cpsi = estfun(theta, obs)
            psi += cpsi
            emat += cpsi * cpsi'
            jmat += - ja_obs(theta, obs)
        end
    end
      
    jmat_inv = try
        inv(jmat)
    catch
        fill(NaN, p, p)
    end
    vcov = jmat_inv * (emat * jmat_inv') 

    if adjustment
        Afun(j::Int64) = -tr(jmat_inv * psi2[j] + vcov * umat[j:p:(p * p - p + j), :] / 2)
        A = map(Afun, 1:p)
        ## if concentrate then redefine A
        if length(concentrate) > 0
            if any((concentrate .> p) .| (concentrate .< 1))
                error(concentrate, " should be a vector of integers between ", 1, " and ", p)
            else
                ist = concentrate
                nce = deleteat!(collect(1:p), concentrate)
                A_ist = A[ist]
                A_nce = A[nce]
                A = vcat(A_ist + inv(jmat_inv[ist, ist]) * jmat_inv[ist, nce] * A_nce,
                         zeros(length(nce)))
            end
        end
        [A, jmat_inv, emat, psi]
    else
        [vcov, jmat_inv, emat, psi]
    end
end
