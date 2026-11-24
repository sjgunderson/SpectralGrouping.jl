import SpectralFitting as SF
import ForwardDiff
import Optimization
using OptimizationOptimJL: Optim

function fit_bounded_wstat(cfg; maxiters = 1_000)
    p0 = SF.get_value.(cfg.u0)
    lower = SF.get_lowerlimit.(cfg.u0)
    upper = SF.get_upperlimit.(cfg.u0)
    isempty(p0) && throw(ArgumentError("the fit needs at least one free parameter"))
    all(isfinite, p0) && all((lower .< p0) .& (p0 .< upper)) ||
        throw(ArgumentError("start every free parameter strictly inside its bounds"))

    scale = map(p -> iszero(p) ? one(p) : abs(p), p0)
    function objective(x, _)
        p = x .* scale
        return sum(i -> SF.measure_objective!(cfg, p, i), eachindex(cfg.data_cache))
    end
    f = Optimization.OptimizationFunction{true}(objective, Optimization.ADTypes.AutoForwardDiff())
    problem = Optimization.OptimizationProblem{true}(
        f, p0 ./ scale, nothing; lb = lower ./ scale, ub = upper ./ scale,
    )
    sol = Optimization.solve(problem, Optim.Fminbox(Optim.LBFGS()); maxiters)
    return SF.finalize_result(cfg, sol.u .* scale, sol)
end
