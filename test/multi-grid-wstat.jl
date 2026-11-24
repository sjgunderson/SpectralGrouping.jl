include(joinpath(@__DIR__, "..", "docs", "examples", "bounded-wstat.jl"))

struct MultiGridTestSource{T} <: SF.AbstractSpectralModel{T,SF.Additive}
    K::T
    slope::T
end

function SF.invoke!(out, edges, model::MultiGridTestSource)
    for i in eachindex(out)
        lo, hi = edges[i], edges[i + 1]
        out[i] = (hi - lo) * (1 + model.slope * (lo + hi) / 2)
    end
    return out
end

multi_grid_model(; norm = 1.4) = MultiGridTestSource(
    SF.FitParam(norm; lower_limit = 0.05, upper_limit = 8.0),
    SF.FitParam(0.18; lower_limit = 0.01, upper_limit = 0.8),
)

function joint_orders(edges, off_counts, scales)
    orders = map(eachindex(off_counts)) do i
        poisson = SF.ErrorStatistics.Poisson
        measured = spectrum(1:2, [12, 26]; exposure = 1.0, statistics = poisson)
        background = spectrum(1:2, off_counts[i]; exposure = 1.0, statistics = poisson,
                              backscale = inv(scales[i]))
        response = [i + (j == row) for row in 1:2, j in 1:(length(edges) - 1)]
        return response_dataset(measured, [1.0, 2], [2.0, 4], edges[1:(end - 1)],
                                edges[2:end], Float64.(response); background)
    end
    return combine_datasets(match_dataset_grids(orders...))
end

@testset "Joint W-stat across independent incident grids" begin
    meg = joint_orders([1.0, 2, 4], ([0, 5], [2, 7]), (0.5, 0.25))
    heg = joint_orders([1.0, 1.5, 2.5, 4], ([6, 0], [9, 1]), (0.2, 0.8))
    third = joint_orders([1.0, 3, 4], ([0, 9], [3, 0], [1, 10]), (0.3, 0.1, 1.0))
    joints = [meg, heg, third]
    @test meg.counts == heg.counts
    @test meg.counts !== heg.counts
    @test length.(getproperty.(joints, :members)) == [2, 2, 3]
    singles = map(joints) do data
        SF.FittingConfig(SF.FittingProblem(multi_grid_model() => data); stat = grid_statistic(data))
    end
    @test SF.statistic_symbol(grid_statistic((meg,)...)) == "W-stat"
    @test_throws ArgumentError grid_statistic(meg, heg, meg)
    bare = match_dataset_grids(dataset(spectrum(1:2, [12, 26])))
    @test_throws ArgumentError grid_statistic(combine_datasets(bare), heg)
    @test_throws ArgumentError grid_statistic(meg, combine_datasets(bare; mode = :sum))
    unsupported = SF.FittingProblem(multi_grid_model() => third)
    @test_throws ArgumentError SF.FittingConfig(unsupported; stat = grid_statistic(meg, heg))
    unknown_counts = copy(meg.counts)
    @test_throws ArgumentError SF.measure(grid_statistic(meg, heg), unknown_counts, ones(4), nothing)

    for n in (2, 3)
        selected = joints[1:n]
        # Reusing a source object still creates two free parameters per model entry.
        source = multi_grid_model()
        problem = SF.FittingProblem((source => data for data in selected)...)
        stat = grid_statistic(reverse(selected)...)
        @test length(SF.FittingConfig(problem; stat).u0) == 2n
        SF.bindall!(problem, :K, :slope)
        cfg = SF.FittingConfig(problem; stat)
        @test length(cfg.u0) == 2
        @test all(cfg.data_cache[i].objective === selected[i].counts for i in 1:n)

        subtotal = (i, p) -> SF.measure_objective!(singles[i], p, 1)
        total = p -> sum(i -> SF.measure_objective!(cfg, p, i), 1:n)
        explicit = p -> sum(i -> subtotal(i, p), 1:n)
        for p in ([0.6, 0.08], [1.4, 0.18], [2.2, 0.32])
            @test isapprox(total(p), explicit(p))
            @test isapprox(ForwardDiff.gradient(total, p), ForwardDiff.gradient(explicit, p))
            for i in 1:n
                @test isapprox(SF.measure_objective!(cfg, p, i), subtotal(i, p))
                predicted = only(SF.calculate_objective!(singles[i], p))
                @test isapprox(SF.measure(stat, predicted, selected[i].counts, nothing), subtotal(i, p))
                @test isapprox(SF.measure(stat, selected[i].counts, predicted, nothing), subtotal(i, p))
            end
        end

        # SF.fit uses model-first measure; the bounded example and finalizer use data first.
        if n == 2
            result = SF.fit(cfg, Optim.NelderMead(); autodiff = Optimization.SciMLBase.NoAD(), maxiters = 250)
            @test sum(result.stats) < total(SF.get_value.(cfg.u0))
            @test isapprox(result.stats, [subtotal(i, result.u) for i in 1:n])
        end
        bounded = fit_bounded_wstat(cfg; maxiters = 300)
        @test sum(bounded.stats) < total(SF.get_value.(cfg.u0))
        @test isapprox(bounded.stats, [subtotal(i, bounded.u) for i in 1:n])
        @test isapprox(sum(bounded.stats), explicit(bounded.u))
        @test occursin("W-stat", repr(MIME"text/plain"(), bounded))
        predicted = map(copy, SF.calculate_objective!(cfg, bounded.u))
        for i in 1:n
            panel = plot_curves(bounded; index = i, kind = :counts, res = 1, rebin = 1)
            slice = plot_curves(bounded[i]; kind = :counts, res = 1)
            @test isapprox(vcat(getproperty.(panel.model, :y)...), predicted[i])
            @test map(c -> c.y, slice.model) == map(c -> c.y, panel.model)
            @test occursin("ΔW", panel.residual_label)
            @test isapprox(sum(c -> sum(abs2, c.y), panel.residuals), bounded.stats[i])
        end
    end

    partial = SF.FittingProblem((multi_grid_model() => data for data in joints)...)
    SF.bind!(partial, (1, :K) => (2, :K) => (3, :K))
    cfg = SF.FittingConfig(partial; stat = grid_statistic(heg, third, meg))
    @test length(cfg.u0) == 4
    total = p -> sum(i -> SF.measure_objective!(cfg, p, i), eachindex(joints))
    explicit = p -> [SF.measure_objective!(singles[i], [p[1], p[i + 1]], 1) for i in eachindex(joints)]
    for p in ([0.75, 0.12, 0.27, 0.38], [2.4, 0.42, 0.15, 0.21])
        @test isapprox(total(p), sum(explicit(p)))
        @test isapprox(ForwardDiff.gradient(total, p), ForwardDiff.gradient(p -> sum(explicit(p)), p))
    end
    bounded = fit_bounded_wstat(cfg; maxiters = 300)
    @test isapprox(bounded.stats, explicit(bounded.u))
    @test sum(bounded.stats) < total(SF.get_value.(cfg.u0))

    edge_model = multi_grid_model(; norm = 0.05)
    edge_cfg = SF.FittingConfig(SF.FittingProblem(edge_model => meg); stat = grid_statistic(meg))
    @test_throws ArgumentError fit_bounded_wstat(edge_cfg)
end
