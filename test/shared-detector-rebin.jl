function rgs_width_pair(; n = 12, good1 = [1:4; 9:12], good2 = 5:12, zero_last = false)
    wave = collect(range(10.0; step = 0.010, length = n + 1))
    hc = SpectralGrouping.HC_KEV_ANGSTROM
    function detector(good, level, exposure, scale, order)
        counts = fill(level, n)
        zero_last && (counts[end] = 0)
        quality = ones(Int, n)
        quality[good] .= 0
        poisson = SF.ErrorStatistics.Poisson
        source = spectrum(order, counts[order]; quality = quality[order], exposure, statistics = poisson)
        off = spectrum(order, fill(scale, n); exposure, backscale = Float64(scale), statistics = poisson)
        return response_dataset(source, hc ./ wave[order .+ 1], hc ./ wave[order],
                                [1.0], [2.0], ones(n, 1); background = off)
    end
    return detector(good1, 1, 1.0, 2, 1:n), detector(good2, 2, 3.0, 4, n:-1:1), wave
end

@testset "Rebin the shared RGS lattice after native alignment" begin
    rgs1, rgs2, wave = rgs_width_pair()
    native_plan = grouping_plan(rgs1, :width, 4)
    match = match_dataset_grids(rgs1, rgs2)
    joint = combine_datasets(match; groupings = (native_plan, nothing))
    options = (; kind = :counts, xaxis = :wavelength, combine = :sum, background = :include)
    # The former pooled path accepted the native plan and returned widths
    # 0.040, 0.010, 0.010, 0.010, 0.010, 0.040 Å across continuous coverage.
    @test_throws ArgumentError plot_curves(match; options..., groupings = (native_plan, nothing))
    curve = only(plot_curves(joint; options..., rebin = 4).data)
    @test isapprox(curve.grid.low, wave[[1, 5, 9]])
    @test isapprox(curve.grid.high .- curve.grid.low, fill(0.040, 3))
    @test isapprox(curve.y, [4, 8, 12])

    hc = SpectralGrouping.HC_KEV_ANGSTROM
    model = GridTestSource(SF.FitParam(1.0))
    for ref in (1, 2)
        match = match_dataset_grids(rgs1, rgs2; reference = ref)
        @test isapprox(detector_grid(match; coordinate = :wavelength).low, wave[1:end-1])
        full = align_detector_grids(match; model, unfold = true)
        @test full.coordinate == :wavelength
        @test full.channels_per_bin == ones(Int, 12)
        @test full.members[1].counts == [ones(4); zeros(4); ones(4)]
        @test full.members[2].counts == [zeros(4); fill(2, 8)]

        shared = rebin_detector_grid(full, 4)
        @test shared.channels_per_bin == [4, 4, 4]
        @test isapprox(shared.members[1].coverage, [1, 0, 1])
        @test isapprox(shared.members[2].coverage, [0, 1, 1])
        @test shared.members[1].counts == [4, 0, 4]
        @test shared.members[2].counts == [0, 8, 8]
        @test shared.members[1].background.alpha .* shared.members[1].background.counts == [4, 0, 4]
        @test shared.members[2].background.alpha .* shared.members[2].background.counts == [0, 4, 4]
        @test shared.members[2].model == [0, 12, 12]
        @test isapprox(shared.members[1].weights, (hc / 2) .* [4, 0, 4])
        @test isapprox(shared.members[2].weights, (hc / 2) .* [0, 12, 12])
        @test_throws ArgumentError rebin_detector_grid(full, 0)

        joint = combine_datasets(match)
        cfg = SF.FittingConfig(SF.FittingProblem(model => joint); stat = grid_statistic(joint))
        score = SF.measure_objective!(cfg, [1.0], 1)
        result = SF.finalize_result(cfg, [1.0], nothing)
        native = [ones(8); fill(3, 8)]
        @test predict_counts(joint, [1.0]) == native
        for input in ((rgs1, rgs2), match, joint, result)
            plotted = plot_curves(input; options..., model = input === result ? nothing : model, rebin = 4)
            curve = only(plotted.data)
            @test isapprox(curve.grid.low, wave[[1, 5, 9]])
            @test isapprox(curve.grid.high .- curve.grid.low, fill(0.040, 3))
            @test isapprox(curve.y, [4, 8, 12])
            @test isapprox(only(plotted.model).y, [8, 16, 24])
            unfold = only(plot_curves(input; kind = :unfold, combine = :sum, xaxis = :wavelength,
                                      background = :include, rebin = 4).data)
            @test isapprox(unfold.weights, (hc / 2) .* [4, 12, 16])
            @test isapprox(unfold.y, [4, 8, 12] ./ unfold.weights)
        end
        @test align_detector_grids(result).members[2].model == full.members[2].model
        @test SF.measure_objective!(cfg, [1.0], 1) == score == only(result.stats)
        @test predict_counts(joint, [1.0]) == native
    end
end

@testset "Grouping phase, partial coverage, and short geometric segments" begin
    a, b, wave = rgs_width_pair(; n = 13, good1 = [1, 3, 12, 13], good2 = [4, 9, 10, 11, 12], zero_last = true)
    match = match_dataset_grids(a, b; detector_grid = wave, detector_coordinate = :wavelength)
    @test detector_grid(match).low == wave[1:end-1]
    @test match.model_edges == [1, 2]
    shared = rebin_detector_grid(align_detector_grids(match; unfold = true), 4)
    @test shared.grid.low == wave[[1, 5, 9, 13]]
    @test shared.grid.high == wave[[5, 9, 13, 14]]
    @test shared.channels_per_bin == [4, 4, 4, 1]
    @test isapprox(shared.members[1].coverage, [0.5, 0, 0.25, 1])
    @test isapprox(shared.members[2].coverage, [0.25, 0, 1, 0])
    options = (; combine = :sum, xaxis = :wavelength, rebin = 4, background = :include)
    counts = only(plot_curves(match; options..., kind = :counts).data)
    @test counts.valid == [true, false, true, true]
    @test isapprox(counts.y[counts.valid], [4, 9, 0])
    @test isnan(counts.y[2])
    unfolded = only(plot_curves(match; options..., kind = :unfold).data)
    @test unfolded.valid == counts.valid
    @test isapprox(unfolded.y[counts.valid], counts.y[counts.valid] ./ unfolded.weights[counts.valid])
    @test unfolded.y[4] == 0
    @test unfolded.weights[2] == 0

    grid = BinGrid(wave[[1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13]],
                   wave[[2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14]])
    gapped = match_dataset_grids(a, b; detector_grid = grid, detector_coordinate = :wavelength)
    grouped = rebin_detector_grid(align_detector_grids(gapped), 4)
    @test grouped.channels_per_bin == [3, 4, 4]
    @test grouped.grid.low == wave[[1, 6, 10]]
    @test grouped.grid.high == wave[[4, 10, 14]]
    @test isapprox(sum(grouped.members[1].counts), 3)
    @test isapprox(sum(grouped.members[2].counts), 8)
end

@testset "Pooled plot defaults to the common fixed fit width" begin
    a, b, wave = rgs_width_pair(; n = 60, good1 = [1:20; 41:60], good2 = 21:60)
    matched = match_dataset_grids(a, b)
    plans = [grouping_plan(d, :width, 20) for d in (a, b)]
    joint = combine_datasets(matched; groupings = plans)
    model = GridTestSource(SF.FitParam(1.0))
    cfg = SF.FittingConfig(SF.FittingProblem(model => joint); stat = grid_statistic(joint))
    fit = SF.finalize_result(cfg, [1.0], nothing)
    opts = (; kind = :counts, combine = :sum, xaxis = :wavelength, background = :include)

    for input in (joint, fit)
        curve = only(plot_curves(input; opts...).data)
        @test isapprox(curve.grid.low, wave[[1, 21, 41]])
        @test isapprox(curve.y, [20, 40, 60])
        @test isapprox(curve.error, sqrt.([20, 40, 60]))
        @test length(only(plot_curves(input; opts..., rebin = 10).data).grid) == 6
        @test length(only(plot_curves(input; opts..., rebin = 1).data).grid) == 60
        @test length(align_detector_grids(input).grid) == 60
    end
    native = plot_curves(fit; kind = :counts, combine = :none, xaxis = :wavelength)
    @test [length(curve.grid) for curve in native.data] == [2, 2]
    @test joint.counts == [20, 20, 40, 40]

    mixed = combine_datasets(matched; groupings = (plans[1], grouping_plan(b, :width, 10)))
    @test length(only(plot_curves(mixed; opts...).data).grid) == 60
    count_grouped = combine_datasets(matched; groupings = [grouping_plan(d, :counts, 10) for d in (a, b)])
    @test length(only(plot_curves(count_grouped; opts...).data).grid) == 60
end

@testset "Fractional overlap, shared covariance, and zero energy bounds" begin
    a = dataset(spectrum(1:4, [8, 900, 12, 900]; quality = [0, 1, 0, 1],
                         statistics = SF.ErrorStatistics.Poisson))
    b = dataset(spectrum(1:4, [900, 10, 900, 20]; statistics = SF.ErrorStatistics.Poisson);
                low = collect(1.5:4.5), mask = BitVector([0, 1, 0, 1]))
    opts = (; kind = :counts, combine = :sum, matching_coordinate = :energy, rebin = 2)
    fit_a = only(plot_curves(match_dataset_grids(a, b); opts...).data)
    fit_b = only(plot_curves(match_dataset_grids(a, b; reference = 2); opts...).data)
    @test fit_a.y == [13, 27]
    @test fit_b.y == [20, 26]
    @test Matrix(fit_a.covariance) == [10.5 2.5; 2.5 19.5]
    @test Matrix(fit_b.covariance) == [15 3; 3 23]

    nonuniform = match_dataset_grids(a, b; detector_grid = [1.0, 2, 2.5, 4, 5])
    aligned = rebin_detector_grid(align_detector_grids(nonuniform; matching_coordinate = :energy), 2)
    @test aligned.channels_per_bin == [2, 2]
    @test aligned.grid.low == [1, 2.5]
    @test aligned.grid.high == [2.5, 5]
    @test only(plot_curves(nonuniform; opts...).data).y == [8, 32]

    function zero_edge(counts)
        spec = spectrum(1:3, counts; quality = [1, 0, 0], exposure = 1.0)
        return response_dataset(spec, [0.0, 1, 2], [1.0, 2, 3], [1.0], [2.0], ones(3, 1))
    end
    data = (zero_edge([900, 4, 6]), zero_edge([900, 5, 7]))
    match = match_dataset_grids(data)
    @test align_detector_grids(match).coordinate == :energy
    counts = only(plot_curves(match; kind = :counts, combine = :sum).data)
    @test isequal(counts.y, [NaN, 9, 13])
    flux = only(plot_curves(match; kind = :unfold, combine = :sum).data)
    @test flux.valid == [false, true, true]
    @test isapprox(flux.y[flux.valid], [9 / 2, 13 / 6])
    rebinned = only(plot_curves(match; kind = :unfold, combine = :sum, rebin = 2).data)
    @test rebinned.grid.low == [0, 2]
    @test rebinned.valid == [false, true]
    @test_throws ArgumentError plot_curves(match; kind = :counts, combine = :sum, xaxis = :wavelength)
    @test_throws ArgumentError align_detector_grids(match; matching_coordinate = :wavelength)
end
