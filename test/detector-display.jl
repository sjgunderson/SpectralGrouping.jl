function complementary_detectors(; notice = false, reverse_second = false)
    function detector(counts, good, off, response, exposure, backscale, order)
        edges = collect(1.0:8.0)
        poisson = SF.ErrorStatistics.Poisson
        quality = notice ? zeros(Int, 7) : Int.(.!good[order])
        source = spectrum(order, counts[order]; quality, exposure, statistics = poisson)
        background = spectrum(order, off[order]; exposure, backscale, statistics = poisson)
        mask = notice ? BitVector(good[order]) : trues(7)
        return response_dataset(source, edges[order], edges[order .+ 1], [1.0], [8.0],
                                reshape(response[order], 7, 1); mask, background)
    end
    a = detector([10, 20, 900, 900, 900, 0, 7], Bool[1, 1, 0, 0, 0, 1, 1],
                 [2, 4, 900, 900, 900, 0, 2], [1.0, 2, 0, 0, 0, 1, 1], 1.0, 2.0, 1:7)
    b = detector([900, 900, 30, 40, 900, 900, 9], Bool[0, 0, 1, 1, 0, 0, 1],
                 [900, 900, 3, 6, 900, 900, 3], [0.0, 0, 3, 4, 0, 0, 2], 2.0, 3.0,
                 reverse_second ? (7:-1:1) : (1:7))
    return a, b
end

@testset "Complementary detector coverage and native likelihood" begin
    opts = (; kind = :counts, combine = :sum, matching_coordinate = :energy)
    valid = BitVector([1, 1, 1, 1, 0, 1, 1])
    model = GridTestSource(SF.FitParam(2.0))
    for notice in (false, true), ref in (1, 2)
        a, b = complementary_detectors(; notice, reverse_second = ref == 2)
        match = match_dataset_grids(a, b; reference = ref)
        manual = GridMatch(match.datasets, match.model_edges, match.native_grids, match.projections, ref)
        @test manual.detector_grids[ref].low == 1:7
        @test match.detector_grids[ref].low == 1:7
        joint = combine_datasets(match)
        @test joint.reference == ref
        @test joint.counts == [10, 20, 0, 7, 30, 40, 9]
        @test joint.member_ranges == [1:4, 5:7]
        stat = grid_statistic(joint)
        cfg = SF.FittingConfig(SF.FittingProblem(model => joint); stat)
        native_score = SF.measure_objective!(cfg, [2.0], 1)
        native_prediction = [14, 28, 14, 14, 84, 112, 56]
        @test predict_counts(joint, [14.0]) == native_prediction
        expected = sum(SpectralGrouping._profile_deviance.(
            joint.counts, native_prediction, [2, 4, 0, 2, 3, 6, 3], [fill(0.5, 4); fill(1 / 3, 3)]))
        @test isapprox(native_score, expected)
        fit = SF.finalize_result(cfg, [2.0], nothing)

        for input in ((a, b), [a, b], match, joint, fit)
            plotted = plot_curves(input; opts..., model = input === fit ? nothing : model)
            curve = only(plotted.data)
            @test curve.grid.low == 1:7
            @test curve.valid == valid
            @test curve.y[valid] == [9, 18, 29, 38, 0, 14]
            @test isnan(curve.y[5])
            @test only(plotted.model).y[valid] == [14, 28, 84, 112, 14, 70]
            @test all(iszero, curve.covariance[5, :])
            on = plot_curves(input; opts..., model = input === fit ? nothing : model, background = :include)
            @test only(on.data).y[valid] == [10, 20, 30, 40, 0, 16]
            @test only(on.model).y[valid] == [15, 30, 85, 114, 14, 72]
            @test only(plot_curves(input; opts..., background = :only).data).y[valid] == [1, 2, 1, 2, 0, 2]
        end
        @test SF.measure_objective!(cfg, [2.0], 1) == native_score == only(fit.stats)
        @test map(c -> c.grid.low, plot_curves(joint; kind = :counts).data) == [[1, 2, 6, 7], [3, 4, 7]]
        @test only(plot_curves(a; kind = :counts).data).grid.low == [1, 2, 6, 7]

        unfolded = only(plot_curves(fit; kind = :unfold, combine = :sum, matching_coordinate = :energy).data)
        wavelength_width = 7 * SpectralGrouping.HC_KEV_ANGSTROM / 8
        @test unfolded.valid == valid
        @test isnan(unfolded.y[5])
        @test isapprox(unfolded.weights, wavelength_width .* [1, 2, 6, 8, 0, 1, 5])
        @test isapprox(unfolded.y[7], (14 / unfolded.weights[7]) * SpectralGrouping.HC_KEV_ANGSTROM / (7 * 8))
    end

    a, b = complementary_detectors()
    match = match_dataset_grids(a, b; grid = 1.0:0.5:8.0)
    plans = (grouping_plan(a, :width, 2), nothing)
    joint = combine_datasets(match; groupings = plans)
    @test joint.model_edges == 1.0:0.5:8.0
    @test joint.detector_grids[1].low == 1:7
    @test joint.counts == [30, 7, 30, 40, 9]
    @test_throws ArgumentError plot_curves(match; opts..., groupings = plans)
    individual = plot_curves(match; kind = :counts, groupings = plans, background = :include)
    @test first(individual.data).grid.low == [1, 6]
    @test first(individual.data).y == [30, 7]
    curve = only(plot_curves(joint; opts..., rebin = 2, background = :include).data)
    @test curve.grid.low == [1, 3, 5, 7]
    @test curve.grid.high == [3, 5, 7, 8]
    @test curve.y == [30, 70, 0, 16]
    rebinned = only(plot_curves(joint; opts..., rebin = 10, background = :include).data)
    @test rebinned.grid.low == [1]
    @test rebinned.grid.high == [8]
    @test rebinned.y == [116]

    a.data_mask .= false
    empty_ref = only(plot_curves((a, b); opts..., background = :include).data)
    @test empty_ref.grid.low == 1:7
    @test isequal(empty_ref.y, [NaN, NaN, 30, 40, NaN, NaN, 9])
end

@testset "Offset references, covariance and missing EBOUNDS" begin
    poisson = SF.ErrorStatistics.Poisson
    a = dataset(spectrum(1:4, [8, 900, 12, 900]; statistics = poisson,
                         quality = [0, 1, 0, 1]))
    b = dataset(spectrum(1:4, [900, 10, 900, 20]; statistics = poisson);
                low = collect(1.5:4.5), mask = BitVector([0, 1, 0, 1]))
    opts = (; kind = :counts, combine = :sum, matching_coordinate = :energy)
    match = match_dataset_grids(a, b; reference = 2)
    joint = combine_datasets(match)
    cfg = SF.FittingConfig(SF.FittingProblem(GridTestSource(SF.FitParam(1.0)) => joint);
                           stat = grid_statistic(joint))
    fit = SF.finalize_result(cfg, [1.0], nothing)
    for input in (match, joint, fit)
        saved = only(plot_curves(input; opts...).data)
        @test saved.grid.low == 1.5:4.5
        @test saved.y == [4, 16, 6, 20]
        @test sum(saved.y) == 46
        @test Matrix(saved.covariance) == [2 0 0 0; 0 13 3 0; 0 3 3 0; 0 0 0 20]
        override = only(plot_curves(input; opts..., reference = 1).data)
        @test override.grid.low == 1:4
        @test override.y == [8, 5, 17, 10]
        @test sum(override.y) == 40
        @test Matrix(override.covariance) == [8 0 0 0; 0 2.5 2.5 0; 0 2.5 14.5 0; 0 0 0 5]
    end

    gap = dataset(spectrum([1, 3], [4, 6]); low = [1.0, 3])
    full = dataset(spectrum(1:3, [2, 50, 7]))
    curves = only(plot_curves((gap, full); opts..., reference = 1, rebin = 8).data)
    @test curves.grid.low == [1, 3]
    @test curves.grid.high == [2, 4]
    @test curves.y == [6, 13]
    @test only(plot_curves((gap, full); opts..., reference = 2).data).y == [6, 50, 13]
    supplied = match_dataset_grids(gap, full; detector_grid = 1.0:4.0)
    @test detector_grid(supplied).low == [1, 2, 3]
    @test detector_grid(supplied; reference = 1).low == [1, 3]
    @test only(plot_curves(supplied; opts..., rebin = 2).data).y == [56, 13]
    @test detector_grid(combine_datasets(supplied)).low == [1, 2, 3]
    @test combine_datasets(supplied; mode = :sum, coordinate = :energy).grid.low == [1, 3]

    # The RMF can retain an EBOUNDS channel that is absent from the source PHA.
    partial = SF.SpectralData{Float64,Nothing,Nothing}(
        gap.spectrum, full.response, nothing, nothing, gap.energy_low, gap.energy_high,
        gap.domain, gap.data_mask, nothing,
    )
    @test only(plot_curves((partial, full); opts...).data).y == [6, 50, 13]

    roundoff = dataset(spectrum(1:2, [1, 2]); low = [1.0, prevfloat(2.0)], high = [2.0, 3.0])
    @test only(match_dataset_grids(roundoff).detector_grids).low == [1, 2]
    overlap = dataset(spectrum(1:2, [1, 2]); low = [1.0, 1.9], high = [2.0, 3.0])
    @test_throws ArgumentError match_dataset_grids(overlap)
end
