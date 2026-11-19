@testset "Count curves, units, errors, and gap boundaries" begin
    spec = spectrum(1:4, [4, 8, 5, 7]; exposure = 2.0, errors = [1.0, 2, 3, 4])
    data = dataset(spec; low = [1.0, 2, 4, 5])
    counts = only(plot_curves(data; kind = :counts, rebin = 4).data)
    @test counts.grid.low == [1.0, 4]
    @test counts.grid.high == [3.0, 6]
    @test counts.y == [12, 12]
    @test Matrix(counts.covariance) == [5 0; 0 25]
    wave = only(plot_curves(data; kind = :counts, rebin = 4, xaxis = :wavelength).data)
    @test wave.y == reverse(counts.y)
    @test wave.error == reverse(counts.error)
    rates = only(plot_curves(data; rebin = 4).data)
    @test rates.y == [3, 3]
    @test rates.error == counts.error ./ 4
    @test spec.data == [4, 8, 5, 7]

    missing_exposure = dataset(spectrum(1:2, [2, 3]; exposure = 0.0))
    @test only(plot_curves(missing_exposure; kind = :counts).data).y == [2, 3]
    @test_throws ArgumentError plot_curves(missing_exposure)
end

@testset "Matched plotting covariance and exposure normalization" begin
    poisson = SF.ErrorStatistics.Poisson
    first = response_dataset(
        spectrum(0:1, [10, 20]; exposure = 10.0, errors = [1.0, 2.0]),
        [1.0, 2], [2.0, 4], [1.0], [4.0], reshape([1.0, 2], 2, 1),
    )
    second = response_dataset(
        spectrum([0], [12]; exposure = 20.0, statistics = poisson),
        [1.0], [4.0], [1.0], [4.0], reshape([3.0], 1, 1),
    )
    match = match_dataset_grids(first, second)
    @test length(plot_curves(match; kind = :counts).data) == 2
    combined = only(plot_curves(match; kind = :counts, combine = :sum, xaxis = :wavelength).data)
    weights = [1 / 3, 2 / 3]
    expected = [4.0 0; 0 1] + 12 .* (weights * weights')
    @test isapprox(combined.y, [24.0, 18])
    @test isapprox(Matrix(combined.covariance), expected)
    @test combined.fractional
    summed = combine_datasets(match; mode = :sum)
    displayed = only(plot_curves(summed; kind = :counts, xaxis = :wavelength).data)
    @test isapprox(displayed.y, summed.counts)
    @test isapprox(Matrix(displayed.covariance), Matrix(summed.covariance))

    rate = only(plot_curves(match; normalization = :rate, combine = :exposure, xaxis = :wavelength).data)
    @test isapprox(rate.y, combined.y ./ 30)
    @test isapprox(Matrix(rate.covariance), expected ./ 30^2)
    density = only(plot_curves(match; combine = :exposure, xaxis = :wavelength).data)
    @test isapprox(density.y .* (density.grid.high .- density.grid.low), rate.y)
end

@testset "Background display and fitted residuals" begin
    poisson = SF.ErrorStatistics.Poisson
    bkg = spectrum([3, 2, 1], [60, 5, 5]; exposure = 2.0, backscale = 5.0, statistics = poisson)
    on = response_dataset(
        spectrum(1:2, [6, 0]; exposure = 4.0, areascale = 0.5, statistics = poisson),
        [1.0, 2], [2.0, 3], [1.0], [3.0], reshape([1.0, 0], 2, 1);
        background = bkg,
    )
    net = only(plot_curves(on; kind = :counts).data)
    @test net.y == [5, -1]
    @test isapprox(net.error .^ 2, [6.2, 1.2])
    @test only(plot_curves(on; kind = :counts, background = :include).data).y == [6, 0]
    @test only(plot_curves(on; kind = :counts, background = :only).data).y == [1, 1]

    joint = combine_datasets(match_dataset_grids(on))
    source = GridTestSource(SF.FitParam(2.0))
    problem = SF.FittingProblem(source => joint)
    config = SF.FittingConfig(problem; stat = grid_statistic(joint))
    fitted = SF.finalize_result(config, [1.25], nothing)
    grid_model_calls[] = 0
    spectra = plot_curves(fitted; kind = :counts, res = 5, scale = 3, offset = 7)
    @test grid_model_calls[] == 1
    @test only(spectra.model).y == [22, 7]
    @test isapprox(only(spectra.residuals).y, [0.0, -10 * log1p(0.2)])
    @test isnothing(only(spectra.residuals).error)
    @test SF.get_value(source.K) == 2.0
    @test isapprox(SpectralGrouping._pooled_profile(14.0, 4.0, [4.0, 9.0], [0.5, 0.25]),
                   2.649042167394466; atol = 1e-10)
end

@testset "Response-corrected spectra" begin
    hc = SpectralGrouping.HC_KEV_ANGSTROM
    arf = SF.AncillaryResponse([1.0, 2.5], [1.5, 4.0], [6.0, 5.0])
    data = response_dataset(
        spectrum(1:3, [10, 21, 3]; exposure = 2.0, areascale = 0.5),
        [1.0, 2, 4], [2.0, 3, 5], [1.0, 2], [2.0, 4], [1.0 0; 0 0; 0 1];
        ancillary = arf,
    )
    curve = only(plot_curves(data; kind = :unfold, xaxis = :wavelength, rebin = 10).data)
    @test isapprox(curve.weights, [0.75hc, 0, 2hc])
    @test curve.valid == [true, false, true]
    @test isnan(curve.y[2])
    @test isapprox(curve.y[curve.valid], [3 / (0.75hc), 10 / (2hc)])
    @test all(iszero, Matrix(curve.covariance)[2, :])
    energy = only(plot_curves(data; kind = :unfold).data)
    @test isapprox(energy.y[energy.valid] .* (energy.grid.high[energy.valid] .- energy.grid.low[energy.valid]),
                   reverse(curve.y[curve.valid] .* (curve.grid.high[curve.valid] .- curve.grid.low[curve.valid])))

    a = response_dataset(spectrum([1], [10]; exposure = 1.0), [1.0], [2.0],
                         [1.0], [2.0], fill(1.0, 1, 1))
    b = response_dataset(spectrum([1], [18]; exposure = 3.0), [1.0], [2.0],
                         [1.0], [2.0], fill(1.0, 1, 1))
    pooled = only(plot_curves((a, b); kind = :unfold, combine = :sum,
                              xaxis = :wavelength).data)
    @test isapprox(only(pooled.y), 14 / hc)
    fitted = plot_curves(a; kind = :unfold, model = GridTestSource(SF.FitParam(2.0)),
                         model_display = :source)
    @test isapprox(only(only(fitted.model).y), 2.0)
end
