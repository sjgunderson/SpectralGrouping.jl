function response_dataset(
    spec, low, high, incident_low, incident_high, response;
    mask = trues(length(spec.channels)), background = nothing, ancillary = nothing,
)
    rmf = SF.ResponseMatrix(sparse(response), copy(spec.channels), low, high, incident_low, incident_high)
    domain = vcat(incident_low, last(incident_high))
    return SF.SpectralData{Float64,Nothing,Nothing}(
        spec, rmf, background, ancillary, copy(low), copy(high), domain, mask, nothing,
    )
end

@testset "Restore descending detector bounds before masking" begin
    spec = spectrum(11:14, [50, 20, 7, 5]; exposure = 1.0)
    rmf = SF.ResponseMatrix(
        sparse(reshape([13.0, 11, 14, 12, 99], 5, 1)), [13, 11, 14, 12, 99],
        [1.0, 3, 0.5, 2, 5], [1.5, 4, 1, 3, 6], [0.5], [4.0],
    )
    low, high = [3.0, 2, 1, 0.5], [2.0, 1, 0.5, 1]
    data = SF.SpectralData{Float64,Nothing,Nothing}(
        spec, rmf, nothing, nothing, copy(low), high, [0.5, 4.0],
        BitVector([true, true, false, true]), nothing,
    )
    incorrect = deepcopy(data)
    SF.mask_energies!(incorrect, 0.75, 3.5)
    @test incorrect.data_mask == [true, true, false, false]

    @test restore_detector_bounds!(data) === data
    @test data.energy_low == low
    @test data.energy_high == [4.0, 3, 1.5, 1]
    @test data.data_mask == [true, true, false, true]
    @test data.domain == [0.5, 4.0]
    @test rmf.channel_bins_high == [1.5, 4, 1, 3, 6]
    @test rmf.channels == [13, 11, 14, 12, 99]
    SF.mask_energies!(data, 0.75, 3.5)
    @test data.data_mask == [false, true, false, false]
    joint = combine_datasets(match_dataset_grids(data))
    @test joint.counts == [20]
    @test predict_counts(joint, [1.0]) == [12]

    missing = deepcopy(incorrect)
    missing.response.channels[3] = 100
    @test_throws ArgumentError restore_detector_bounds!(missing)
    @test missing.energy_high == high
end

const grid_model_calls = Ref(0)
struct GridTestSource{T} <: SF.AbstractSpectralModel{T,SF.Additive}
    K::T
end

function SF.invoke!(out, edges, ::GridTestSource)
    grid_model_calls[] += 1
    return out .= diff(edges)
end

@testset "Shared incident grid and native joint fit" begin
    arf = SF.AncillaryResponse([1.0, 2], [2.0, 3], [2.0, 3])
    first = response_dataset(
        spectrum(0:2, [11, 100, 17]; exposure = 2.0),
        [1.0, 2, 3], [2.0, 3, 4], [1.0, 2], [2.0, 3], [1.0 0; 0 0; 0.25 1];
        mask = BitVector([true, false, true]), ancillary = arf,
    )
    second = response_dataset(
        spectrum(10:11, [4, 5]; exposure = 3.0, areascale = 0.5),
        [1.0, 2], [2.0, 3], [1.0, 1.5, 2.5], [1.5, 2.5, 3], [1.0 0 0; 0 1 1],
    )
    match = match_dataset_grids(first, second; reference = 2)
    @test match.model_edges == [1, 1.5, 2, 2.5, 3]
    @test match.projections == [[1:2, 3:4], [1:1, 2:3, 4:4]]
    @test_throws ArgumentError match_dataset_grids(first, first)
    @test_throws ArgumentError match_dataset_grids(first, second; grid = [1.0, 2, 3])
    @test match_dataset_grids(first, second; grid = 1.0:0.25:3.0).model_edges == collect(1.0:0.25:3.0)
    joint = combine_datasets(match)
    @test joint.member_ranges == [1:2, 3:4]
    @test joint.counts == [11, 17, 4, 5]
    @test predict_counts(joint, [1.0, 2, 4, 8]) == [12, 75, 1.5, 21]
    @test first.spectrum.data == [11, 100, 17]
    @test first.data_mask == [true, false, true]
    @test nonzeros(first.response.matrix) == [1, 0.25, 1]

    model = GridTestSource(SF.FitParam(2.0))
    problem = SF.FittingProblem(model => joint)
    config = SF.FittingConfig(problem; stat = grid_statistic(joint))
    grid_model_calls[] = 0
    predicted = only(SF.calculate_objective!(config, [2.0]))
    @test grid_model_calls[] == 1
    @test predicted == predict_counts(joint, ones(4))
    derivative = SF.ForwardDiff.derivative(2.0) do norm
        sum(only(SF.calculate_objective!(config, [norm])))
    end
    @test isapprox(derivative, sum(predicted) / 2)

    gapped = response_dataset(
        spectrum([0], [12]; exposure = 1.0),
        [1.0], [4.0], [1.0, 3], [2.0, 4], [1.0 2.0],
    )
    gap_match = match_dataset_grids(gapped)
    @test gap_match.model_edges == [1, 2, 3, 4]
    @test predict_counts(combine_datasets(gap_match), [2.0, 100, 3]) == [8]
    gapped.response.bins_high[1] = 2.5
    @test_throws ArgumentError combine_datasets(gap_match)
end

@testset "Wavelength sums and full covariance" begin
    poisson = SF.ErrorStatistics.Poisson
    ref = response_dataset(
        spectrum(0:1, [10, 20]; exposure = 1.0, statistics = poisson),
        [1.0, 2], [2.0, 4], [1.0], [4.0], reshape([1.0, 2], 2, 1),
    )
    other = response_dataset(
        spectrum([0], [12]; exposure = 2.0, statistics = poisson),
        [1.0], [4.0], [1.0], [4.0], reshape([3.0], 1, 1),
    )
    match = match_dataset_grids(ref, other)
    combined = combine_datasets(match; mode = :sum)
    @test combined.coordinate == :wavelength
    @test isapprox(combined.grid.low, SpectralGrouping.HC_KEV_ANGSTROM ./ [4, 2])
    @test isapprox(combined.counts, [24, 18])
    @test isapprox(predict_counts(combined, [2.0]), [8, 10])
    @test combined.fractional
    expected_cov = [20.0 0; 0 10] + 12 .* ([1 / 3, 2 / 3] * [1 / 3, 2 / 3]')
    @test isapprox(Matrix(combined.covariance), expected_cov)
    @test grid_statistic(combined) isa SF.ChiSquared

    problem = SF.FittingProblem(GridTestSource(SF.FitParam(1.0)) => combined)
    stat = grid_statistic(combined)
    config = SF.FittingConfig(problem; stat)
    residual = combined.counts - predict_counts(combined, [3.0])
    expected_chisq = sum(residual .* (expected_cov \ residual))
    @test isapprox(SF.measure_objective!(config, [1.0], 1), expected_chisq)
    @test_throws ArgumentError SF.FittingConfig(problem; stat = SF.Cash())

    exact = combine_datasets(match; mode = :sum, grid = [1.0, 4.0], coordinate = :energy)
    @test exact.counts == [42]
    @test only(exact.variance) == 42
    @test !exact.fractional
    cstat = grid_statistic(exact)
    @test SF.measure(cstat, exact.counts, [42.0], nothing) == 0
    @test SF.measure(cstat, [40.0], exact.counts, nothing) > 0

    singular = combine_datasets(match_dataset_grids(other); mode = :sum,
                                grid = [1.0, 2, 4], coordinate = :energy)
    @test singular.fractional
    @test_throws ArgumentError SF.preferred_units(singular, grid_statistic(singular))
end

@testset "Use native channels before applying the reference grouping" begin
    ref = response_dataset(
        spectrum(1:4, [1, 2, 3, 4]),
        [1.0, 2, 3, 4], [2.0, 3, 4, 5], [1.0], [5.0], fill(1.0, 4, 1),
    )
    other = response_dataset(
        spectrum(0:2, [9, 6, 12]),
        [1.0, 2.5, 4], [2.5, 4.0, 5], [1.0], [5.0], fill(1.0, 3, 1),
    )
    plans = (grouping_plan(ref, :width, 2), grouping_plan(other, :width, 2))
    match = match_dataset_grids(ref, other)
    combined = combine_datasets(match; mode = :sum, groupings = plans, coordinate = :energy)
    @test combined.grid.low == [1, 3]
    @test combined.grid.high == [3, 5]
    @test isapprox(combined.counts, [14, 23])
    @test length(combined.members[2].counts) == 3
    joint = combine_datasets(match; groupings = plans)
    @test joint.counts == [3, 7, 15, 12]
    @test occursin("combined with 4 bins from 2 datasets", repr(MIME"text/plain"(), joint))
    @test occursin("combined with 2 bins from 2 datasets", repr(MIME"text/plain"(), combined))

    rebin = overlap_projection(BinGrid([1.0, 3], [2.0, 4]), BinGrid([1.0, 4]))
    @test rebin * [2.0, 3.0] == [5]
end

@testset "Measured backgrounds and zeros in the Poisson likelihood" begin
    poisson = SF.ErrorStatistics.Poisson
    bkg = spectrum([3, 2, 1], [60, 5, 5]; exposure = 2.0, backscale = 5.0, statistics = poisson)
    on = response_dataset(
        spectrum(1:2, [6, 0]; exposure = 4.0, areascale = 0.5, statistics = poisson),
        [1.0, 2], [2.0, 3], [1.0], [3.0], reshape([1.0, 0], 2, 1);
        background = bkg,
    )
    match = match_dataset_grids(on)
    joint = combine_datasets(match)
    stat = grid_statistic(joint)
    expected = 10 * log1p(0.2)
    @test isapprox(SF.measure(stat, joint.counts, [5.0, 0.0], joint.variance), expected)
    @test isapprox(SF.measure(stat, [5.0, 0.0], joint.counts, joint.variance), expected)
    problem = SF.FittingProblem(GridTestSource(SF.FitParam(1.25)) => joint)
    config = SF.FittingConfig(problem; stat)
    @test isapprox(SF.measure_objective!(config, [1.25], 1), expected)
    @test_throws ArgumentError combine_datasets(match; mode = :sum)
    @test_throws ArgumentError SF.preferred_units(joint, SF.ChiSquared())

    empty = dataset(spectrum(1:2, [0, 3]; statistics = poisson))
    empty_joint = combine_datasets(match_dataset_grids(empty))
    empty_stat = grid_statistic(empty_joint)
    @test SF.measure(empty_stat, empty_joint.counts, [0.0, 3.0], nothing) == 0
    @test isinf(SF.measure(empty_stat, empty_joint.counts, [0.0, 0.0], nothing))
end
