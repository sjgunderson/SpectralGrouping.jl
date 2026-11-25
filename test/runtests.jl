using SparseArrays: sparse, issparse, nonzeros
using SpectralFitting: SpectralUnits, @u_str
import SpectralFitting as SF
using SpectralGrouping
using Test

function spectrum(channels, counts;
                  units = u"counts", exposure = 10.0, backscale = 1.0, areascale = 1.0,
                  flags = ones(Int, length(channels)), quality = zeros(Int, length(channels)),
                  errors = ones(length(channels)), statistics = SF.ErrorStatistics.Numeric)
    return SF.Spectrum{Float64}(
        collect(channels), quality, flags, collect(Float64, counts), units, exposure, backscale, areascale,
        statistics, isnothing(errors) ? nothing : collect(Float64, errors), 0.0, "TEST", "TEST",
    )
end

function dataset(spec;
                 low = collect(1.0:length(spec.channels)), high = low .+ 1.0,
                 mask = trues(length(spec.channels)), background = nothing, ancillary = nothing)
    n = length(spec.channels)
    matrix = sparse(reshape(collect(1.0:2n), n, 2))
    response = SF.ResponseMatrix(matrix, copy(spec.channels), copy(low), copy(high), [1.0, 2.0], [2.0, 3.0])
    return SF.SpectralData{Float64, Nothing, Nothing}(
        spec, response, background, ancillary, copy(low), copy(high), [1.0, 2.0, 3.0], mask, nothing,
    )
end

@testset "Counts, channel selection, and response folding" begin
    channels = [0, 1, 2, 3, 4, 5, 7, 8, 9]
    source = spectrum(channels, [3, 7, 4, 99, 99, 10, 6, 4, 2];
                      quality = [0, 0, 0, 0, 5, 0, 0, 0, 0], errors = Float64.(1:9))
    bkg_channels = collect(9:-1:-1)
    background = spectrum(bkg_channels, bkg_channels .+ 2; exposure = 20.0, backscale = 3.0, areascale = 0.5)
    arf = SF.AncillaryResponse([1.0, 2.0], [2.0, 3.0], [1.25, 2.5])
    data = dataset(source; background, ancillary = arf,
                   low = [0.5, 1, 1.5, 2, 2.5, 3, 4, 4.5, 5.5],
                   high = [1.0, 1.5, 2, 2.5, 3, 3.5, 4.5, 5, 6],
                   mask = BitVector([true, true, true, false, true, true, true, true, true]))
    plan = grouping_plan(data, :counts, 10)
    @test plan.groups == [1:2, 3:3, 6:6, 7:8, 9:9]
    @test plan.met_target == [true, false, true, true, false]
    grouped = grouped_dataset(data, plan)
    @test grouped.spectrum.data == [10, 4, 10, 10, 2]
    @test group_counts(data, 10).spectrum.data == grouped.spectrum.data
    @test SF.make_objective(SF.ContiguouslyBinned(u"counts"), grouped) == grouped.spectrum.data
    @test isapprox(grouped.spectrum.errors, sqrt.([5, 9, 36, 113, 81]))
    @test grouped.background.data == [5, 4, 7, 19, 11]
    @test isapprox(grouped.background.errors, sqrt.([2, 1, 1, 2, 1]))
    @test grouped.background.exposure_time == 20
    @test grouped.background.background_scale == 3
    @test grouped.background.area_scale == 0.5
    @test grouped.energy_low == [0.5, 1.5, 3, 4, 5.5]
    @test grouped.energy_high == [1.5, 2, 3.5, 5, 6]
    @test grouped.domain == data.domain
    @test grouped.ancillary.effective_area == data.ancillary.effective_area
    @test issparse(grouped.response.matrix)
    incident_flux = [2.0, 3.0]
    native_prediction = 10 .* (data.response.matrix * (arf.effective_area .* incident_flux))
    transformer = SF.objective_transformer(SF.ContiguouslyBinned(u"counts"), grouped)
    @test isapprox(vec(transformer(grouped.domain, incident_flux)), group_sum(plan, native_prediction))
    @test data.spectrum.data == [3, 7, 4, 99, 99, 10, 6, 4, 2]
    @test count(data.data_mask) == 8
    @test length(data.background.channels) == 11
    @test size(data.response.matrix) == (9, 2)

    bad_background = deepcopy(data)
    popfirst!(bad_background.background.channels)
    @test_throws ArgumentError grouped_dataset(bad_background, plan)
end

@testset "Widths, S/N and OGIP flags" begin
    data = dataset(spectrum(0:2, [1, 3, 4]))
    @test group_width(data, 2).spectrum.data == [4, 4]
    @test grouping_plan(data, :width, 2).met_target == [true, false]
    @test group_snr(data, 2).spectrum.data == [4, 4]
    @test group_snr(data, 2; errors = [2, 2, 2]).spectrum.data == [8]
    @test grouping_plan(data, :snr, 2; errors = zeros(3)).met_target == [false]
    @test grouping_plan(data, :counts, 8; max_channels = 1).met_target == falses(3)
    @test group_counts(data, 100).spectrum.data == [8]
    @test group_width(data, 2; mask = Bool[true, false, true]).spectrum.data == [1, 4]
    missing_channel = dataset(spectrum([0, 2], [1, 3]))
    @test grouping_plan(missing_channel, :width, 2).groups == [1:1, 2:2]
    @test_throws ArgumentError group_width(data, 0)
    @test_throws ArgumentError group_counts(data, Inf)
    @test_throws ArgumentError group_counts(data, 2; mask = falses(3))
    @test_throws DimensionMismatch group_snr(data, 2; errors = ones(2))

    file = dataset(spectrum(1:6, 1:6; flags = [1, -1, -1, 0, 1, -1]);
                   mask = BitVector([false, true, true, true, true, true]))
    @test grouping_plan(file, :ogip).groups == [2:3, 4:4, 5:6]
    @test group_ogip(file).spectrum.data == [5, 4, 11]
    @test data.spectrum.grouping == ones(Int, 3)
end

@testset "RATE, density, Poisson errors and reversed detector grid" begin
    rates = dataset(spectrum(0:2, [0.2, 0.3, 0.1]; units = u"counts / s", statistics = SF.ErrorStatistics.Poisson))
    grouped = group_counts(rates, 5)
    @test isapprox(grouped.spectrum.data, [0.5, 0.1])
    @test isapprox(grouped.spectrum.errors, SF.count_error.([5, 1], 1.0) ./ 10)

    density = dataset(spectrum(0:2, [2, 4, 1]; units = u"counts / keV", errors = fill(2.0, 3));
                      low = [1.0, 1.5, 3.0], high = [1.5, 3.0, 4.0])
    density_grouped = group_counts(density, 7)
    @test isapprox(density_grouped.spectrum.data, [3.5, 1.0])
    @test isapprox(density_grouped.spectrum.errors, [sqrt(10) / 2, 2.0])

    reversed = dataset(spectrum(0:2, [1, 3, 4]); low = [3.0, 2.0, 1.0])
    plan = grouping_plan(reversed, :width, 2)
    @test plan.groups == [3:3, 1:2]
    @test grouped_dataset(reversed, plan).energy_low == [1, 2]
    @test group_sum(plan, [10, 20, 30]) == [30, 30]
end

include("grid-matching.jl")
include("multi-grid-wstat.jl")
include("fitting-bindings.jl")
include("plotting.jl")
include("detector-display.jl")
include("shared-detector-rebin.jl")
include("channel-scalings.jl")
include("plot-rendering.jl")
