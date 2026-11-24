module PlottingExample

import CairoMakie as CM
import SpectralGrouping as SG
using SparseArrays: sparse
using SpectralFitting: SpectralUnits, @u_str

include("bounded-wstat.jl")  # also imports SpectralFitting as SF

function observation(label, detector_edges, incident_edges, counts;
                     quality = zeros(Int, length(counts)), exposure = 10.0)
    n, m = length(counts), length(incident_edges) - 1
    channels = collect(1:n)
    poisson = SF.ErrorStatistics.Poisson
    on = SF.Spectrum{Float64}(
        channels, quality, ones(Int, n), Float64.(counts), u"counts", exposure,
        1.0, 1.0, poisson, nothing, 0.0, "Demo", label,
    )
    off = SF.Spectrum{Float64}(
        copy(channels), zeros(Int, n), ones(Int, n), collect(Float64, 0:(n - 1)), u"counts",
        exposure, 3.0, 1.0, poisson, nothing, 0.0, "Demo", label,
    )
    response = sparse([iszero(quality[i]) ? 25 * exp(-(i - j)^2 / 2) : 0.0
                       for i in 1:n, j in 1:m])
    low, high = detector_edges[1:(end - 1)], detector_edges[2:end]
    rmf = SF.ResponseMatrix(response, channels, low, high,
                            incident_edges[1:(end - 1)], incident_edges[2:end])
    return SF.SpectralData{Float64,Nothing,Nothing}(
        on, rmf, off, nothing, copy(low), copy(high), copy(incident_edges), trues(n), nothing,
    )
end

powerlaw() = SF.PowerLaw(
    K = SF.FitParam(0.2; lower_limit = 0.001, upper_limit = 10.0),
    a = SF.FitParam(1.8; frozen = true),
)

function inputs()
    meg_grid = [1.0, 1.3, 1.7, 2.2, 2.9, 3.8, 4.7]
    heg_grid = [1.7, 2.1, 2.6, 3.3, 4.1, 5.0, 6.4]
    rgs_grid = collect(0.5:0.1:1.1)
    pn_grid = [0.5, 0.8, 1.1, 1.7, 2.5, 4.0, 7.0]
    mos_grid = [0.5, 0.9, 1.3, 1.9, 2.7, 4.3, 7.0]
    megm1 = observation("MEG −1", meg_grid, meg_grid, [26, 19, 18, 9, 7, 5])
    megp1 = observation("MEG +1", meg_grid, meg_grid, [28, 22, 13, 11, 8, 4])
    hegm1 = observation("HEG −1", heg_grid, heg_grid, [18, 16, 15, 9, 6, 4])
    hegp1 = observation("HEG +1", heg_grid, heg_grid, [20, 15, 12, 8, 5, 3])
    rgs1 = observation("RGS1", rgs_grid, rgs_grid, [16, 400, 400, 0, 25, 400];
                       quality = [0, 1, 1, 0, 0, 1])
    rgs2 = observation("RGS2", rgs_grid, rgs_grid, [400, 18, 22, 400, 30, 400];
                       quality = [1, 0, 0, 1, 0, 1])
    pn = observation("PN", pn_grid, pn_grid, [61, 53, 40, 22, 14, 9])
    mos = observation("MOS", mos_grid, mos_grid, [42, 28, 21, 16, 9, 5]; exposure = 13.0)
    return (; megm1, megp1, hegm1, hegp1, rgs1, rgs2, pn, mos)
end

function demo(; output_dir = nothing)
    (; megm1, megp1, hegm1, hegp1, rgs1, rgs2, pn, mos) = inputs()
    current = powerlaw()
    meg = SG.combine_datasets(SG.match_dataset_grids(megm1, megp1); name = "MEG")
    heg = SG.combine_datasets(SG.match_dataset_grids(hegm1, hegp1); name = "HEG")
    problem = SF.FittingProblem(current => meg, powerlaw() => heg)
    SF.bindall!(problem, :K)
    cfg = SF.FittingConfig(problem; stat = SG.grid_statistic(meg, heg))
    result = fit_bounded_wstat(cfg)

    counts = SG.plot_counts(megm1; model = current)
    rates = SG.plot_data(megm1; xaxis = :frequency, normalization = :rate,
                         yscale = :log, errorbars = false)
    unfolded = SG.plot_unfold(result[1]; xaxis = :wavelength, flux_unit = :ergs,
                              model_display = :source, combine = :sum)

    rgs = SG.match_dataset_grids(rgs1, rgs2; reference = 2)
    gaps = SG.plot_counts(rgs; xaxis = :wavelength, combine = :sum,
                          background = :include, markersize = 5, color = :darkgreen)
    numeric = only(SG.plot_curves(rgs; kind = :counts, combine = :sum,
                                  background = :include).data)
    @assert numeric.y[numeric.valid] == [16, 18, 22, 0, 55]
    @assert !numeric.valid[6] && isnan(numeric.y[6])

    # Both top and residual axes are reused.
    pmeg = SG.plot_fit(result; index = 1, combine = :sum, res = 4,
                       xaxis = :wavelength, color = :steelblue, linewidth = 2)
    pheg = SG.plot_fit(result; index = 2, combine = :sum, res = 4,
                       xaxis = :wavelength, color = :darkorange, markersize = 4,
                       axes = pmeg.axes, figure = pmeg.figure, errorbars = false)
    for collection in (pmeg.artists.model, pheg.artists.model), group in collection, line in group
        line.color = :black
    end
    CM.axislegend(pmeg.axes.data)

    columns = CM.Figure(size = (1000, 500))
    left = (data = CM.Axis(columns[1, 1]; title = "MEG"), residual = CM.Axis(columns[2, 1]))
    right = (data = CM.Axis(columns[1, 2]; title = "HEG"), residual = CM.Axis(columns[2, 2]))
    for (index, axes) in enumerate((left, right))
        SG.plot_fit(result; index, axes, figure = columns, combine = :sum, res = 4)
    end
    CM.rowsize!(columns.layout, 2, CM.Auto(false, 0.32))
    CM.rowgap!(columns.layout, 6)
    CM.xlims!(left.residual, 1, 4.7)
    CM.xlims!(right.residual, 1.7, 6.4)
    right.residual.xticks = 2:2:6

    stacked = CM.Figure(size = (720, 700))
    ccd_layout = CM.GridLayout(stacked[1, 1])
    rgs_layout = CM.GridLayout(stacked[2, 1])
    ccd_axes = (data = CM.Axis(ccd_layout[1, 1]; title = "PN and MOS"),
                residual = CM.Axis(ccd_layout[2, 1]))
    rgs_axes = (data = CM.Axis(rgs_layout[1, 1]; title = "RGS"),
                residual = CM.Axis(rgs_layout[2, 1]))
    SG.plot_data(pn; model = current, axes = ccd_axes, figure = stacked,
                 res = 1, color = :steelblue)
    SG.plot_data(mos; model = current, axes = ccd_axes, figure = stacked,
                 res = 1, color = :darkorange, legend = true)
    SG.plot_data(rgs; model = current, axes = rgs_axes, figure = stacked,
                 combine = :sum, xaxis = :wavelength, res = 4)
    for layout in (ccd_layout, rgs_layout)
        CM.rowsize!(layout, 2, CM.Auto(false, 0.32))
        CM.rowgap!(layout, 6)
    end
    CM.xlims!(ccd_axes.residual, 0.5, 7)
    CM.xlims!(rgs_axes.residual, 11, 25)

    if !isnothing(output_dir)
        mkpath(output_dir)
        CM.save(joinpath(output_dir, "hetg-panels.pdf"), columns)
        CM.save(joinpath(output_dir, "rgs-gaps.png"), gaps.figure)
    end
    return (; counts, rates, unfolded, gaps, numeric, pmeg, pheg, columns, stacked, result)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    PlottingExample.demo(; output_dir = isempty(ARGS) ? "spectral-plots" : only(ARGS))
end
