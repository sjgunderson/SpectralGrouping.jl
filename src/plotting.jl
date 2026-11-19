"""One spectrum on a physical grid, with display values, errors, and validity."""
struct PlotCurve
    grid::BinGrid
    y::Vector{Float64}
    error::Union{Nothing,Vector{Float64}}
    covariance::Union{Nothing,SparseMatrixCSC{Float64,Int}}
    valid::BitVector
    label::String
    fractional::Bool
    weights::Union{Nothing,Vector{Float64}}
end

"""Numerical curves and axis labels, independent of the plotting backend."""
struct SpectrumPlot
    data::Vector{PlotCurve}
    model::Vector{PlotCurve}
    residuals::Vector{PlotCurve}
    xlabel::String
    ylabel::String
    residual_label::String
    residual_baseline::Float64
end

const KEV_TO_ERG = 1.602176634e-9

_widths(grid::BinGrid) = grid.high .- grid.low
_plot_label(members) = join((member.label for member in members), " + ")
_coverage(members) = reduce((a, b) -> a .| b, (member.covered for member in members))

function _power_weights(grid, power, power_scale)
    power == 1 && return ones(length(grid))
    lo, hi = grid.low, grid.high
    weights = if power_scale == :arithmetic
        (lo ./ 2 .+ hi ./ 2) .^ (power - 1)
    elseif power_scale == :geometric
        (sqrt.(lo) .* sqrt.(hi)) .^ (power - 1)
    elseif power == 0
        ratio = lo ./ hi
        (1 .+ ratio .+ ratio .^ 2) ./ (3 .* lo)
    elseif power == 2
        delta = (hi .- lo) ./ lo
        hi .* log1p.(delta) ./ delta
    else
        lo .* hi
    end
    all(x -> isfinite(x) && x >= 0, weights) ||
        throw(ArgumentError("the selected power weighting requires finite, suitable bin boundaries"))
    return weights
end

function _plot_align(members, grid, coordinate)
    return map(members) do member
        oriented = _orient_plot_member(member, coordinate)
        if oriented.grid.low == grid.low && oriented.grid.high == grid.high
            return oriented
        end
        projection = overlap_projection(oriented.grid, grid)
        return _move_plot_member(oriented, grid, coordinate, projection)
    end
end

function _plot_rebin(members, factor)
    factor == 1 && return members
    base = first(members)
    grid = base.grid
    usable = _coverage(members)
    if !isnothing(base.weights)
        usable .&= reduce(+, (member.weights for member in members)) .> 0
    end

    starts, stops, destinations = Int[], Int[], Int[]
    start = firstindex(grid.low)
    for i in eachindex(grid.low)
        push!(destinations, length(starts) + 1)
        at_end = i == length(grid)
        split = at_end || !usable[i] || !usable[i + 1] ||
            !_roundtrip_edge(grid.high[i], grid.low[i + 1])
        if split || i - start + 1 == factor
            push!(starts, start)
            push!(stops, i)
            start = i + 1
        end
    end
    rebinned = BinGrid(grid.low[starts], grid.high[stops])
    projection = sparse(destinations, collect(1:length(grid)), ones(length(grid)),
                        length(rebinned), length(grid))
    return [_move_plot_member(member, rebinned, base.coordinate, projection) for member in members]
end

function _plot_coefficients(members, normalization, combine)
    n = length(members)
    if normalization == :counts
        combine in (:exposure, :mean_exposure) &&
            throw(ArgumentError("exposure combination requires a rate normalization"))
        return fill(combine == :mean ? 1 / n : 1.0, n)
    end
    exposures = [member.exposure for member in members]
    all(x -> isfinite(x) && x > 0, exposures) ||
        throw(ArgumentError("rate plots require a positive PHA exposure for each observation"))
    if combine in (:exposure, :mean_exposure)
        return fill((combine == :exposure ? 1.0 : n) / sum(exposures), n)
    end
    return (combine == :mean ? 1 / n : 1.0) ./ exposures
end

function _display_counts(member, background)
    off = member.background
    background == :only && isnothing(off) &&
        return zeros(length(member.grid)), spzeros(length(member.grid), length(member.grid)), nothing
    isnothing(off) && return member.counts, member.covariance, member.model
    scaled = off.alpha .* off.counts
    if background == :only
        return scaled, off.alpha^2 * off.covariance, nothing
    elseif background == :subtract
        return member.counts - scaled, member.covariance + off.alpha^2 * off.covariance, member.model
    end
    model = isnothing(member.model) ? nothing : member.model + scaled
    return member.counts, member.covariance, model
end

function _make_plot_curve(grid, counts, covariance, multiplier, valid, label, fractional;
                          offset = 0.0, weights = nothing)
    values = ifelse.(valid, counts .* multiplier .+ offset, NaN)
    errors, transformed = if isnothing(covariance)
        nothing, nothing
    else
        scaling = spdiagm(0 => ifelse.(valid, multiplier, 0.0))
        transformed = scaling * covariance * scaling
        ifelse.(valid, sqrt.(max.(0.0, diag(transformed))), NaN), transformed
    end
    return PlotCurve(grid, values, errors, transformed, BitVector(valid), label, fractional, weights)
end

function _flux_multiplier(grid, xaxis, flux_unit, power, power_scale)
    weighting = _power_weights(grid, power, power_scale)
    flux_unit == :photons && return weighting
    conversion = if xaxis == :wavelength
        KEV_TO_ERG * HC_KEV_ANGSTROM
    elseif xaxis == :frequency
        KEV_TO_ERG / HZ_PER_KEV
    else
        KEV_TO_ERG
    end
    return conversion .* weighting
end

function _display_run(members, opts)
    base = first(members)
    grid, order = _convert_plot_grid(base.grid, base.coordinate, opts.xaxis)
    n = length(grid)
    covered = _coverage(members)[order]
    unfolding = opts.kind == :unfold
    coefficients = unfolding ? ones(length(members)) :
        _plot_coefficients(members, opts.normalization, opts.combine)
    counts, covariance = zeros(n), spzeros(n, n)
    prediction = opts.show_model && !isnothing(base.model) && opts.background != :only ? zeros(n) : nothing
    for (member, coefficient) in zip(members, coefficients)
        values, cov, model = _display_counts(member, opts.background)
        counts .+= coefficient .* values
        covariance += coefficient^2 * cov
        if !isnothing(prediction)
            prediction .+= coefficient .* model
        end
    end
    valid = covered
    weights = nothing
    multiplication = if unfolding
        wavelength, wave_order = _convert_plot_grid(base.grid, base.coordinate, :wavelength)
        wavelength_width = _widths(wavelength)[invperm(wave_order)][order]
        weights = reduce(+, (member.weights for member in members))[order]
        valid = covered .& isfinite.(weights) .& (weights .> 0)
        any(valid) || throw(ArgumentError("the selected bins have no finite positive response sensitivity"))
        multiplier = _flux_multiplier(grid, opts.xaxis, opts.flux_unit, opts.power, opts.power_scale)
        wavelength_width .* multiplier ./ _widths(grid) ./ ifelse.(valid, weights, 1.0)
    else
        multiplier = _power_weights(grid, opts.power, opts.power_scale)
        opts.normalization == :rate_density ? multiplier ./ _widths(grid) : multiplier
    end
    multiplication .*= opts.scale
    fractional = any(member -> member.fractional, members)
    label = _plot_label(members)
    curve = _make_plot_curve(grid, counts[order], covariance[order, order], multiplication,
                             valid, label, fractional; offset = opts.offset, weights)
    model = if !isnothing(prediction)
        _make_plot_curve(grid, prediction[order], nothing, multiplication,
                         valid, label, fractional; offset = opts.offset)
    end
    return curve, model
end

function _source_plot_curve(source, data_curve, opts)
    grid, order = _convert_plot_grid(source.grid, :energy, opts.xaxis)
    # Draw only source bins wholly inside a run of valid detector bins.
    bounds = data_curve.grid
    runs = Tuple{Float64,Float64}[]
    for i in eachindex(bounds.low)
        data_curve.valid[i] || continue
        if !isempty(runs) && _roundtrip_edge(last(runs)[2], bounds.low[i])
            runs[end] = (last(runs)[1], bounds.high[i])
        else
            push!(runs, (bounds.low[i], bounds.high[i]))
        end
    end
    valid = falses(length(grid))
    j = 1
    for i in eachindex(grid.low)
        lo, hi = grid.low[i], grid.high[i]
        while j <= length(runs) && runs[j][2] < hi && !_roundtrip_edge(runs[j][2], hi)
            j += 1
        end
        j > length(runs) && break
        valid[i] = runs[j][1] <= lo || _roundtrip_edge(runs[j][1], lo)
    end
    multiplier = opts.scale .* _flux_multiplier(grid, opts.xaxis, opts.flux_unit, opts.power, opts.power_scale) ./
        _widths(grid)
    return _make_plot_curve(grid, source.values[order], nothing, multiplier, valid,
                            data_curve.label, false; offset = opts.offset)
end

function _plot_labels(opts, statistic, res, fractional)
    xlabel, unit, symbol = if opts.xaxis == :energy
        "Energy (keV)", "keV", "E"
    elseif opts.xaxis == :wavelength
        "Wavelength (Å)", "Å", "λ"
    else
        "Frequency (Hz)", "Hz", "ν"
    end
    exponent = opts.power - 1
    ylabel = if opts.kind == :unfold && opts.flux_unit == :photons
        "Photon flux density (photons cm⁻² s⁻¹ $(unit)⁻¹)"
    elseif opts.kind == :unfold
        energy_exponent = opts.xaxis == :wavelength ? exponent + 1 : exponent - 1
        exponent = energy_exponent
        "Energy flux density (erg cm⁻² s⁻¹ $(unit)⁻¹)"
    elseif opts.normalization == :rate_density
        "Count rate density (counts s⁻¹ $(unit)⁻¹)"
    elseif opts.normalization == :rate
        "Count rate (counts s⁻¹)"
    else
        "Counts per bin"
    end
    if opts.kind == :unfold && opts.flux_unit == :ergs && exponent == 1
        ylabel = "$symbol F_$symbol (erg cm⁻² s⁻¹)"
    elseif exponent != 0
        ylabel = "$symbol^$exponent × $ylabel"
    end
    label = if res == 0
        ""
    elseif mod1(res, 3) == 3
        "Data / model"
    elseif statistic == :chi2
        mod1(res, 3) == 1 ? "Residual (σ)" : "Signed χ²"
    else
        stat = statistic == :wstat ? "W" : "C"
        prefix = mod1(res, 3) == 1 ? "Signed √Δ$stat" : "Signed Δ$stat"
        fractional ? prefix * " (rebinned diagnostic)" : prefix
    end
    return xlabel, ylabel, label
end

"""
    plot_curves(input; kind=:data, xaxis=:energy, combine=:none, rebin=1, ...)

Prepare detector-bin curves without loading a graphics backend. Pass a
SpectralData, GridMatch, prepared joint/summed dataset, or SpectralFitting fit
result. Use `kind=:counts` for counts per bin and `kind=:unfold` for
response-corrected flux. For native inputs, pass `model` to overlay a source
and `groupings` to apply detector plans. Fit results supply parameter values.
"""
function plot_curves(
    input; kind::Symbol = :data, xaxis::Symbol = :energy, normalization::Symbol = :rate_density,
    combine::Symbol = :none, matching_coordinate::Symbol = :wavelength, reference = nothing,
    groupings = nothing, rebin::Integer = 1, background::Symbol = :subtract,
    model = nothing, index::Integer = 1, res::Integer = 0, residual_stat::Symbol = :auto,
    show_model::Bool = true, model_display::Symbol = :convolved, flux_unit::Symbol = :photons,
    power = nothing, power_scale::Symbol = :arithmetic, scale::Real = 1.0, offset::Real = 0.0,
)
    kind in (:data, :counts, :unfold) || throw(ArgumentError("kind must be :data, :counts, or :unfold"))
    xaxis in (:energy, :wavelength, :frequency) || throw(ArgumentError("unknown display coordinate"))
    matching_coordinate in (:energy, :wavelength) || throw(ArgumentError("unknown matching coordinate"))
    normalization in (:counts, :rate, :rate_density) || throw(ArgumentError("unknown count normalization"))
    combine in (:none, :sum, :mean, :exposure, :mean_exposure) ||
        throw(ArgumentError("unknown combination mode"))
    background in (:subtract, :include, :only) || throw(ArgumentError("unknown background view"))
    model_display in (:convolved, :source) || throw(ArgumentError("unknown unfolded model display"))
    flux_unit in (:photons, :ergs) || throw(ArgumentError("flux_unit must be :photons or :ergs"))
    power_scale in (:arithmetic, :geometric, :powerlaw) || throw(ArgumentError("unknown power weighting"))
    0 <= res <= 6 || throw(ArgumentError("res must be between 0 and 6"))
    rebin > 0 || throw(ArgumentError("rebin must be positive"))
    isfinite(scale) && scale > 0 && isfinite(offset) ||
        throw(ArgumentError("scale must be positive and finite, and offset must be finite"))
    kind == :counts && (normalization = :counts)
    unfolding = kind == :unfold
    if unfolding && !(combine in (:none, :sum))
        throw(ArgumentError("unfolded spectra combine by pooling counts and response weights"))
    end
    if isnothing(power)
        power = unfolding && flux_unit == :ergs && xaxis != :wavelength ? 3 : 1
    end
    power isa Integer && 0 <= power <= 3 || throw(ArgumentError("power must be an integer from 0 to 3"))
    data, source_model, extension, fit_stat = _resolve_plot_input(
        input, model, index, (show_model && background != :only) || res != 0,
    )
    res > 0 && isnothing(source_model) && throw(ArgumentError("residuals require a source model or a fit result"))
    pooling = combine != :none || data isa SummedDataset
    inputs = if data isa AbstractGridDataset
        isnothing(groupings) || throw(ArgumentError("prepared datasets already have detector grouping; use rebin for display"))
        _prepared_plot_inputs(data, source_model, extension; unfold = unfolding)
    elseif data isa SF.SpectralData && !isnothing(source_model) && !isnothing(fit_stat)
        _native_fit_plot_input(data, source_model, extension; groupings, unfold = unfolding)
    else
        _native_plot_inputs(data, source_model, extension; groupings, pooling, reference,
                            unfold = unfolding, read_background = background != :include)
    end
    members = inputs.members
    groups = if pooling
        if inputs.summed
            # The detector overlap matrices on the fitted sum define the physical view.
            if combine == :none
                combine = normalization == :counts || unfolding ? :sum : :exposure
            end
        else
            reference_grid = isnothing(inputs.target) ? members[inputs.reference].grid : inputs.target
            grid, _ = _convert_plot_grid(reference_grid, :energy, matching_coordinate)
            members = _plot_align(members, grid, matching_coordinate)
        end
        [_plot_rebin(members, rebin)]
    else
        [_plot_rebin([member], rebin) for member in members]
    end

    opts = (; kind, xaxis, normalization, combine, background, show_model, flux_unit,
            power, power_scale, scale = Float64(scale), offset = Float64(offset))
    curves, models, residuals = PlotCurve[], PlotCurve[], PlotCurve[]
    statistic = _plot_statistic(residual_stat, fit_stat)
    if res != 0 && mod1(res, 3) != 3 && statistic == :wstat &&
            all(m -> isnothing(m.background), members)
        throw(ArgumentError("W-stat residuals require measured off-region counts"))
    end
    for group in groups
        curve, folded = _display_run(group, opts)
        push!(curves, curve)
        if !isnothing(folded)
            displayed = unfolding && model_display == :source ?
                _source_plot_curve(inputs.source, curve, opts) : folded
            push!(models, displayed)
        end
        if res in 1:3
            append!(residuals, [_residual_curve([member], xaxis, res, statistic, background) for member in group])
        elseif res != 0
            push!(residuals, _residual_curve(group, xaxis, res, statistic, background))
        end
    end
    xlabel, ylabel, residual_label = _plot_labels(opts, statistic, res, any(c -> c.fractional, residuals))
    baseline = res > 0 && mod1(res, 3) == 3 ? 1.0 : 0.0
    return SpectrumPlot(curves, models, residuals, xlabel, ylabel, residual_label, baseline)
end

function _render_plot(plot; kwargs...)
    throw(ArgumentError("load CairoMakie to render spectra, or use plot_curves for the numerical curves"))
end

function _plot(
    input, kind; figure = nothing, axes = nothing, xscale = nothing, yscale = nothing,
    color = nothing, linewidth = 1.5, markersize = 0, errorbars::Bool = true, legend::Bool = false,
    kwargs...,
)
    plot = plot_curves(input; kind, kwargs...)
    return _render_plot(plot; figure, axes, xscale, yscale, color, linewidth, markersize, errorbars, legend)
end

"""Render a rate-density spectrum or a caller-selected count normalization."""
plot_data(input; kwargs...) = _plot(input, :data; kwargs...)

"""Render counts per detector bin, with optional ISIS-style weighting."""
plot_counts(input; kwargs...) = _plot(input, :counts; kwargs...)

"""Render a fit result, optionally adding an ISIS-style residual panel."""
plot_fit(result::Union{SF.FitResult,SF.FitResultSlice}; kwargs...) = _plot(result, :data; kwargs...)

"""Render response-corrected photon or energy flux without requiring a fit."""
plot_unfold(input; kwargs...) = _plot(input, :unfold; kwargs...)
