struct _PlotBackground
    counts::Vector{Float64}
    covariance::SparseMatrixCSC{Float64,Int}
    alpha::Float64
end

struct _ChannelPlotBackground
    counts::Vector{Float64}
    covariance::SparseMatrixCSC{Float64,Int}
    off_counts::Vector{Float64}
    alpha::Vector{Float64}
    # Columns map displayed bins to the contributing W-stat bins.
    fit_overlap::SparseMatrixCSC{Float64,Int}
end

_plot_background(counts, variance, alpha::Float64, ::Nothing = nothing) =
    _PlotBackground(copy(counts), spdiagm(0 => variance), alpha)

function _plot_background(counts, variance, alpha::Vector{Float64}, display = nothing)
    scaled, scaled_variance = isnothing(display) ?
        (alpha .* counts, alpha.^2 .* variance) : (display.counts, display.variance)
    return _ChannelPlotBackground(scaled, spdiagm(0 => scaled_variance),
                                  copy(counts), copy(alpha), spdiagm(0 => ones(length(counts))))
end

_scaled_background(off::_PlotBackground) = (off.alpha .* off.counts, off.alpha^2 * off.covariance)
_scaled_background(off::_ChannelPlotBackground) = (off.counts, off.covariance)

function _move_plot_background(off::_PlotBackground, transform)
    return _PlotBackground(transform * off.counts, transform * off.covariance * transpose(transform),
                           off.alpha)
end

function _move_plot_background(off::_ChannelPlotBackground, transform)
    return _ChannelPlotBackground(transform * off.counts, transform * off.covariance * transpose(transform),
                                  off.off_counts, off.alpha, off.fit_overlap * transpose(transform))
end

_reorder_plot_background(off::_PlotBackground, order) =
    _PlotBackground(off.counts[order], off.covariance[order, order], off.alpha)

_reorder_plot_background(off::_ChannelPlotBackground, order) =
    _ChannelPlotBackground(off.counts[order], off.covariance[order, order], off.off_counts,
                           off.alpha, off.fit_overlap[:, order])

"""
One observation on a detector grid. `counts` and `covariance` describe the
on-region measurement. For ordinary `background`, multiply its off-region
`counts` by `alpha` and covariance by `alpha^2` to obtain source-region
equivalents. With channel scaling, `background.counts` and
`background.covariance` are already in source-region units.
`model` holds folded source counts, and `weights` hold response sensitivity
when requested. `coverage` is the fraction of each bin's geometric width
covered by this observation's selected channels in `coordinate`.
"""
struct AlignedDetectorSpectrum
    grid::BinGrid
    coordinate::Symbol
    counts::Vector{Float64}
    covariance::SparseMatrixCSC{Float64,Int}
    background::Union{Nothing,_PlotBackground,_ChannelPlotBackground}
    model::Union{Nothing,Vector{Float64}}
    weights::Union{Nothing,Vector{Float64}}
    coverage::Vector{Float64}
    fractional::Bool
    exposure::Float64
    label::String
end

"""
Aligned detector spectra in input order, with a common `grid` and
`coordinate`. `channels_per_bin` records how many original target bins each
output bin contains. Measurements are available separately in `members`.
"""
struct DetectorAlignment
    grid::BinGrid
    coordinate::Symbol
    members::Vector{AlignedDetectorSpectrum}
    channels_per_bin::Vector{Int}
end

const HZ_PER_KEV = 2.417989242084918e17

function _convert_plot_grid(grid::BinGrid, from, to)
    order = collect(1:length(grid))
    from == to && return grid, order
    if from == :frequency
        energy = BinGrid(grid.low ./ HZ_PER_KEV, grid.high ./ HZ_PER_KEV)
    elseif from == :wavelength
        energy, _ = _detector_grid(grid.low, grid.high, :wavelength)
        reverse!(order)
    else
        energy = grid
    end
    if to == :frequency
        return BinGrid(energy.low .* HZ_PER_KEV, energy.high .* HZ_PER_KEV), order
    elseif to == :wavelength
        iszero(first(energy.low)) &&
            throw(ArgumentError("a zero-energy edge has no finite wavelength; use energy or a positive-energy detector grid"))
        wavelength, _ = _detector_grid(energy.low, energy.high, :wavelength)
        return wavelength, reverse!(order)
    end
    return energy, order
end

function _detector_target(data::Union{GridMatch,JointDataset}, reference)
    if isnothing(reference)
        return (grid = data.detector_target, coordinate = data.detector_coordinate)
    end
    1 <= reference <= length(data.detector_grids) ||
        throw(ArgumentError("reference must index an observation"))
    return (grid = data.detector_grids[reference], coordinate = :energy)
end

"""
    detector_grid(match; coordinate=nothing, reference=nothing)

Return a copy of the ungrouped detector target in its stored coordinate
(keV for the default reference) or convert it to `:energy` (keV),
`:wavelength` (Å), or `:frequency` (Hz). An explicit `reference` selects that
observation's original RMF detector grid. Also accepts a prepared `JointDataset`.
"""
function detector_grid(data::Union{GridMatch,JointDataset}; coordinate = nothing, reference = nothing)
    target = _detector_target(data, reference)
    units = isnothing(coordinate) ? target.coordinate : coordinate
    units in (:energy, :wavelength, :frequency) || throw(ArgumentError("unknown detector coordinate"))
    return _bin_grid(first(_convert_plot_grid(target.grid, target.coordinate, units)))
end

function _move_plot_member(member, grid, coordinate, transform)
    off = member.background
    background = if !isnothing(off)
        _move_plot_background(off, transform)
    end
    model = isnothing(member.model) ? nothing : transform * member.model
    weights = isnothing(member.weights) ? nothing : transform * member.weights
    counts = transform * member.counts
    covariance = transform * member.covariance * transpose(transform)
    covered_width = transform * (member.coverage .* _widths(member.grid))
    coverage = clamp.(covered_width ./ _widths(grid), 0.0, 1.0)
    fractional = member.fractional || any(!=(1.0), nonzeros(transform))
    return AlignedDetectorSpectrum(grid, coordinate, counts, covariance, background, model, weights,
                                   coverage, fractional, member.exposure, member.label)
end

function _orient_plot_member(member, coordinate)
    member.coordinate == coordinate && return member
    grid, order = _convert_plot_grid(member.grid, member.coordinate, coordinate)
    off = member.background
    background = if !isnothing(off)
        _reorder_plot_background(off, order)
    end
    model = isnothing(member.model) ? nothing : member.model[order]
    weights = isnothing(member.weights) ? nothing : member.weights[order]
    return AlignedDetectorSpectrum(grid, coordinate, member.counts[order], member.covariance[order, order],
                                   background, model, weights, member.coverage[order], member.fractional,
                                   member.exposure, member.label)
end

function _wavelength_widths(energy::BinGrid)
    all(>(0), energy.low) || throw(ArgumentError("unfolding requires positive incident energy boundaries"))
    return HC_KEV_ANGSTROM .* ((energy.high .- energy.low) ./ energy.low ./ energy.high)
end

function _unfold_weights(data::_DetectorData, rows, response_rows)
    rmf = data.response
    incident = BinGrid(rmf.bins_low, rmf.bins_high)
    size(rmf.matrix, 2) == length(incident) ||
        throw(DimensionMismatch("response columns must match the incident bins"))
    arf = data.ancillary
    reference = if isnothing(arf)
        _wavelength_widths(incident)
    else
        arf_energy = BinGrid(arf.bins_low, arf.bins_high)
        length(arf.effective_area) == length(arf_energy) ||
            throw(DimensionMismatch("ARF areas must match its incident bins"))
        all(x -> isfinite(x) && x >= 0, arf.effective_area) ||
            throw(ArgumentError("ARF effective areas must be finite and nonnegative"))
        arf_wave, arf_order = _convert_plot_grid(arf_energy, :energy, :wavelength)
        rmf_wave, _ = _convert_plot_grid(incident, :energy, :wavelength)
        integrated = arf.effective_area[arf_order] .* (arf_wave.high .- arf_wave.low)
        reverse(overlap_projection(arf_wave, rmf_wave) * integrated)
    end
    exposure = _effective_exposure(data, data.spectrum, rows; allow_zero = true)
    weights = exposure .* (rmf.matrix[response_rows, :] * reference)
    all(x -> isfinite(x) && x >= 0, weights) ||
        throw(ArgumentError("response weights must be finite and nonnegative"))
    return weights
end

function _read_plot_member(data, rows, geometry; unfold, read_background, model = nothing)
    rows = geometry.ascending ? rows : reverse(rows)
    grid = _detector_energy_grid(geometry.low[rows], geometry.high[rows])
    counts, factors = _in_counts(data.spectrum, rows, grid.high .- grid.low)
    variance, _ = _count_variance(data.spectrum, rows, counts, factors)
    background = if read_background && !isnothing(data.background)
        exposure = _effective_exposure(data, data.spectrum, rows; allow_zero = true)
        off = _background_observation(data, rows, grid.high .- grid.low,
                                      exposure; allow_zero = true)
        var, _ = _count_variance(data.background, off.rows, off.counts, off.factors)
        _plot_background(off.counts, var, off.scale, _background_display(data, rows))
    end
    weights = unfold ? _unfold_weights(data, rows, geometry.response_rows[rows]) : nothing
    label = string(SF.make_label(data))
    return AlignedDetectorSpectrum(grid, :energy, counts, spdiagm(0 => variance), background, model,
                                   weights, ones(length(grid)), false, Float64(data.spectrum.exposure_time), label)
end

function _empty_plot_member(data, target; unfold, has_model)
    n = length(target.grid)
    label = string(SF.make_label(data))
    return AlignedDetectorSpectrum(target.grid, target.coordinate, zeros(n), spzeros(n, n), nothing,
                                   has_model ? zeros(n) : nothing, unfold ? zeros(n) : nothing,
                                   zeros(n), false, Float64(data.spectrum.exposure_time), label)
end

function _read_plot_member(member::_MemberResponse, coordinate, flux, label; unfold)
    model = isnothing(flux) ? nothing : member.response * _native_flux(member, flux)
    weights = unfold ? member.response * _wavelength_widths(member.incident) : nothing
    background = if !isnothing(member.background)
        _plot_background(member.background, member.background_variance, member.alpha,
                         member.background_display)
    end
    return AlignedDetectorSpectrum(member.grid, coordinate, copy(member.counts), spdiagm(0 => member.variance),
                                   background, model, weights, ones(length(member.grid)), false, member.exposure, label)
end

_integrated_plot_flux(edges, model) = SF.invokemodel(edges, model)

function _integrated_plot_flux(edges, model::SF.AbstractSpectralModel{<:Number})
    # A fitted model already contains numerical values instead of FitParams.
    output = SF.allocate_model_output(model, edges)
    SF._inner_invokemodel!(output, edges, model)
    return view(output, :, 1)
end

function _plot_model_flux(edges, model, extension)
    isnothing(model) && return nothing
    model isa SF.AbstractSpectralModel || throw(ArgumentError("model must be a SpectralFitting source model"))
    if isnothing(extension)
        return (; grid = BinGrid(edges), values = collect(Float64, _integrated_plot_flux(edges, model)))
    end
    extended = vcat(extension.low, edges, extension.high)
    values = _integrated_plot_flux(extended, model)
    first_bin = length(extension.low) + 1
    last_bin = first_bin + length(edges) - 2
    return (; grid = BinGrid(edges), values = collect(Float64, view(values, first_bin:last_bin)))
end

function _resolve_plot_input(input, model, index, need_model)
    result = input isa SF.FitResultSlice ? input.parent : input
    if result isa SF.FitResult
        idx = input isa SF.FitResultSlice ? input.index : index
        config = result.config
        data = config.prob.data.d[idx]
        fitted = if need_model && isnothing(model)
            cache = deepcopy(config.parameter_cache)
            all_params = SF.update_free_parameters!(cache, result.u)
            SF.remake_with_parameters(config.prob.model.m[idx], all_params[config.parameter_bindings[idx]])
        elseif need_model
            model
        end
        return data, fitted, config.prob.data.extension[idx], SF.fit_statistic(config)
    end
    return input, (need_model ? model : nothing), nothing, nothing
end

function _plot_plans(groupings, n)
    plans = if isnothing(groupings)
        ntuple(_ -> nothing, n)
    elseif groupings isa GroupingPlan
        (groupings,)
    else
        Tuple(groupings)
    end
    length(plans) == n && all(plan -> isnothing(plan) || plan isa GroupingPlan, plans) ||
        throw(ArgumentError("provide one GroupingPlan or nothing per observation"))
    return plans
end

function _native_fit_plot_input(data, model, extension; groupings, pooling, reference, unfold)
    isnothing(reference) || reference == 1 || throw(ArgumentError("reference must index an observation"))
    plan = only(_plot_plans(groupings, 1))
    pooling && !isnothing(plan) && _pooled_grouping_error()
    target = pooling ? (grid = _original_detector_grid(data), coordinate = :energy) : nothing
    grouped = isnothing(plan) ? nothing : _group_geometry(data, plan)
    rows = isnothing(grouped) ? _selected_rows(data, nothing) : grouped.active_rows
    source = _plot_model_flux(data.domain, model, extension)
    if pooling && isempty(rows)
        member = _empty_plot_member(data, target; unfold, has_model = true)
        return (; members = [member], target, source, summed = false)
    end
    geometry = isnothing(grouped) ? _geometry(data, rows) : grouped.geometry
    fold = SF.objective_transformer(SF.ContiguouslyBinned(u"counts"), data)
    folded = vec(fold(data.domain, copy(source.values)))
    fit_rows = findall(data.data_mask)
    ordered_rows = geometry.ascending ? rows : reverse(rows)
    counts = collect(Float64, folded[searchsortedfirst.(Ref(fit_rows), ordered_rows)])
    member = _read_plot_member(data, rows, geometry; unfold, read_background = true, model = counts)
    if !isnothing(grouped)
        spans = _detector_energy_grid(grouped.low, grouped.high)
        member = _move_plot_member(member, spans, :energy, overlap_projection(member.grid, spans))
    end
    return (; members = [member], target, source, summed = false)
end

_pooled_grouping_error() =
    throw(ArgumentError("groupings applies to native spectra; use rebin=N to group a pooled detector grid"))

function _native_plot_inputs(input, model, extension; groupings, pooling, reference, unfold, read_background)
    datasets = if input isa GridMatch
        input.datasets
    elseif input isa _DetectorData
        (input,)
    elseif input isa Union{Tuple,AbstractVector}
        Tuple(input)
    else
        throw(ArgumentError("provide SpectralData, a grid match, a prepared dataset, or a fit result"))
    end
    !isempty(datasets) && all(data -> data isa _DetectorData, datasets) ||
        throw(ArgumentError("provide at least one SpectralData observation"))
    ref = isnothing(reference) ? (input isa GridMatch ? input.reference : 1) : reference
    1 <= ref <= length(datasets) || throw(ArgumentError("reference must index an observation"))
    plans = _plot_plans(groupings, length(datasets))
    pooling && any(!isnothing, plans) && _pooled_grouping_error()
    shown = if pooling
        datasets
    else
        map(datasets, plans) do data, plan
            isnothing(plan) ? data : grouped_dataset(data, plan)
        end
    end
    has_model = !isnothing(model)
    match = if has_model
        edges = input isa GridMatch ? input.model_edges : nothing
        match_dataset_grids(shown; grid = edges, reference = ref)
    end
    source = has_model ? _plot_model_flux(match.model_edges, model, extension) : nothing
    members = AlignedDetectorSpectrum[]
    target = if pooling
        input isa GridMatch ? _detector_target(input, reference) :
            (grid = _original_detector_grid(datasets[ref]), coordinate = :energy)
    end
    for i in eachindex(shown)
        data = shown[i]
        rows = _selected_rows(data, nothing)
        if pooling && isempty(rows)
            push!(members, _empty_plot_member(data, target; unfold, has_model))
            continue
        end
        geometry = _geometry(data, rows)
        label = string(SF.make_label(data))
        member = if has_model
            prepared = _member_response(data, match.native_grids[i], match.projections[i], :energy, rows, geometry)
            _read_plot_member(prepared, :energy, source.values, label; unfold)
        else
            _read_plot_member(data, rows, geometry; unfold, read_background)
        end
        push!(members, member)
    end
    return (; members, target, source, summed = false)
end

function _prepared_plot_inputs(data::AbstractGridDataset, model, extension; unfold, pooling, reference)
    source = _plot_model_flux(data.model_edges, model, extension)
    flux = isnothing(source) ? nothing : source.values
    summed = data isa SummedDataset
    if summed && !isnothing(reference)
        throw(ArgumentError("a summed dataset already has a reference detector grid"))
    end
    target = if summed
        (grid = data.grid, coordinate = data.coordinate)
    elseif pooling || !isnothing(reference)
        _detector_target(data, reference)
    end
    coordinate = summed ? data.coordinate : :energy
    members = [_read_plot_member(member, coordinate, flux, "$(data.name) [$i]"; unfold)
               for (i, member) in enumerate(data.members)]
    if summed
        members = [_move_plot_member(member, data.grid, coordinate, overlap)
                   for (member, overlap) in zip(members, data.overlaps)]
    end
    return (; members, target, source, summed)
end

function _read_plot_inputs(data, model, extension, fit_stat; groupings, pooling, reference, unfold, read_background)
    if data isa AbstractGridDataset
        isnothing(groupings) || throw(ArgumentError("prepared datasets already have detector grouping; use rebin for display"))
        return _prepared_plot_inputs(data, model, extension; unfold, pooling, reference)
    elseif data isa SF.SpectralData && !isnothing(model) && !isnothing(fit_stat)
        return _native_fit_plot_input(data, model, extension; groupings, pooling, reference, unfold)
    end
    return _native_plot_inputs(data, model, extension; groupings, pooling, reference, unfold, read_background)
end
