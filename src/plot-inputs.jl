struct _PlotBackground
    counts::Vector{Float64}
    covariance::SparseMatrixCSC{Float64,Int}
    alpha::Float64
end

struct _PlotMember
    grid::BinGrid
    coordinate::Symbol
    counts::Vector{Float64}
    covariance::SparseMatrixCSC{Float64,Int}
    background::Union{Nothing,_PlotBackground}
    model::Union{Nothing,Vector{Float64}}
    weights::Union{Nothing,Vector{Float64}}
    covered::BitVector
    fractional::Bool
    exposure::Float64
    label::String
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
        wavelength, _ = _detector_grid(energy.low, energy.high, :wavelength)
        return wavelength, reverse!(order)
    end
    return energy, order
end

function _move_plot_member(member, grid, coordinate, transform)
    off = member.background
    background = if !isnothing(off)
        _PlotBackground(transform * off.counts, transform * off.covariance * transpose(transform),
                        off.alpha)
    end
    model = isnothing(member.model) ? nothing : transform * member.model
    weights = isnothing(member.weights) ? nothing : transform * member.weights
    counts = transform * member.counts
    covariance = transform * member.covariance * transpose(transform)
    covered = BitVector(transform * Float64.(member.covered) .> 0)
    fractional = member.fractional || any(!=(1.0), nonzeros(transform))
    return _PlotMember(grid, coordinate, counts, covariance, background, model, weights,
                       covered, fractional, member.exposure, member.label)
end

function _orient_plot_member(member, coordinate)
    member.coordinate == coordinate && return member
    grid, order = _convert_plot_grid(member.grid, member.coordinate, coordinate)
    off = member.background
    background = if !isnothing(off)
        _PlotBackground(off.counts[order], off.covariance[order, order], off.alpha)
    end
    model = isnothing(member.model) ? nothing : member.model[order]
    weights = isnothing(member.weights) ? nothing : member.weights[order]
    return _PlotMember(grid, coordinate, member.counts[order], member.covariance[order, order],
                       background, model, weights, member.covered[order], member.fractional,
                       member.exposure, member.label)
end

function _wavelength_widths(energy::BinGrid)
    all(>(0), energy.low) || throw(ArgumentError("unfolding requires positive incident energy boundaries"))
    return HC_KEV_ANGSTROM .* ((energy.high .- energy.low) ./ energy.low ./ energy.high)
end

function _unfold_weights(data::SF.SpectralData, response_rows)
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
    spec = data.spectrum
    exposure, area_scale = spec.exposure_time, spec.area_scale
    isfinite(exposure) && exposure > 0 || throw(ArgumentError("unfolding requires a positive PHA exposure"))
    isfinite(area_scale) && area_scale >= 0 || throw(ArgumentError("AREASCAL must be finite and nonnegative"))
    weights = (Float64(exposure) * Float64(area_scale)) .* (rmf.matrix[response_rows, :] * reference)
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
        exposure, area = data.spectrum.exposure_time, data.spectrum.area_scale
        isfinite(exposure) && exposure > 0 && isfinite(area) && area >= 0 ||
            throw(ArgumentError("background scaling requires positive exposure and nonnegative AREASCAL"))
        off = _background_observation(data, rows, grid.high .- grid.low,
                                      Float64(exposure) * Float64(area); allow_zero = true)
        var, _ = _count_variance(data.background, off.rows, off.counts, off.factors)
        _PlotBackground(off.counts, spdiagm(0 => var), off.scale)
    end
    weights = unfold ? _unfold_weights(data, geometry.response_rows[rows]) : nothing
    label = string(SF.make_label(data))
    return _PlotMember(grid, :energy, counts, spdiagm(0 => variance), background, model,
                       weights, trues(length(grid)), false, Float64(data.spectrum.exposure_time), label)
end

function _empty_plot_member(data, target; unfold, has_model)
    n = length(target)
    label = string(SF.make_label(data))
    return _PlotMember(target, :energy, zeros(n), spzeros(n, n), nothing,
                       has_model ? zeros(n) : nothing, unfold ? zeros(n) : nothing,
                       falses(n), false, Float64(data.spectrum.exposure_time), label)
end

function _read_plot_member(member::_MemberResponse, coordinate, flux, label; unfold)
    model = isnothing(flux) ? nothing : member.response * _native_flux(member, flux)
    weights = unfold ? member.response * _wavelength_widths(member.incident) : nothing
    background = if !isnothing(member.background)
        _PlotBackground(copy(member.background), spdiagm(0 => member.background_variance), member.alpha)
    end
    return _PlotMember(member.grid, coordinate, copy(member.counts), spdiagm(0 => member.variance),
                       background, model, weights, trues(length(member.grid)), false, member.exposure, label)
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

function _coarsen_plot_grid(original, spans)
    low, high = Float64[], Float64[]
    i, group = 1, 1
    while i <= length(original)
        stop = i
        while group <= length(spans) && spans.low[group] < original.low[i] &&
                !_roundtrip_edge(spans.low[group], original.low[i])
            group += 1
        end
        if group <= length(spans) && _roundtrip_edge(spans.low[group], original.low[i])
            endpoint = spans.high[group]
            while stop < length(original) && original.high[stop] < endpoint &&
                    !_roundtrip_edge(original.high[stop], endpoint) &&
                    _roundtrip_edge(original.high[stop], original.low[stop + 1])
                stop += 1
            end
            _roundtrip_edge(original.high[stop], endpoint) || (stop = i)
            group += 1
        end
        push!(low, original.low[i])
        push!(high, original.high[stop])
        i = stop + 1
    end
    return BinGrid(low, high)
end

function _native_fit_plot_input(data, model, extension; groupings, pooling, reference, unfold)
    isnothing(reference) || reference == 1 || throw(ArgumentError("reference must index an observation"))
    plan = only(_plot_plans(groupings, 1))
    target = pooling ? _original_detector_grid(data) : nothing
    if isnothing(plan)
        rows = _selected_rows(data, nothing)
    else
        grouped = _group_geometry(data, plan)
        rows = grouped.active_rows
        spans = _detector_energy_grid(grouped.low, grouped.high)
        target = pooling ? _coarsen_plot_grid(target, spans) : spans
    end
    source = _plot_model_flux(data.domain, model, extension)
    if pooling && isempty(rows)
        member = _empty_plot_member(data, target; unfold, has_model = true)
        return (; members = [member], target, source, summed = false)
    end
    geometry = _geometry(data, rows)
    fold = SF.objective_transformer(SF.ContiguouslyBinned(u"counts"), data)
    folded = vec(fold(data.domain, copy(source.values)))
    fit_rows = findall(data.data_mask)
    ordered_rows = geometry.ascending ? rows : reverse(rows)
    counts = collect(Float64, folded[searchsortedfirst.(Ref(fit_rows), ordered_rows)])
    member = _read_plot_member(data, rows, geometry; unfold, read_background = true, model = counts)
    if !pooling && !isnothing(target)
        member = _move_plot_member(member, target, :energy, overlap_projection(member.grid, target))
    end
    return (; members = [member], target, source, summed = false)
end

function _native_plot_inputs(input, model, extension; groupings, pooling, reference, unfold, read_background)
    datasets = if input isa GridMatch
        input.datasets
    elseif input isa SF.SpectralData
        (input,)
    elseif input isa Union{Tuple,AbstractVector}
        Tuple(input)
    else
        throw(ArgumentError("provide SpectralData, a grid match, a prepared dataset, or a fit result"))
    end
    !isempty(datasets) && all(data -> data isa SF.SpectralData, datasets) ||
        throw(ArgumentError("provide at least one SpectralData observation"))
    ref = isnothing(reference) ? (input isa GridMatch ? input.reference : 1) : reference
    1 <= ref <= length(datasets) || throw(ArgumentError("reference must index an observation"))
    plans = _plot_plans(groupings, length(datasets))
    shown = if pooling
        datasets
    else
        map(datasets, plans) do data, plan
            isnothing(plan) ? data : grouped_dataset(data, plan)
        end
    end
    raw_plans = pooling ? plans : ntuple(_ -> nothing, length(plans))
    has_model = !isnothing(model)
    match = if has_model
        edges = input isa GridMatch ? input.model_edges : nothing
        match_dataset_grids(shown; grid = edges, reference = ref)
    end
    source = has_model ? _plot_model_flux(match.model_edges, model, extension) : nothing
    members = _PlotMember[]
    target = if pooling
        input isa GridMatch ? input.detector_grids[ref] : _original_detector_grid(datasets[ref])
    end
    for i in eachindex(shown)
        data, plan = shown[i], raw_plans[i]
        if isnothing(plan)
            rows = _selected_rows(data, nothing)
            if pooling && isempty(rows)
                push!(members, _empty_plot_member(data, target; unfold, has_model))
                continue
            end
            geometry = _geometry(data, rows)
        else
            grouped = _group_geometry(data, plan)
            rows, geometry = grouped.active_rows, grouped.geometry
            if i == ref
                target = _coarsen_plot_grid(target, _detector_energy_grid(grouped.low, grouped.high))
            end
        end
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
    target = if !summed
        ref = isnothing(reference) ? data.reference : reference
        1 <= ref <= length(data.members) || throw(ArgumentError("reference must index an observation"))
        pooling ? _coarsen_plot_grid(data.detector_grids[ref], data.members[ref].grid) : nothing
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
