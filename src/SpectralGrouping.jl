module SpectralGrouping

using LinearAlgebra: Symmetric, LowerTriangular, cholesky, diag, dot, issuccess, ldiv!, mul!
using SparseArrays: SparseMatrixCSC, nonzeros, nzrange, rowvals, sparse, spdiagm, spzeros
import SpectralFitting as SF
import FITSIO
using SpectralFitting: SpectralUnits, @u_str

export GroupingPlan, grouping_plan, group_sum, grouped_dataset
export group_counts, group_width, group_snr, group_ogip
export BinGrid, GridMatch, JointDataset, SummedDataset
export match_dataset_grids, combine_datasets, overlap_projection, predict_counts, grid_statistic
export with_channel_scalings
export DetectorAlignment, detector_grid, align_detector_grids, rebin_detector_grid
export PlotCurve, SpectrumPlot, plot_curves, plot_data, plot_counts, plot_fit, plot_unfold

include("channel-scalings.jl")

"""
    GroupingPlan

Detector-channel groups in increasing energy order. Each range in `groups`
indexes the original spectrum in its original row order. `met_target` marks
groups meeting the requested counts, width, or S/N target, and is all true for
file grouping. `target_width` records the requested channel count for a
`:width` plan. Construct plans with `grouping_plan`.
"""
struct GroupingPlan
    channels::Vector{Int}
    groups::Vector{UnitRange{Int}}
    met_target::BitVector
    target_width::Union{Nothing,Int}
end

GroupingPlan(channels, groups, met_target) = GroupingPlan(channels, groups, met_target, nothing)

function _selected_rows(data::SF.SpectralData, mask)
    n = length(data.spectrum.channels)
    length(data.spectrum.data) == length(data.spectrum.quality) ==
        length(data.data_mask) == n || throw(DimensionMismatch("source channel arrays differ in length"))
    selected = data.data_mask .& (data.spectrum.quality .== 0)
    if !isnothing(mask)
        Base.require_one_based_indexing(mask)
        length(mask) == n || throw(DimensionMismatch("mask must have one value per source channel"))
        selected .&= mask
    end
    return findall(selected)
end

function _channel_index(channels, label)
    index = Dict(channel => row for (row, channel) in enumerate(channels))
    length(index) == length(channels) || throw(ArgumentError("duplicate $label channel labels"))
    return index
end

function _geometry(data::_DetectorData, rows)
    isempty(rows) && throw(ArgumentError("no noticed, good-quality source channels"))
    channels = data.spectrum.channels
    allunique(channels) || throw(ArgumentError("duplicate source channel labels"))
    response = data.response
    n = length(response.channels)
    length(response.channel_bins_low) == length(response.channel_bins_high) ==
        size(response.matrix, 1) == n || throw(DimensionMismatch("response channel arrays differ in length"))
    index = _channel_index(response.channels, "response")
    response_rows = zeros(Int, length(channels))
    low = similar(response.channel_bins_low, length(channels))
    high = similar(low)
    for row in rows
        channel = channels[row]
        response_row = get(index, channel, 0)
        response_row == 0 && throw(ArgumentError("source channel $channel is missing from the response"))
        response_rows[row] = response_row
        low[row] = response.channel_bins_low[response_row]
        high[row] = response.channel_bins_high[response_row]
        isfinite(low[row]) && isfinite(high[row]) && low[row] < high[row] ||
            throw(ArgumentError("invalid EBOUNDS for source channel $channel"))
    end

    ascending = low[first(rows)] <= low[last(rows)]
    for k in 2:length(rows)
        a, b = rows[k - 1], rows[k]
        left, right = ascending ? (a, b) : (b, a)
        low[left] < low[right] && (high[left] <= low[right] || _same_edge(high[left], low[right])) ||
            throw(ArgumentError("selected EBOUNDS must be ordered and nonoverlapping"))
    end
    return (; response_rows, low, high, ascending)
end

# FITS EBOUNDS can be written at single precision and loaded as Float64.
_same_edge(a, b) = isapprox(a, b; rtol = 4 * eps(Float32))

function _adjacent(channels, geometry, a, b)
    b == a + 1 && abs(channels[a] - channels[b]) == 1 || return false
    left, right = geometry.ascending ? (a, b) : (b, a)
    return _same_edge(geometry.high[left], geometry.low[right])
end

function _counts_conversion(spectrum::SF.Spectrum)
    units = spectrum.units
    if units == u"counts" || units == u"counts / keV"
        return one(spectrum.exposure_time), units == u"counts / keV"
    elseif units == u"counts / s" || units == u"counts / (s * keV)"
        exposure = spectrum.exposure_time
        isfinite(exposure) && exposure > 0 || throw(ArgumentError("count rates require a positive, finite exposure"))
        return exposure, units == u"counts / (s * keV)"
    end
    throw(ArgumentError("unsupported spectrum units: $units"))
end

"""
    grouping_plan(data, method, [target]; mask=nothing, max_channels=nothing, errors=nothing)

Build a grouping plan on the detector grid of a SpectralFitting `SpectralData`
or a spectrum returned by `with_channel_scalings`.
Use `:counts` for a minimum number of observed counts, `:width`
for a fixed number of channels, `:snr` for a minimum signal-to-noise ratio, or
`:ogip` to apply the source file's GROUPING flags.

An optional Boolean `mask` further restricts `data.data_mask`. Groups stop at
excluded rows, missing channel labels, and gaps in the response EBOUNDS.
Short groups are retained and identified by `met_target`. `max_channels`
caps group size for `:counts` and `:snr`. For `:snr`, the default is Poisson
noise, and `errors` can supply independent one-sigma errors in the spectrum's
current units.
"""
function grouping_plan(
    data::_DetectorData, method::Symbol, target = nothing;
    mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    max_channels::Union{Nothing, Integer} = nothing,
    errors::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    method in (:counts, :width, :snr, :ogip) || throw(ArgumentError("unknown grouping method: $method"))
    if method == :ogip
        isnothing(target) || throw(ArgumentError("OGIP grouping does not take a target"))
    elseif method == :width
        target isa Integer && target > 0 || throw(ArgumentError("width must be a positive integer"))
    else
        target isa Real && isfinite(target) && target > 0 ||
            throw(ArgumentError("target must be positive and finite"))
    end
    if !isnothing(max_channels)
        max_channels > 0 && method in (:counts, :snr) ||
            throw(ArgumentError("max_channels must be positive and is only available for counts and S/N grouping"))
    end
    isnothing(errors) || method == :snr || throw(ArgumentError("errors is only available for S/N grouping"))

    rows = _selected_rows(data, mask)
    geometry = _geometry(data, rows)
    spectrum = data.spectrum
    channels = spectrum.channels
    flags = spectrum.grouping
    n = length(channels)
    cap = isnothing(max_channels) ? n : Int(min(max_channels, n))
    if method == :ogip
        length(flags) == n || throw(DimensionMismatch("GROUPING must have one flag per source channel"))
        all(flag -> flag in (-1, 0, 1), flags) || throw(ArgumentError("invalid OGIP GROUPING flag"))
    end
    if !isnothing(errors)
        Base.require_one_based_indexing(errors)
        length(errors) == n || throw(DimensionMismatch("errors must have one value per source channel"))
    end

    counts = zeros(Float64, n)
    variances = isnothing(errors) ? counts : similar(counts)
    if method in (:counts, :snr)
        scale, density = _counts_conversion(spectrum)
        for row in rows
            factor = density ? scale * (geometry.high[row] - geometry.low[row]) : scale
            count = spectrum.data[row] * factor
            isfinite(count) && count >= 0 ||
                throw(ArgumentError("counts and S/N grouping require finite, nonnegative source counts"))
            counts[row] = count
            if !isnothing(errors)
                sigma = errors[row] * factor
                isfinite(sigma^2) && sigma >= 0 || throw(ArgumentError("invalid S/N error at source row $row"))
                variances[row] = sigma^2
            end
        end
    end

    groups = UnitRange{Int}[]
    met = BitVector()
    start = first(rows)
    total_counts = 0.0
    total_variance = 0.0
    for (k, row) in enumerate(rows)
        total_counts += counts[row]
        total_variance += variances[row]
        width = row - start + 1
        at_break = k == length(rows) || !_adjacent(channels, geometry, row, rows[k + 1])
        reached = if method == :counts
            total_counts >= target
        elseif method == :snr
            total_variance > 0 && total_counts / sqrt(total_variance) >= target
        elseif method == :width
            width == target
        else
            at_break || flags[row] == 0 || flags[rows[k + 1]] != -1
        end

        if reached || at_break || width == cap
            push!(groups, start:row)
            push!(met, reached)
            if k < length(rows)
                start = rows[k + 1]
            end
            total_counts = 0.0
            total_variance = 0.0
        end
    end
    if !geometry.ascending
        reverse!(groups)
        reverse!(met)
    end
    target_width = method == :width ? Int(min(target, typemax(Int))) : nothing
    return GroupingPlan(copy(channels), groups, met, target_width)
end

"""
    group_sum(plan, values)

Sum a vector of integrated values on the original source channel grid. This
also applies the plan to folded model counts and other channel-aligned vectors.
"""
function group_sum(plan::GroupingPlan, values::AbstractVector)
    Base.require_one_based_indexing(values)
    length(values) == length(plan.channels) ||
        throw(DimensionMismatch("values must have one entry per original source channel"))
    return [sum(view(values, group)) for group in plan.groups]
end

function _group_spectrum(
    spectrum::SF.Spectrum{T}, groups, spectrum_rows, geometry, low, high,
) where {T}
    n = length(spectrum.channels)
    errors = spectrum.errors
    length(spectrum.data) == length(spectrum.quality) == n &&
        (isnothing(errors) || length(errors) == n) ||
        throw(DimensionMismatch("spectrum channel arrays differ in length"))
    exposure_factor, density = _counts_conversion(spectrum)
    counts_conversion = density ? exposure_factor .* (high .- low) : fill(exposure_factor, length(groups))
    output = zeros(T, length(groups))
    quality = zeros(Int, length(groups))
    output_errors = isnothing(errors) ? nothing : similar(output)
    poisson = spectrum.error_statistics == SF.ErrorStatistics.Poisson
    for (i, group) in enumerate(groups)
        counts = zero(promote_type(T, Float64))
        variance = zero(counts)
        for source_row in group
            row = spectrum_rows[source_row]
            factor = density ?
                exposure_factor * (geometry.high[source_row] - geometry.low[source_row]) : exposure_factor
            counts += spectrum.data[row] * factor
            quality[i] = max(quality[i], spectrum.quality[row])
            if !isnothing(errors) && !poisson
                variance += (errors[row] * factor)^2
            end
        end
        output[i] = counts / counts_conversion[i]
        if !isnothing(output_errors)
            if poisson
                isfinite(counts) && counts >= 0 ||
                    throw(ArgumentError("Poisson error propagation requires finite, nonnegative grouped counts"))
                output_errors[i] = SF.count_error(counts, 1.0) / counts_conversion[i]
            else
                output_errors[i] = sqrt(variance) / counts_conversion[i]
            end
        end
    end
    return SF.Spectrum{T}(
        collect(1:length(groups)), quality, ones(Int, length(groups)), output,
        spectrum.units, spectrum.exposure_time, spectrum.background_scale, spectrum.area_scale,
        spectrum.error_statistics, output_errors, spectrum.systematic_error,
        spectrum.telescope_name, spectrum.instrument,
    )
end

function _group_geometry(data::_DetectorData, plan::GroupingPlan)
    channels = data.spectrum.channels
    channels == plan.channels || throw(ArgumentError("plan was built for a different set of source channels"))
    groups = plan.groups
    length(groups) == length(plan.met_target) && all(group -> !isempty(group), groups) ||
        throw(ArgumentError("plan must have one target flag per nonempty group"))
    active_rows = sort!(collect(Iterators.flatten(groups)))
    selected = _selected_rows(data, nothing)
    active_rows == intersect(selected, active_rows) ||
        throw(ArgumentError("plan contains overlapping, excluded, or invalid channel rows"))
    geometry = _geometry(data, active_rows)
    for group in groups
        all(row -> _adjacent(channels, geometry, row, row + 1), first(group):(last(group) - 1)) ||
            throw(ArgumentError("plan crosses a gap in the current detector grid"))
    end

    starts = geometry.ascending ? first.(groups) : last.(groups)
    ends = geometry.ascending ? last.(groups) : first.(groups)
    low, high = geometry.low[starts], geometry.high[ends]
    issorted(low) || throw(ArgumentError("plan groups must follow increasing detector energy"))
    return (; active_rows, geometry, low, high)
end

function _background_rows(data, rows)
    background = data.background
    length(background.channels) == length(background.data) ||
        throw(DimensionMismatch("background channels and data differ in length"))
    index = _channel_index(background.channels, "background")
    mapped = [get(index, data.spectrum.channels[row], 0) for row in rows]
    all(>(0), mapped) ||
        throw(ArgumentError("a selected source channel is missing from the background"))
    return mapped
end

"""
    grouped_dataset(data, plan)

Return a fresh SpectralFitting dataset with the plan applied to the source,
background, and response rows. The incident model grid and ARF are carried
through. The output's energy_low and energy_high arrays retain detector gaps.
For a channel-scaled input, return a fresh calibrated dataset with the native
response factors applied before summing the rows.
"""
function grouped_dataset(data::SF.SpectralData, plan::GroupingPlan)
    grouped = _group_geometry(data, plan)
    mapped = isnothing(data.background) ? nothing : _background_rows(data, grouped.active_rows)
    return _grouped_spectral_data(data, plan, grouped, mapped)
end

function _grouped_spectral_data(data::SF.SpectralData{T}, plan, grouped, mapped;
                                response_weights = nothing) where {T}
    channels, groups = data.spectrum.channels, plan.groups
    (; active_rows, geometry, low, high) = grouped
    n = length(groups)
    output_rows = Int[]
    response_rows = Int[]
    for (i, group) in enumerate(groups), row in group
        push!(output_rows, i)
        push!(response_rows, geometry.response_rows[row])
    end
    response = data.response
    weights = isnothing(response_weights) ? ones(T, length(active_rows)) : T.(response_weights)
    projection = sparse(output_rows, response_rows, weights, n, size(response.matrix, 1))
    grouped_response = SF.ResponseMatrix(
        projection * response.matrix, collect(1:n), copy(low), copy(high),
        copy(response.bins_low), copy(response.bins_high),
    )
    spectrum = _group_spectrum(data.spectrum, groups, eachindex(channels), geometry, low, high)
    background = if !isnothing(mapped)
        background_rows = zeros(Int, length(channels))
        background_rows[active_rows] = mapped
        _group_spectrum(data.background, groups, background_rows, geometry, low, high)
    end
    return typeof(data)(
        spectrum, grouped_response, background, deepcopy(data.ancillary),
        low, high, copy(data.domain), trues(n), data.user_data,
    )
end

"""Return a dataset grouped to a minimum number of observed counts per bin."""
group_counts(data::_DetectorData, minimum; kwargs...) =
    grouped_dataset(data, grouping_plan(data, :counts, minimum; kwargs...))

"""Return a dataset grouped into bins of at most `width` detector channels."""
group_width(data::_DetectorData, width; kwargs...) =
    grouped_dataset(data, grouping_plan(data, :width, width; kwargs...))

"""Return a dataset grouped to a minimum signal-to-noise ratio per bin."""
group_snr(data::_DetectorData, minimum; kwargs...) =
    grouped_dataset(data, grouping_plan(data, :snr, minimum; kwargs...))

"""Return a dataset grouped using the original source's OGIP GROUPING flags."""
group_ogip(data::_DetectorData; kwargs...) =
    grouped_dataset(data, grouping_plan(data, :ogip; kwargs...))

include("grids.jl")
include("combination.jl")
include("grid-statistics.jl")
include("plot-inputs.jl")
include("plot-residuals.jl")
include("plotting.jl")

end
