"""
    BinGrid(low, high)
    BinGrid(edges)

Increasing, nonoverlapping bins. Separate lower and upper boundaries allow
gaps. Use keV for incident grids and the chosen detector coordinate for sums.
"""
struct BinGrid
    low::Vector{Float64}
    high::Vector{Float64}

    function BinGrid(low::AbstractVector{<:Real}, high::AbstractVector{<:Real})
        lo, hi = collect(Float64, low), collect(Float64, high)
        length(lo) == length(hi) && !isempty(lo) ||
            throw(ArgumentError("provide equally sized, nonempty lower and upper bin edges"))
        for i in eachindex(lo)
            isfinite(lo[i]) && isfinite(hi[i]) && 0 <= lo[i] < hi[i] ||
                throw(ArgumentError("bin edges must be finite and satisfy 0 ≤ low < high"))
            i == 1 || hi[i - 1] <= lo[i] || throw(ArgumentError("bins must increase without overlap"))
        end
        return new(lo, hi)
    end
end

function BinGrid(edges::AbstractVector{<:Real})
    length(edges) >= 2 || throw(ArgumentError("an edge grid requires at least two edges"))
    return BinGrid(edges[begin:(end - 1)], edges[(begin + 1):end])
end

Base.length(grid::BinGrid) = length(grid.low)
_bin_grid(grid::BinGrid) = BinGrid(grid.low, grid.high)
_bin_grid(edges::AbstractVector{<:Real}) = BinGrid(edges)

const HC_KEV_ANGSTROM = 12.398419843320026
const EDGE_RTOL = 16 * eps(Float64)

_roundtrip_edge(a, b) = abs(a - b) <= EDGE_RTOL * max(abs(a), abs(b))

function _edge_index(edges, value)
    i = searchsortedfirst(edges, value)
    i = min(i, length(edges))
    if i > 1 && abs(edges[i - 1] - value) < abs(edges[i] - value)
        i -= 1
    end
    _roundtrip_edge(edges[i], value) ||
        throw(ArgumentError("incident edge $value is absent from the shared model grid"))
    return i
end

function _incident_grid(data::SF.SpectralData)
    rmf = data.response
    grid = BinGrid(rmf.bins_low, rmf.bins_high)
    size(rmf.matrix, 2) == length(grid) ||
        throw(DimensionMismatch("response columns must match the incident grid"))
    arf = data.ancillary
    if !isnothing(arf)
        length(arf.effective_area) == length(arf.bins_low) == length(arf.bins_high) == length(grid) ||
            throw(DimensionMismatch("ARF and RMF must have the same number of incident bins"))
        all(_roundtrip_edge.(grid.low, arf.bins_low)) && all(_roundtrip_edge.(grid.high, arf.bins_high)) ||
            throw(ArgumentError("ARF and RMF incident-energy boundaries differ within a dataset"))
        all(x -> isfinite(x) && x >= 0, arf.effective_area) ||
            throw(ArgumentError("ARF effective areas must be finite and nonnegative"))
    end
    return grid
end

function _detector_energy_grid(low, high)
    length(low) == length(high) || throw(DimensionMismatch("detector boundary arrays differ in length"))
    lo = collect(Float64, low)
    for i in 2:length(lo)
        # FITS bounds can differ by roundoff at a nominally shared edge.
        if high[i - 1] > lo[i] && _roundtrip_edge(high[i - 1], lo[i])
            lo[i] = high[i - 1]
        end
    end
    return BinGrid(lo, high)
end

function _original_detector_grid(data::SF.SpectralData)
    response = data.response
    low, high = response.channel_bins_low, response.channel_bins_high
    length(low) == length(high) == size(response.matrix, 1) == length(response.channels) ||
        throw(DimensionMismatch("response channel arrays differ in length"))
    allunique(response.channels) || throw(ArgumentError("duplicate response channel labels"))
    order = sortperm(low)
    return _detector_energy_grid(low[order], high[order])
end

"""Matched incident grids and original RMF detector grids, all in keV."""
struct GridMatch{D<:Tuple}
    datasets::D
    model_edges::Vector{Float64}
    native_grids::Vector{BinGrid}
    projections::Vector{Vector{UnitRange{Int}}}
    reference::Int
    detector_grids::Vector{BinGrid}
end

function GridMatch{D}(datasets, model_edges, native_grids, projections, reference) where {D<:Tuple}
    return GridMatch{D}(datasets, model_edges, native_grids, projections, reference,
                        BinGrid[_original_detector_grid(data) for data in datasets])
end

GridMatch(datasets::D, model_edges, native_grids, projections, reference) where {D<:Tuple} =
    GridMatch{D}(datasets, model_edges, native_grids, projections, reference)

"""
    match_dataset_grids(datasets; grid=nothing, reference=1)

Build a common incident model grid in keV. By default it contains every native
RMF input edge. A supplied contiguous grid must include all these boundaries.
`reference` indexes the member that supplies the default detector target.
"""
function match_dataset_grids(datasets::Union{Tuple, AbstractVector}; grid = nothing, reference::Integer = 1)
    members = Tuple(datasets)
    !isempty(members) && all(data -> data isa SF.SpectralData, members) ||
        throw(ArgumentError("provide one or more SpectralFitting SpectralData objects"))
    length(IdDict(data => nothing for data in members)) == length(members) ||
        throw(ArgumentError("a dataset cannot appear twice in a match"))
    1 <= reference <= length(members) || throw(ArgumentError("reference must index a matched dataset"))
    native = BinGrid[_incident_grid(data) for data in members]
    edges = if isnothing(grid)
        sort!(unique!(reduce(vcat, (vcat(g.low, g.high) for g in native))))
    else
        supplied = _bin_grid(grid)
        supplied.low[2:end] == supplied.high[1:(end - 1)] ||
            throw(ArgumentError("the shared model grid must be contiguous"))
        vcat(supplied.low, last(supplied.high))
    end
    projections = map(native) do g
        starts = [_edge_index(edges, edge) for edge in g.low]
        stops = [_edge_index(edges, edge) - 1 for edge in g.high]
        all(starts .<= stops) || throw(ArgumentError("a native incident bin collapsed on the shared grid"))
        [start:stop for (start, stop) in zip(starts, stops)]
    end
    return GridMatch(members, edges, native, projections, Int(reference))
end

match_dataset_grids(first::SF.SpectralData, rest::SF.SpectralData...; kwargs...) =
    match_dataset_grids((first, rest...); kwargs...)

"""
    overlap_projection(source, target)

Sparse map from source-bin integrals to target-bin integrals, assuming constant
density within each source bin. Bins and overlaps use the same coordinate.
The map crops at target boundaries and leaves gaps without extrapolation.
"""
function overlap_projection(source::BinGrid, target::BinGrid)
    rows, cols, weights = Int[], Int[], Float64[]
    i, j = 1, 1
    while i <= length(source) && j <= length(target)
        lo = max(source.low[i], target.low[j])
        hi = min(source.high[i], target.high[j])
        if hi > lo && !_roundtrip_edge(lo, hi)
            width = source.high[i] - source.low[i]
            weight = _roundtrip_edge(lo, source.low[i]) && _roundtrip_edge(hi, source.high[i]) ?
                1.0 : (hi - lo) / width
            push!(rows, j)
            push!(cols, i)
            push!(weights, weight)
        end
        if source.high[i] <= target.high[j]
            i += 1
        else
            j += 1
        end
    end
    return sparse(rows, cols, weights, length(target), length(source))
end

function _detector_grid(low, high, coordinate)
    energy = _detector_energy_grid(low, high)
    if coordinate == :energy
        return energy, false
    end
    all(>(0), energy.low) || throw(ArgumentError("wavelength matching requires positive EBOUNDS; use coordinate=:energy"))
    return BinGrid(reverse(HC_KEV_ANGSTROM ./ energy.high), reverse(HC_KEV_ANGSTROM ./ energy.low)), true
end
