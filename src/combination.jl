abstract type AbstractGridDataset <: SF.AbstractDataset end

struct _MemberResponse
    grid::BinGrid
    projection::Vector{UnitRange{Int}}
    response::SparseMatrixCSC{Float64,Int}
    counts::Vector{Float64}
    variance::Vector{Float64}
    explicit_errors::Bool
    background::Union{Nothing,Vector{Float64}}
    alpha::Float64
end

"""Native detector spectra packed into one SpectralFitting fit objective."""
struct JointDataset <: AbstractGridDataset
    model_edges::Vector{Float64}
    members::Vector{_MemberResponse}
    member_ranges::Vector{UnitRange{Int}}
    counts::Vector{Float64}
    variance::Vector{Float64}
    name::String
end

"""Detector-bin sum with its overlap projections and full count covariance."""
struct SummedDataset <: AbstractGridDataset
    model_edges::Vector{Float64}
    members::Vector{_MemberResponse}
    grid::BinGrid
    coordinate::Symbol
    overlaps::Vector{SparseMatrixCSC{Float64,Int}}
    counts::Vector{Float64}
    variance::Vector{Float64}
    covariance::SparseMatrixCSC{Float64,Int}
    fractional::Bool
    whitener::Union{Nothing,LowerTriangular{Float64,SparseMatrixCSC{Float64,Int}}}
    name::String
end

function _in_counts(spectrum, rows, widths)
    scale, density = _counts_conversion(spectrum)
    factors = density ? Float64(scale) .* widths : fill(Float64(scale), length(rows))
    counts = Float64.(spectrum.data[rows]) .* factors
    all(isfinite, counts) || throw(ArgumentError("selected counts must be finite"))
    return counts, factors
end

function _count_variance(spectrum, rows, counts, factors)
    errors = spectrum.errors
    explicit = !isnothing(errors) && (any(<(0), counts) ||
        !(spectrum.error_statistics in (SF.ErrorStatistics.Poisson, SF.ErrorStatistics.Unknown)))
    variance = if explicit
        length(errors) == length(spectrum.data) ||
            throw(DimensionMismatch("statistical errors must match the spectrum"))
        all(x -> isfinite(x) && x >= 0, view(errors, rows)) ||
            throw(ArgumentError("statistical errors must be finite and nonnegative"))
        (errors[rows] .* factors) .^ 2
    else
        max.(counts, 1.0)
    end
    sys = spectrum.systematic_error
    isfinite(sys) && sys >= 0 || throw(ArgumentError("systematic error must be finite and nonnegative"))
    if !iszero(sys)
        variance .+= (sys .* counts) .^ 2
    end
    all(isfinite, variance) || throw(ArgumentError("count variances must be finite"))
    return variance, explicit || !iszero(sys)
end

function _effective_exposure(spectrum)
    exposure = Float64(spectrum.exposure_time) * Float64(spectrum.area_scale)
    isfinite(exposure) && exposure > 0 ||
        throw(ArgumentError("effective exposure must be positive and finite"))
    return exposure
end

function _background_counts(data, rows, widths, source_exposure)
    bkg = data.background
    isnothing(bkg) && return nothing, 0.0
    length(bkg.channels) == length(bkg.data) ||
        throw(DimensionMismatch("background channels and data differ in length"))
    index = _channel_index(bkg.channels, "background")
    wanted = data.spectrum.channels[rows]
    mapped = [get(index, channel, 0) for channel in wanted]
    all(>(0), mapped) || throw(ArgumentError("a selected source channel is missing from the background"))
    counts, _ = _in_counts(bkg, mapped, widths)
    source_backscale = Float64(data.spectrum.background_scale)
    background_backscale = Float64(bkg.background_scale)
    all(x -> isfinite(x) && x > 0, (source_backscale, background_backscale)) ||
        throw(ArgumentError("source and background BACKSCAL must be finite and positive"))
    scale = (source_exposure / _effective_exposure(bkg)) * (source_backscale / background_backscale)
    isfinite(scale) && scale > 0 || throw(ArgumentError("invalid source-to-background scaling"))
    return counts, scale
end

function _member_response(data, projection, coordinate, selected, geometry)
    rows = geometry.ascending ? selected : reverse(selected)
    grid, reversed = _detector_grid(geometry.low[rows], geometry.high[rows], coordinate)
    reversed && (rows = reverse(rows))
    widths = Float64.(geometry.high[rows]) .- Float64.(geometry.low[rows])
    counts, factors = _in_counts(data.spectrum, rows, widths)
    variance, explicit = _count_variance(data.spectrum, rows, counts, factors)
    exposure = _effective_exposure(data.spectrum)
    background, alpha = _background_counts(data, rows, widths, exposure)

    matrix = convert(SparseMatrixCSC{Float64,Int}, data.response.matrix[geometry.response_rows[rows], :])
    area = isnothing(data.ancillary) ? nothing : data.ancillary.effective_area
    values = nonzeros(matrix)
    for j in axes(matrix, 2)
        factor = isnothing(area) ? exposure : exposure * area[j]
        for i in nzrange(matrix, j)
            values[i] *= factor
        end
    end
    all(isfinite, values) || throw(ArgumentError("response entries must be finite after exposure and ARF scaling"))
    return _MemberResponse(grid, copy(projection), matrix, counts, variance, explicit, background, alpha)
end

function _matched_members(match, plans, coordinate)
    members = _MemberResponse[]
    for (data, native, projection, plan) in zip(match.datasets, match.native_grids, match.projections, plans)
        current = _incident_grid(data)
        current.low == native.low && current.high == native.high ||
            throw(ArgumentError("incident response boundaries changed; match the grids again"))
        if isnothing(plan)
            selected = _selected_rows(data, nothing)
            geometry = _geometry(data, selected)
        else
            grouped = _group_geometry(data, plan)
            selected, geometry = grouped.active_rows, grouped.geometry
        end
        push!(members, _member_response(data, projection, coordinate, selected, geometry))
    end
    return members
end

function _joint_dataset(match, plans, name)
    datasets = map(match.datasets, plans) do data, plan
        isnothing(plan) ? data : grouped_dataset(data, plan)
    end
    grouped_match = GridMatch(datasets, match.model_edges, match.native_grids, match.projections, match.reference)
    members = _matched_members(grouped_match, ntuple(_ -> nothing, length(plans)), :energy)
    offsets = cumsum(vcat(0, [length(member.counts) for member in members]))
    ranges = [(offsets[i] + 1):offsets[i + 1] for i in eachindex(members)]
    counts = reduce(vcat, (member.counts for member in members))
    variance = reduce(vcat, (member.variance for member in members))
    return JointDataset(copy(match.model_edges), members, ranges, counts, variance, name)
end

_raw_counts(values) = all(x -> x >= 0 && abs(x - round(x)) <= EDGE_RTOL * max(1.0, abs(x)), values)

function _summed_dataset(match, plans, coordinate, grid, name)
    all(data -> isnothing(data.background), match.datasets) ||
        throw(ArgumentError("sum mode requires background-free source data; use joint mode for measured backgrounds"))
    members = _matched_members(match, plans, coordinate)
    all(member -> _raw_counts(member.counts), members) ||
        throw(ArgumentError("sum mode requires nonnegative integer counts before rebinning"))

    target = if !isnothing(grid)
        _bin_grid(grid)
    elseif isnothing(plans[match.reference])
        _bin_grid(members[match.reference].grid)
    else
        grouped = _group_geometry(match.datasets[match.reference], plans[match.reference])
        first(_detector_grid(grouped.low, grouped.high, coordinate))
    end
    n = length(target)
    counts, covariance = zeros(n), spzeros(n, n)
    overlaps = [overlap_projection(member.grid, target) for member in members]
    fractional = any(p -> any(!=(1.0), nonzeros(p)), overlaps)
    for (overlap, member) in zip(overlaps, members)
        mul!(counts, overlap, round.(member.counts), 1, 1)
        covariance += overlap * spdiagm(0 => member.variance) * transpose(overlap)
    end
    variance = collect(diag(covariance))
    whitener = if !fractional
        nothing
    else
        factor = cholesky(Symmetric(covariance); perm = collect(1:n), check = false)
        if issuccess(factor)
            lower = sparse(factor.L)
            pivots = abs2.(diag(lower))
            all(pivots .> 64 * eps(Float64) .* variance) ? LowerTriangular(lower) : nothing
        end
    end
    return SummedDataset(copy(match.model_edges), members, target, coordinate, overlaps,
                         counts, variance, covariance, fractional, whitener, name)
end

"""
    combine_datasets(match; mode=:joint, groupings=nothing, grid=nothing,
                     coordinate=:wavelength, name="combined")

Prepare a fit on the native detector bins (`:joint`) or sum the spectra onto
one detector grid (`:sum`). In both cases a single source feeds each member's
response. Optional `groupings` contains one `GroupingPlan` or `nothing` per
member. In sum mode the reference plan defines the output bin boundaries and
the originals supply the detector counts and response rows for each overlap.
Use `grid` to override the summed detector grid, in Å by default or keV with
`coordinate=:energy`. The sum is for background-free raw counts.
"""
function combine_datasets(
    match::GridMatch; mode::Symbol = :joint, groupings = nothing, grid = nothing,
    coordinate::Symbol = :wavelength, name::AbstractString = "combined",
)
    n = length(match.datasets)
    plans = isnothing(groupings) ? ntuple(_ -> nothing, n) : Tuple(groupings)
    length(plans) == n && all(plan -> isnothing(plan) || plan isa GroupingPlan, plans) ||
        throw(ArgumentError("provide one GroupingPlan or nothing per matched dataset"))
    if mode == :joint
        isnothing(grid) || throw(ArgumentError("a detector target grid is only available in sum mode"))
        return _joint_dataset(match, plans, String(name))
    elseif mode == :sum
        coordinate in (:wavelength, :energy) || throw(ArgumentError("coordinate must be :wavelength or :energy"))
        return _summed_dataset(match, plans, coordinate, grid, String(name))
    end
    throw(ArgumentError("mode must be :joint or :sum"))
end

_native_flux(member, flux) = [sum(view(flux, bin)) for bin in member.projection]

function _fold_counts!(output, data::JointDataset, flux)
    for (member, range) in zip(data.members, data.member_ranges)
        mul!(view(output, range), member.response, _native_flux(member, flux))
    end
    return output
end

function _fold_counts!(output, data::SummedDataset, flux)
    fill!(output, zero(eltype(output)))
    for (member, overlap) in zip(data.members, data.overlaps)
        native = member.response * _native_flux(member, flux)
        mul!(output, overlap, native, 1, 1)
    end
    return output
end

"""
    predict_counts(data, incident_flux)

Fold one vector of bin-integrated source values on `data.model_edges` into the
same detector bins as `data.counts`. Joint predictions are packed in member
order and `member_ranges` selects an individual spectrum.
"""
function predict_counts(data::AbstractGridDataset, incident_flux::AbstractVector)
    Base.require_one_based_indexing(incident_flux)
    length(incident_flux) == length(data.model_edges) - 1 ||
        throw(DimensionMismatch("incident flux must have one value per shared model bin"))
    output = zeros(promote_type(Float64, eltype(incident_flux)), length(data.counts))
    return _fold_counts!(output, data, incident_flux)
end

SF.supports(::Type{<:AbstractGridDataset}) = (SF.ContiguouslyBinned(),)
SF.make_label(data::AbstractGridDataset) = data.name
SF._printinfo(io::IO, data::AbstractGridDataset) =
    print(io, data.name, " with ", length(data.counts), " bins from ", length(data.members), " datasets")
SF.make_model_domain(::SF.ContiguouslyBinned, data::AbstractGridDataset) = copy(data.model_edges)
SF.make_output_domain(::SF.ContiguouslyBinned, data::AbstractGridDataset) = collect(0.0:length(data.counts))

_whitened(layout) = SF.support_units(layout) === :whitened_counts

function SF.make_objective(layout::SF.ContiguouslyBinned, data::AbstractGridDataset)
    return _whitened(layout) ? ldiv!(data.whitener, copy(data.counts)) : data.counts
end

SF.make_objective_variance(layout::SF.ContiguouslyBinned, data::AbstractGridDataset) =
    _whitened(layout) ? ones(length(data.counts)) : data.variance

function SF.objective_transformer(layout::SF.ContiguouslyBinned, data::AbstractGridDataset)
    function fold!(output, domain, flux)
        domain == data.model_edges || throw(ArgumentError("incident model grid differs from the matched grid"))
        counts = _fold_counts!(vec(output), data, vec(flux))
        _whitened(layout) && ldiv!(data.whitener, counts)
        return output
    end
    function fold!(domain, flux)
        output = similar(vec(flux), promote_type(Float64, eltype(flux)), length(data.counts))
        return fold!(output, domain, flux)
    end
    return fold!
end
