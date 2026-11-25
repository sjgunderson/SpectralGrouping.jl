# The observed vector identifies the data in both SpectralFitting 0.7.x
# measure call orders. Its general optimizer passes model then data, while
# finalization and result diagnostics pass data then model.
struct _GridPoisson <: SF.AbstractStatistic
    observed::Vector{Float64}
    background::Union{Nothing,Vector{Float64}}
    alpha::Union{Nothing,Vector{Float64}}
end

SF.statistic_symbol(stat::_GridPoisson) = isnothing(stat.background) ? "C-stat" : "W-stat"

struct _JointWStat <: SF.AbstractStatistic
    by_observation::IdDict{Vector{Float64},_GridPoisson}
end

SF.statistic_symbol(::_JointWStat) = "W-stat"

_has_background(data::JointDataset) = any(member -> !isnothing(member.background), data.members)
_has_explicit_errors(data::AbstractGridDataset) = any(member -> member.explicit_errors, data.members)

"""
    grid_statistic(data)

Choose a statistic for the prepared dataset. Uses a profiled Poisson
likelihood for joint data with measured backgrounds, Cash deviance for
background-free integer Poisson counts, and chi-squared for supplied errors,
weighted joint counts or fractional sums. Chi-squared on a fractional sum
uses the full propagated covariance.
"""
function grid_statistic(data::JointDataset)
    if _has_background(data)
        _raw_counts(data.counts) || throw(ArgumentError("W-stat requires raw integer source counts"))
        background, alpha = zeros(length(data.counts)), zeros(length(data.counts))
        for (member, rows) in zip(data.members, data.member_ranges)
            if !isnothing(member.background)
                _raw_counts(member.background) ||
                    throw(ArgumentError("W-stat requires raw integer background counts"))
                background[rows] .= member.background
                alpha[rows] .= member.alpha
            end
        end
        return _GridPoisson(data.counts, background, alpha)
    end
    return _has_explicit_errors(data) || !_raw_counts(data.counts) ?
        SF.ChiSquared() : _GridPoisson(data.counts, nothing, nothing)
end

function grid_statistic(data::SummedDataset)
    return data.fractional || _has_explicit_errors(data) ?
        SF.ChiSquared() : _GridPoisson(data.counts, nothing, nothing)
end

"""
    grid_statistic(joint1, joint2, rest...)

Use one W-stat for a SpectralFitting problem with multiple prepared
`JointDataset` objects. Each dataset must have raw on/off counts and select
W-stat on its own. Each model/data pair keeps its incident grid and prepared
likelihood bins. Bind shared source parameters in the fitting problem before
creating its configuration.
"""
function grid_statistic(first::AbstractGridDataset, second::AbstractGridDataset, rest::AbstractGridDataset...)
    by_observation = IdDict{Vector{Float64},_GridPoisson}()
    for data in (first, second, rest...)
        data isa JointDataset ||
            throw(ArgumentError("multi-dataset W-stat requires prepared JointDataset inputs"))
        haskey(by_observation, data.counts) &&
            throw(ArgumentError("a dataset cannot be registered twice for multi-dataset W-stat"))
        stat = grid_statistic(data)
        stat isa _GridPoisson && !isnothing(stat.background) ||
            throw(ArgumentError("multi-dataset W-stat requires measured backgrounds in every joint dataset"))
        by_observation[data.counts] = stat
    end
    return _JointWStat(by_observation)
end

function SF.preferred_units(data::AbstractGridDataset, stat::_JointWStat)
    member_stat = get(stat.by_observation, data.counts, nothing)
    data isa JointDataset && !isnothing(member_stat) ||
        throw(ArgumentError("register this joint dataset with grid_statistic before fitting"))
    return SF.preferred_units(data, member_stat)
end

SF.preferred_units(::SF.AbstractDataset, ::_JointWStat) =
    throw(ArgumentError("multi-dataset W-stat requires registered JointDataset inputs"))

function SF.measure(stat::_JointWStat, first, second, variance)
    # The optimizer and result APIs pass the observed vector on opposite sides.
    left = get(stat.by_observation, first, nothing)
    right = get(stat.by_observation, second, nothing)
    if !isnothing(left) && !isnothing(right) && left !== right
        throw(ArgumentError("W-stat received two different registered observation vectors"))
    end
    member_stat = isnothing(left) ? right : left
    isnothing(member_stat) &&
        throw(ArgumentError("W-stat requires a registered joint dataset's observed count vector"))
    return SF.measure(member_stat, first, second, variance)
end

function SF.preferred_units(data::AbstractGridDataset, stat::SF.AbstractStatistic)
    if stat isa SF.Cash
        throw(ArgumentError("use stat=grid_statistic(data) for Poisson fits on a matched dataset"))
    elseif stat isa _GridPoisson
        stat.observed === data.counts ||
            throw(ArgumentError("build the Poisson statistic from this same dataset with grid_statistic"))
    elseif data isa JointDataset && _has_background(data)
        throw(ArgumentError("joint data with measured backgrounds require stat=grid_statistic(data)"))
    elseif data isa SummedDataset && data.fractional
        stat isa SF.ChiSquared ||
            throw(ArgumentError("fractional counts require covariance-aware ChiSquared()"))
        isnothing(data.whitener) &&
            throw(ArgumentError("the summed covariance is singular; use joint mode or a coarser detector grid"))
        return :whitened_counts
    elseif stat isa SF.ChiSquared
        all(>(0), data.variance) ||
            throw(ArgumentError("chi-squared requires positive variance in every fitted bin"))
    end
    return nothing  # The fit objective is in integrated counts.
end

function _poisson_deviance(observed, expected)
    isfinite(expected) && expected >= 0 || return Inf
    iszero(observed) && return 2 * expected
    iszero(expected) && return Inf
    delta = expected - observed
    if abs(delta) <= 0.5 * observed
        relative = delta / observed
        return 2 * observed * (relative - log1p(relative))
    end
    return 2 * (delta + observed * (log(observed) - log(expected)))
end

function _profile_deviance(on, source, off, alpha)
    isfinite(source) && source >= 0 || return Inf
    k = 1 + inv(alpha)
    on_background = if iszero(off)
        max(zero(source), on / k - source)
    else
        scaled = k * source
        z = on + off - scaled
        root = sqrt(z * z + 4 * off * scaled)
        z >= 0 ? (z + root) / (2 * k) : 2 * off * source / (root - z)
    end
    return _poisson_deviance(on, source + on_background) +
        _poisson_deviance(off, on_background / alpha)
end

function SF.measure(stat::_GridPoisson, first, second, _)
    predicted = if first === stat.observed
        second
    elseif second === stat.observed
        first
    else
        throw(ArgumentError("the Poisson statistic must be used with its prepared dataset"))
    end
    observed = stat.observed
    length(predicted) == length(observed) || throw(DimensionMismatch("predictions must match the count bins"))
    total = zero(eltype(predicted))
    for i in eachindex(observed)
        total += if isnothing(stat.alpha) || iszero(stat.alpha[i])
            _poisson_deviance(observed[i], predicted[i])
        else
            _profile_deviance(observed[i], predicted[i], stat.background[i], stat.alpha[i])
        end
    end
    return total
end
