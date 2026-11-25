struct _PHAScale
    values::Vector{Float64}
    low::Vector{Float64}
    high::Vector{Float64}
end

_PHAScale(values) = _PHAScale(values, values, values)

struct _PHAScalings
    channels::Vector{Int}
    area_scale::_PHAScale
    background_scale::_PHAScale
end

struct _ScaledBackground
    counts::Vector{Float64}
    variance::Vector{Float64}
end

struct _ChannelScaledData{D<:SF.SpectralData}
    data::D
    source_scalings::_PHAScalings
    background_scalings::Union{Nothing,_PHAScalings}
    background_display::Union{Nothing,_ScaledBackground}
end

const _DetectorData = Union{SF.SpectralData,_ChannelScaledData}

_loaded_data(data::SF.SpectralData) = data
_loaded_data(data::_ChannelScaledData) = getfield(data, :data)

function Base.getproperty(data::_ChannelScaledData, key::Symbol)
    if key in (:data, :source_scalings, :background_scalings, :background_display)
        return getfield(data, key)
    end
    return getproperty(_loaded_data(data), key)
end

SF.make_label(data::_ChannelScaledData) = SF.make_label(data.data)

function SF.mask_energies!(data::_ChannelScaledData, args...)
    SF.mask_energies!(_loaded_data(data), args...)
    return data
end

function SF.restrict_domain!(data::_ChannelScaledData, args...)
    SF.restrict_domain!(_loaded_data(data), args...)
    return data
end

SF.regroup!(::_ChannelScaledData, args...; kwargs...) =
    throw(ArgumentError("use SpectralGrouping.grouped_dataset or a SpectralGrouping group_* helper for channel-scaled data"))

function _pha_column(hdu, name, n, path)
    values = read(hdu, name)
    values isa AbstractVector{<:Real} && length(values) == n ||
        throw(ArgumentError("$name must contain one numeric value per Type-I channel in $path"))
    return values
end

function _check_pha_counts(data, spectrum, hdu, header, names, rows, n, path)
    has_rates = "RATE" in names
    column = has_rates ? "RATE" : "COUNTS"
    column in names || throw(ArgumentError("expected a RAW COUNTS or RATE column in $path"))
    observed = _pha_column(hdu, column, n, path)
    exposure = get(header, "EXPOSURE", 0.0)
    if exposure != 0 && !isapprox(exposure, spectrum.exposure_time; rtol = 8eps(Float32))
        throw(ArgumentError("PHA exposure differs from the loaded spectrum in $path"))
    end
    file_factor = has_rates ? Float64(spectrum.exposure_time) : 1.0
    factor, density = _counts_conversion(spectrum)
    rmf = data.response
    rmf_rows = density ? _channel_index(rmf.channels, "response") : nothing
    for i in eachindex(rows)
        width = 1.0
        if density
            r = get(rmf_rows, spectrum.channels[i], 0)
            # Off-region channels outside the source response cannot contribute.
            r == 0 && continue
            width = rmf.channel_bins_high[r] - rmf.channel_bins_low[r]
        end
        loaded = Float64(spectrum.data[i]) * factor * width
        original = Float64(observed[rows[i]]) * file_factor
        isequal(loaded, original) || isapprox(loaded, original; rtol = 8eps(Float32)) ||
            throw(ArgumentError("PHA counts differ on channel $(spectrum.channels[i]) in $path; use the original ungrouped, unsubtracted spectrum"))
    end
    return nothing
end

function _read_pha_scalings(path, data, spectrum, is_background)
    fits = FITSIO.FITS(String(path), "r")
    try
        candidates = [fits[i] for i in 1:length(fits) if fits[i] isa FITSIO.TableHDU &&
                      uppercase(strip(get(FITSIO.read_header(fits[i]), "EXTNAME", ""))) == "SPECTRUM"]
        length(candidates) == 1 ||
            throw(ArgumentError("expected exactly one binary SPECTRUM extension in $path"))
        hdu = only(candidates)
        header = FITSIO.read_header(hdu)
        spectrum_type = uppercase(strip(get(header, "HDUCLAS4", "")))
        spectrum_type in ("TYPE:II", "TYPE II") &&
            throw(ArgumentError("channel scalings require a Type-I SPECTRUM extension in $path"))
        role = uppercase(strip(get(header, "HDUCLAS2", "")))
        role == "NET" &&
            throw(ArgumentError("use the original source-region PHA and measured background in place of NET data in $path"))
        wrong_role = is_background ? role == "TOTAL" : role in ("BKG", "BACKGROUND", "OFF")
        wrong_role && throw(ArgumentError("PHA role $role does not match the supplied spectrum in $path"))
        names = FITSIO.colnames(hdu)
        "CHANNEL" in names || throw(ArgumentError("missing CHANNEL column in $path"))
        raw_channels = read(hdu, "CHANNEL")
        raw_channels isa AbstractVector{<:Integer} ||
            throw(ArgumentError("expected one integer CHANNEL label per Type-I row in $path"))
        channels = Int.(raw_channels)
        wanted = spectrum.channels
        _channel_index(wanted, "loaded spectrum")
        channel_rows = _channel_index(channels, "PHA")
        rows = [get(channel_rows, channel, 0) for channel in wanted]
        all(>(0), rows) ||
            throw(ArgumentError("a loaded channel is missing from SPECTRUM in $path"))
        _check_pha_counts(data, spectrum, hdu, header, names, rows, length(channels), path)
        function read_scale(name)
            if name in names
                return Float64.(_pha_column(hdu, name, length(channels), path)[rows])
            end
            return fill(Float64(get(header, name, 1.0)), length(rows))
        end
        return _PHAScalings(copy(wanted), _PHAScale(read_scale("AREASCAL")),
                            _PHAScale(read_scale("BACKSCAL")))
    finally
        close(fits)
    end
end

"""
    with_channel_scalings(data; source, background=nothing)

Attach OGIP `AREASCAL` and `BACKSCAL` to a loaded, ungrouped
SpectralFitting dataset for use with SpectralGrouping. Supply the original
source PHA and, if the dataset has a background, its original background PHA.
Read the `SPECTRUM` columns by `CHANNEL`, falling back to scalar keywords and
then to one. The returned object shares the original dataset and its mask.

Use `SpectralFitting.mask_energies!` or `SpectralFitting.restrict_domain!`
on the returned object or the original dataset to select channels. Pass the
returned object to `plot_data`, where `rebin` groups bins for display, or
prepare a fit with `match_dataset_grids` and
`combine_datasets(...; mode=:joint, groupings=...)`. Omit `groupings` to fit
on native channels. Prepare the joint dataset again after changing the mask.
"""
function with_channel_scalings(
    data::SF.SpectralData; source::AbstractString,
    background::Union{Nothing,AbstractString} = nothing,
)
    isnothing(data.background) == isnothing(background) ||
        throw(ArgumentError("supply the original background PHA exactly when the dataset has an attached background"))
    source_scalings = _read_pha_scalings(source, data, data.spectrum, false)
    background_scalings = if !isnothing(background)
        _read_pha_scalings(background, data, data.background, true)
    end
    return _ChannelScaledData(data, source_scalings, background_scalings, nothing)
end

function _selected_rows(data::_ChannelScaledData, mask)
    off, off_scales = data.background, data.background_scalings
    aligned = data.spectrum.channels == data.source_scalings.channels &&
        (isnothing(off_scales) ? isnothing(off) : !isnothing(off) && off.channels == off_scales.channels)
    aligned || throw(ArgumentError("loaded channel labels or background changed; attach the original PHA scalings again"))
    return _selected_rows(data.data, mask)
end

_pha_scale(data::SF.SpectralData, spectrum, rows, key) = Float64(getproperty(spectrum, key))

function _pha_scale(data::_ChannelScaledData, spectrum, rows, key)
    scalings = spectrum === data.spectrum ? data.source_scalings : data.background_scalings
    return getproperty(scalings, key).values[rows]
end
