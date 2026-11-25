import FITSIO
using LinearAlgebra: I

function write_scale_pha(path, channels, counts;
                         areas = nothing, backscales = nothing, exposure = 100.0,
                         role = "TOTAL", rate = false, keywords = ())
    files = FITSIO.FITS(path, "w")
    try
        columns = Dict{String,Any}("CHANNEL" => Int32.(channels),
                                   (rate ? "RATE" : "COUNTS") => Float64.(counts))
        isnothing(areas) || (columns["AREASCAL"] = Float64.(areas))
        isnothing(backscales) || (columns["BACKSCAL"] = Float64.(backscales))
        write(files, columns; name = "SPECTRUM")
        hdu = files["SPECTRUM"]
        FITSIO.write_key(hdu, "HDUCLAS2", role)
        FITSIO.write_key(hdu, "EXPOSURE", exposure)
        for (key, value) in keywords
            FITSIO.write_key(hdu, key, value)
        end
    finally
        close(files)
    end
    return path
end

@testset "Channel-scaled RGS counts, response, and W-stat" begin
    mktempdir() do dir
        source = write_scale_pha(joinpath(dir, "source.FTZ"), [22, 21], [20, 30];
                                 areas = [1.0, 0.5], backscales = [0.5, 0.4],
                                 keywords = ("AREASCAL" => 9.0, "BACKSCAL" => 8.0))
        background = write_scale_pha(joinpath(dir, "background.FTZ"), [21, 22], [100, 10];
                                     exposure = 200.0, areas = [1.0, 0.5], role = "BKG")
        poisson = SF.ErrorStatistics.Poisson
        on = spectrum([21, 22], [30, 20]; exposure = 100.0, statistics = poisson)
        off = spectrum([22, 21], [10, 100]; exposure = 200.0, statistics = poisson)
        arf = SF.AncillaryResponse([1.0, 2.0], [2.0, 3.0], [2.0, 3.0])
        data = response_dataset(on, [1.0, 2.0], [2.0, 3.0], [1.0, 2.0], [2.0, 3.0],
                                Matrix{Float64}(I, 2, 2); background = off, ancillary = arf)
        scaled = with_channel_scalings(data; source, background)
        @test scaled.data === data
        @test scaled.data_mask === data.data_mask
        @test data.spectrum.area_scale == data.background.background_scale == 1.0
        @test data.response.matrix == Matrix{Float64}(I, 2, 2)
        @test data.spectrum.data == [30, 20]
        @test data.background.data == [10, 100]

        match = match_dataset_grids(scaled)
        @test_throws ArgumentError match_dataset_grids(scaled, data)
        joint = combine_datasets(match)
        member = only(joint.members)
        @test joint.counts == [30, 20]
        @test member.background == [100, 10]
        @test isapprox(member.alpha, [0.1, 0.5])
        @test predict_counts(joint, [1.0, 1.0]) == [100, 300]
        @test_throws ArgumentError SF.regroup!(scaled; min_counts = 50)
        @test_throws ArgumentError combine_datasets(match; mode = :sum)
        @test_throws ArgumentError with_channel_scalings(data; source)

        @test SF.mask_energies!(scaled, 1.5, 4) === scaled
        @test data.data_mask == [false, true]
        masked = combine_datasets(match)
        @test masked.counts == [20]
        @test only(masked.members).background == [10]
        @test isapprox(only(masked.members).alpha, [0.5])
        @test isapprox(only(plot_curves(scaled; kind = :counts, rebin = 2).data).y, [15])
        @test joint.counts == [30, 20]
        SF.mask_energies!(scaled, 0, 4)
        @test data.data_mask == [false, true]
        fill!(scaled.data_mask, true)
        @test SF.restrict_domain!(scaled, e -> e < 2.5) === scaled
        @test combine_datasets(match).counts == [30]
        fill!(scaled.data_mask, true)
        @test isapprox(only(plot_curves(scaled; kind = :counts, rebin = 2).data).y, [35])

        stat = grid_statistic(joint)
        wstat(p) = SF.measure(stat, joint.counts, p, joint.variance)
        @test isapprox(wstat([15.0, 12.0]), 1.3319045383)
        @test isapprox(SF.ForwardDiff.gradient(wstat, [15.0, 12.0]), [-0.3814771167, -0.2984378813])
        @test isapprox(wstat([20.0, 15.0]), 0.0; atol = 1e-12)

        model = GridTestSource(SF.FitParam(1.0))
        plan = grouping_plan(scaled, :width, 2)
        @test plan.groups == [1:2]
        @test grouping_plan(scaled, :counts, 40).groups == plan.groups
        grouped = group_width(scaled, 2)
        @test grouped.spectrum.data == [50]
        @test grouped.background.data == [110]
        @test data.spectrum.data == [30, 20]
        grouped_fitdata = combine_datasets(match; groupings = (plan,))
        grouped_member = only(grouped_fitdata.members)
        @test grouped_fitdata.counts == [50]
        @test grouped_member.background == [110]
        @test isapprox(grouped_member.alpha, [0.225])
        @test isapprox(predict_counts(grouped_fitdata, [1.0, 1.0]), [400])
        @test isapprox(predict_counts(grouped_fitdata, [1.0, 2.0]), [700])
        @test isapprox(predict_counts(combine_datasets(match_dataset_grids(grouped)), [1.0, 2.0]), [700])
        grouped_stat = grid_statistic(grouped_fitdata)
        grouped_cfg = SF.FittingConfig(SF.FittingProblem(model => grouped_fitdata); stat = grouped_stat)
        @test isapprox(SF.measure_objective!(grouped_cfg, [0.075], 1), 0.3866405997830449)
        grouped_fit = SF.finalize_result(grouped_cfg, [0.075], nothing)
        grouped_plot = plot_curves(grouped_fit; kind = :counts, res = 2, residual_stat = :wstat)
        @test isapprox(only(grouped_plot.residuals).y, [-only(grouped_fit.stats)])
        for curves in (grouped_plot, plot_curves(grouped; kind = :counts),
                       plot_curves(scaled; kind = :counts, groupings = plan))
            @test isapprox(only(curves.data).y, [35])
            @test isapprox(only(curves.data).error .^ 2, [53.5])
        end
        @test isapprox(only(plot_curves(grouped; kind = :counts, background = :only).data).y, [15])

        cfg = SF.FittingConfig(SF.FittingProblem(model => joint); stat)
        fit = SF.finalize_result(cfg, [1.0], nothing)
        @test isapprox(only(fit.stats), wstat([100.0, 300.0]))
        for input in (scaled, match, joint, fit)
            opts = (; kind = :counts, combine = :sum, matching_coordinate = :energy,
                    model = input === fit ? nothing : model)
            net = plot_curves(input; opts..., background = :subtract, rebin = 2)
            @test isapprox(only(net.data).y, [35])
            @test isapprox(only(net.data).error .^ 2, [53.5])
            @test isapprox(only(net.model).y, [400])
            include_off = plot_curves(input; opts..., background = :include, rebin = 2)
            @test isapprox(only(include_off.data).y, [50])
            @test isapprox(only(include_off.model).y, [415])
            @test isapprox(only(plot_curves(input; opts..., background = :only, rebin = 2).data).y, [15])
            unfolded = only(plot_curves(input; kind = :unfold, matching_coordinate = :energy).data)
            @test isapprox(unfolded.y, [0.2, 0.05])
        end

        native = plot_curves(fit; kind = :counts, res = 2, residual_stat = :wstat)
        @test isapprox(sum(abs, only(native.residuals).y), only(fit.stats))
        @test_throws ArgumentError plot_curves(fit; combine = :sum, rebin = 2, res = 4)
        @test only(plot_curves(fit; combine = :sum, rebin = 2, res = 4,
                               residual_stat = :chi2).residuals).valid == [true]
        ratio = plot_curves(fit; combine = :sum, rebin = 2, res = 6)
        @test isapprox(only(ratio.residuals).y, [35 / 400])

        offset = match_dataset_grids(scaled; detector_grid = [1.0, 1.5, 2.0, 3.0])
        fractional = only(plot_curves(offset; kind = :counts, combine = :sum,
                                       matching_coordinate = :energy).data)
        @test isapprox(fractional.y, [10, 10, 15])
        @test isapprox(Matrix(fractional.covariance), [7.75 7.75 0; 7.75 7.75 0; 0 0 22.5])
        @test_throws ArgumentError plot_curves(offset; combine = :sum, model, res = 4,
                                               residual_stat = :wstat, matching_coordinate = :energy)

        coupled = deepcopy(data)
        coupled.response = SF.ResponseMatrix(sparse([0.5 1.0; 1.0 0.25]), [22, 21],
                                             [2.0, 1.0], [3.0, 2.0], [1.0, 2.0], [2.0, 3.0])
        rematched = match_dataset_grids(with_channel_scalings(coupled; source, background), data)
        mixed = combine_datasets(rematched)
        @test isapprox(predict_counts(mixed, [1.0, 1.0]), [137.5, 400.0, 200.0, 300.0])
        @test isapprox(grid_statistic(mixed).alpha, [0.1, 0.5, 0.5, 0.5])
        plain = joint_orders([1.0, 1.5, 3.0], ([0, 5],), (0.5,))
        multistat = grid_statistic(joint, plain, mixed)
        for input in (joint, plain, mixed)
            predicted = predict_counts(input, diff(input.model_edges))
            @test SF.measure(multistat, input.counts, predicted, input.variance) ==
                  SF.measure(grid_statistic(input), input.counts, predicted, input.variance)
        end
    end
end

@testset "Regrouping preserves native calibration and mask gaps" begin
    mktempdir() do dir
        source = write_scale_pha(joinpath(dir, "source.pha"), 1:4, [6, 7, 8, 9];
                                 areas = [1, 9, 2, 3], backscales = [2, 8, 1, 3])
        background = write_scale_pha(joinpath(dir, "background.pha"), 4:-1:1, [4, 3, 2, 1];
                                     role = "BKG")
        on = spectrum(1:4, [6, 7, 8, 9]; exposure = 100.0, flags = [1, -1, 1, -1])
        off = spectrum(1:4, [1, 2, 3, 4]; exposure = 100.0)
        scaled = with_channel_scalings(dataset(on; background = off); source, background)
        plan = grouping_plan(scaled, :ogip)
        @test plan.groups == [1:2, 3:4]
        regrouped = group_width(group_ogip(scaled), 2)
        grouped = group_width(scaled, 4)
        joint(data) = combine_datasets(match_dataset_grids(data))
        @test isapprox(only(joint(regrouped).members).alpha, [22.5])
        @test isapprox(predict_counts(joint(regrouped), [2.0, 1.0]),
                       predict_counts(joint(grouped), [2.0, 1.0]))
        @test isapprox(only(plot_curves(regrouped; kind = :counts, background = :only).data).y,
                       [188])
        SF.mask_energies!(scaled, e -> e != 3.0)
        @test grouping_plan(scaled, :width, 4).groups == [1:1, 4:4]
        @test_throws ArgumentError grouped_dataset(scaled, plan)
        @test joint(group_width(scaled, 4)).counts == [6, 9]
    end
end

@testset "PHA validation and selected-channel scales" begin
    mktempdir() do dir
        # The first row is masked and may have invalid calibration values.
        source = write_scale_pha(joinpath(dir, "source.pha"), [1, 2], [0.1, 0.2];
                                 areas = [0.0, 0.5], rate = true,
                                 keywords = ("BACKSCAL" => 0.4,))
        background = write_scale_pha(joinpath(dir, "background.pha"), [2, 1], [6, 8]; role = "BKG")
        on = spectrum(1:2, [10, 20]; exposure = 100.0)
        off = spectrum(1:2, [8, 6]; exposure = 100.0)
        data = dataset(on; mask = BitVector([false, true]), background = off)
        scaled = with_channel_scalings(data; source, background)
        joint = combine_datasets(match_dataset_grids(scaled))
        @test joint.counts == [20]
        @test isapprox(only(joint.members).alpha, [0.2])
        @test isapprox(only(plot_curves(scaled; kind = :counts).data).y, [18.8])
        data.data_mask[1] = true
        @test_throws ArgumentError combine_datasets(match_dataset_grids(scaled))
        data.data_mask[1] = false

        off.quality[2] = 1
        @test_throws ArgumentError combine_datasets(match_dataset_grids(scaled))
        @test_throws ArgumentError plot_curves(scaled; kind = :counts)
        @test only(plot_curves(scaled; kind = :counts, background = :include).data).y == [20]
        off.quality[2] = 0

        zero_off = write_scale_pha(joinpath(dir, "zero_off.pha"), [2, 1], [6, 8];
                                   role = "BKG", areas = [0.0, 1.0])
        bad = with_channel_scalings(data; source, background = zero_off)
        @test_throws ArgumentError combine_datasets(match_dataset_grids(bad))
        missing = write_scale_pha(joinpath(dir, "missing.pha"), [1], [8]; role = "BKG")
        duplicate = write_scale_pha(joinpath(dir, "duplicate.pha"), [2, 2], [6, 6]; role = "BKG")
        wrong_counts = write_scale_pha(joinpath(dir, "wrong.pha"), [1, 2], [8, 9]; role = "BKG")
        for path in (missing, duplicate, wrong_counts)
            @test_throws ArgumentError with_channel_scalings(data; source, background = path)
        end
        malformed = write_scale_pha(joinpath(dir, "malformed.pha"), [1, 2], [8, 6];
                                    role = "BKG", areas = [1.0 1.0; 1.0 1.0],
                                    keywords = ("AREASCAL" => 1.0,))
        @test_throws ArgumentError with_channel_scalings(data; source, background = malformed)
        source_only = dataset(deepcopy(on); mask = copy(data.data_mask))
        calibrated = with_channel_scalings(source_only; source)
        original = combine_datasets(match_dataset_grids(source_only))
        corrected = combine_datasets(match_dataset_grids(calibrated))
        @test isapprox(predict_counts(corrected, [1.0, 1.0]), 0.5 .* predict_counts(original, [1.0, 1.0]))
        @test grid_statistic(corrected) isa SF.ChiSquared

        net = write_scale_pha(joinpath(dir, "net.pha"), [1, 2], [10, 20]; role = "NET")
        @test_throws ArgumentError with_channel_scalings(source_only; source = net)
        type_ii = write_scale_pha(joinpath(dir, "type-ii.pha"), [1, 2], [10, 20];
                                   keywords = ("HDUCLAS4" => "TYPE:II",))
        @test_throws ArgumentError with_channel_scalings(source_only; source = type_ii)
        regrouped = group_width(dataset(deepcopy(on)), 2)
        @test_throws ArgumentError with_channel_scalings(regrouped; source)
    end
end
