struct BindingTestWrapper{M,T,K} <: SF.AbstractModelWrapper{M,T,K}
    model::M
end

BindingTestWrapper(model::SF.AbstractSpectralModel{T,K}) where {T,K} =
    BindingTestWrapper{typeof(model),T,K}(model)

function binding_test_model(norm; freeze_second = false)
    first = SF.PowerLaw(K = SF.FitParam(norm), a = SF.FitParam(1.7; frozen = true))
    second = SF.PowerLaw(K = SF.FitParam(2 * norm; frozen = freeze_second), a = SF.FitParam(2.2))
    return BindingTestWrapper(first + second)
end

@testset "Share free parameters on wrapped composite fit models" begin
    joints = (
        joint_orders([1.0, 2, 4], ([0, 5],), (0.5,)),
        joint_orders([1.0, 1.5, 2.5, 4], ([6, 0],), (0.2,)),
        joint_orders([1.0, 3, 4], ([0, 9],), (0.3,)),
    )
    models = binding_test_model.((1.0, 3.0, 5.0))
    @test !(first(models) isa SF.CompositeModel)
    @test SF.is_composite(first(models))
    problem = SF.FittingProblem((model => data for (model, data) in zip(models, joints))...)
    stat = grid_statistic(joints...)
    @test length(SF.FittingConfig(problem; stat).u0) == 9
    @test tie_all_free!(problem) === problem
    cfg = SF.FittingConfig(problem; stat)
    @test sort(SF.get_value.(cfg.u0)) == [1.0, 2.0, 2.2]

    # Repeated normalizations keep their distinct component labels.
    components = propertynames(first(models))
    for c in components
        root = problem.lookup[SF.ParameterTriple((1, c, :K))]
        targets = [problem.lookup[SF.ParameterTriple((i, c, :K))] for i in (2, 3)]
        @test problem.bindings[root] == targets
    end
    before = deepcopy(problem.bindings)
    @test tie_all_free!(problem) === problem
    @test problem.bindings == before

    mismatch = SF.FittingProblem(models[1] => joints[1], models[2] => joints[2],
                                 binding_test_model(5.0; freeze_second = true) => joints[3])
    @test_throws ArgumentError tie_all_free!(mismatch)
    @test isempty(mismatch.bindings)

    mixed = SF.FittingProblem(models[1] => joints[1], SF.PowerLaw() => joints[2])
    @test_throws ArgumentError tie_all_free!(mixed)
    @test isempty(mixed.bindings)
end

@testset "Share free parameters with existing ties" begin
    data = dataset(spectrum(1:3, [5, 8, 12]))
    single = SF.FittingProblem(multi_grid_model() => data)
    @test tie_all_free!(single) === single
    @test isempty(single.bindings)

    fresh_problem() = SF.FittingProblem(multi_grid_model() => data,
                                        multi_grid_model() => data,
                                        multi_grid_model() => data)
    partially_tied = fresh_problem()
    SF.bind!(partially_tied, (2, :K) => (3, :K))
    tie_all_free!(partially_tied)
    @test length(SF.FittingConfig(partially_tied).u0) == 2

    for existing in ((2, :K) => (1, :K), (1, :K) => (2, :slope))
        conflicting = fresh_problem()
        SF.bind!(conflicting, existing)
        before = deepcopy(conflicting.bindings)
        @test_throws ArgumentError tie_all_free!(conflicting)
        @test conflicting.bindings == before
    end
end
