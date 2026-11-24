function _free_parameter_selectors(model)
    components = SF.is_composite(model) ?
        [(c, getproperty(model, c)) for c in propertynames(model)] :
        [(nothing, model)]
    return [isnothing(c) ? name : (c, name)
            for (c, part) in components
            for (name, par) in zip(SF.parameter_names(part), SF.parameter_vector(part))
            if SF.isfree(par)]
end

function _check_existing_ties(problem, selectors)
    isempty(problem.bindings) && return nothing

    intended_roots = Dict{Int,Int}()
    for selector in selectors
        fields = selector isa Symbol ? (selector,) : selector
        indices = [SF.parameter_index(problem, SF.ParameterTriple((i, fields...)))
                   for i in eachindex(problem.model.m)]
        for index in indices
            intended_roots[index] = first(indices)
        end
    end

    # Cross-parameter ties and ties back to model 1 can create multiple roots or cycles.
    for (root, targets) in problem.bindings, target in targets
        intended_root = get(intended_roots, root, nothing)
        target_root = get(intended_roots, target, nothing)
        if intended_root != target_root || target == target_root ||
                (!isnothing(target_root) && root == target)
            throw(ArgumentError("existing bindings conflict with tying corresponding parameters to model slot 1"))
        end
    end
    return nothing
end

"""
    tie_all_free!(problem::SF.FittingProblem)

Bind corresponding free parameters across all models in `problem` using
model slot 1 as the reference. Qualify composite parameters by component
label, including for wrapped composites. Every slot must have the same
free parameter selectors. Frozen parameters retain their values.

Call before constructing the fitting configuration. Use `SF.bind!` for
explicit ties across different model layouts or when individual parameters
should vary independently. Throws before changing any ties if the free
selectors differ or an existing binding conflicts with the requested ties.
"""
function tie_all_free!(problem::SF.FittingProblem)
    models = problem.model.m
    length(models) <= 1 && return problem
    selectors = _free_parameter_selectors(first(models))
    expected = Set(selectors)
    for i in 2:length(models)
        Set(_free_parameter_selectors(models[i])) == expected ||
            throw(ArgumentError("model slot $i has a different set of free parameter selectors than slot 1"))
    end

    _check_existing_ties(problem, selectors)
    return SF.bindall!(problem, selectors...)
end
