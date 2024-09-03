module KomaoneAPIExt

using oneAPI
import KomaMRICore, KomaMRIBase
import KomaMRICore.KA
import Adapt
import LinearAlgebra

KomaMRICore.name(::oneAPIBackend) = "oneAPI"
KomaMRICore.isfunctional(::oneAPIBackend) = oneAPI.functional()
KomaMRICore.set_device!(::oneAPIBackend, val) = oneAPI.device!(val)
KomaMRICore.device_name(::oneAPIBackend) = oneAPI.properties(oneAPI.device()).name
@inline KomaMRICore._cis(x) = cos(x) + im * sin(x)

function KomaMRICore._print_devices(::oneAPIBackend)
    devices = [
        Symbol("($(i-1)$(i == 1 ? "*" : " "))") => oneAPI.properties(d).name for
        (i, d) in enumerate(oneAPI.devices())
    ]
    @info "$(length(oneAPI.devices())) oneAPI capable device(s)." devices...
end

#Temporary workaround since oneAPI.jl (similar to Metal) does not support some array operations
#Once run_spin_excitation! and run_spin_precession! are kernel-based, this code can be removed
Base.cumsum(x::oneVector{T}) where T = convert(oneVector{T}, cumsum(KomaMRICore.cpu(x)))
Base.findall(x::oneVector{Bool}) = convert(oneVector, findall(KomaMRICore.cpu(x)))

using KernelAbstractions: @index, @kernel, @Const, synchronize

"""Naive cumsum implementation for matrix, parallelizes along the first dimension"""
Base.cumsum(A::oneArray{T}; dims) where T = begin
    dims == 2 || @error "oneAPI cumsum implementation only supports keyword argument dims=2"
    backend = oneAPIBackend()
    B = similar(A)
    cumsum_kernel = naive_cumsum!(backend)
    cumsum_kernel(B, A, ndrange=size(A,1))
    synchronize(backend)
    return B
end

Base.cumsum!(B::oneArray{T}, A::oneArray{T}; dims) where T = begin
    dims == 2 || @error "oneAPI cumsum implementation only supports keyword argument dims=2"
    backend = oneAPIBackend()
    cumsum_kernel = naive_cumsum!(backend)
    cumsum_kernel(B, A, ndrange=size(A,1))
    synchronize(backend)
end

## COV_EXCL_START
@kernel function naive_cumsum!(B, @Const(A))
    i = @index(Global)

    cur_val = 0.0f0
    for k ∈ 1:size(A, 2)
        @inbounds cur_val += A[i, k]
        @inbounds B[i, k] = cur_val
    end
end
## COV_EXCL_STOP

function __init__()
    push!(KomaMRICore.LOADED_BACKENDS[], oneAPIBackend())
    @warn "oneAPI does not support all array operations used by KomaMRI. GPU performance may be slower than expected"
end

const AdjointOneArray{T, N, M} = LinearAlgebra.Adjoint{T, oneArray{T, N, M}} where {T, N, M}
## Extend KomaMRIBase.unit_time (until bug with oneAPI is solved)
function KomaMRIBase.unit_time(t::AdjointOneArray{T, N, M}, ts::KomaMRIBase.TimeRange{T}) where {T, N, M}
    if ts.t_start == ts.t_end
        return (t .>= ts.t_start) .* oneunit(T)
    else
        tmp = (t .- ts.t_start) ./ (ts.t_end - ts.t_start)
        mask = (tmp .> zero(T))
        tmp .*= mask

        # tmp = max.((t .- ts.t_start) ./ (ts.t_end - ts.t_start), zero(T))
        # mini = min.(maxi, oneunit(T))
        KA.synchronize(KA.get_backend(t))
        # _ = sum(t)

        mask .= (tmp .<= oneunit(T))
        return tmp .* mask .+ .!mask
    end
end

end