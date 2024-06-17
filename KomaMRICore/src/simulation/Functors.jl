import Adapt: adapt, adapt_storage
import Functors: @functor, functor, fmap, isleaf

#Aux. funcitons to check if the variable we want to move to the GPU is numeric
_isleaf(x) = isleaf(x)
_isleaf(::AbstractArray{<:Number}) = true
_isleaf(::AbstractArray{T}) where T = isbitstype(T)
_isleaf(::AbstractRange) = true

"""
	gpu(x, backend)

Tries to move `x` to the GPU backend specified in the 'backend' parameter. 

This works for functions, and any struct marked with `@functor`.

Use [`cpu`](@ref) to copy back to ordinary `Array`s.

See also [`f32`](@ref) and [`f64`](@ref) to change element type only.

# Examples
```julia
x = gpu(x, CUDABackend())
```
"""
function gpu(x, backend::KA.GPU)
    return fmap(x -> adapt(backend, x), x; exclude=_isleaf)
end

# To CPU
"""
	cpu(x)

Tries to move object to CPU. This works for functions, and any struct marked with `@functor`.

See also [`gpu`](@ref).

# Examples
```julia
x = x |> cpu
```
"""
cpu(x) = fmap(x -> adapt(KA.CPU(), x), x, exclude=_isleaf)

#MotionModel structs
adapt_storage(::KA.GPU, x::SimpleMotion) = x

#Precision
paramtype(T::Type{<:Real}, m) = fmap(x -> adapt(T, x), m)
adapt_storage(T::Type{<:Real}, xs::Real) = convert(T, xs)
adapt_storage(T::Type{<:Real}, xs::AbstractArray{<:Real}) = convert.(T, xs)
adapt_storage(T::Type{<:Real}, xs::AbstractArray{<:Complex}) = convert.(Complex{T}, xs)
adapt_storage(T::Type{<:Real}, xs::AbstractArray{<:Bool}) = xs
adapt_storage(T::Type{<:Real}, xs::NoMotion) = NoMotion{T}()
adapt_storage(T::Type{<:Real}, xs::SimpleMotion) = SimpleMotion(paramtype(T, xs.types))

"""
    f32(m)

Converts the `eltype` of model's parameters to `Float32`
Recurses into structs marked with `@functor`.

See also [`f64`](@ref).
"""
f32(m) = paramtype(Float32, m)

"""
    f64(m)

Converts the `eltype` of model's parameters to `Float64` (which is Koma's default)..
Recurses into structs marked with `@functor`.

See also [`f32`](@ref).
"""
f64(m) = paramtype(Float64, m)

#The functor macro makes it easier to call a function in all the parameters
# Phantom
@functor Phantom
# SimpleMotion
@functor Translation
@functor Rotation
@functor HeartBeat
@functor PeriodicTranslation
@functor PeriodicRotation
@functor PeriodicHeartBeat
# ArbitraryMotion
@functor ArbitraryMotion
# Spinor
@functor Spinor
# DiscreteSequence
@functor DiscreteSequence

export gpu, cpu, f32, f64