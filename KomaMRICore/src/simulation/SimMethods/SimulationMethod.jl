#Bloch is the default simulation method
struct Bloch <: SimulationMethod end

export Bloch

include("Magnetization.jl")

@functor Mag #Gives gpu acceleration capabilities, see GPUFunctions.jl

function sim_output_dim(
    obj::Phantom{T}, seq::Sequence, sys::Scanner, sim_method::SimulationMethod
) where {T<:Real}
    return (sum(seq.ADC.N), 1) #Nt x Ncoils, This should consider the coil info from sys
end

"""Magnetization initialization for Bloch simulation method."""
function initialize_spins_state(
    obj::Phantom{T}, sim_method::SimulationMethod
) where {T<:Real}
    Nspins = length(obj)
    Mxy = zeros(T, Nspins)
    Mz = obj.ρ
    Xt = Mag{T}(Mxy, Mz)
    return Xt, obj
end

"""Stores pre-allocated arrays for use in run_spin_precession! and run_spin_excitation!"""
abstract type PreallocResult{T<:Real} end

"""Default preallocation struct, stores nothing."""
struct DefaultPrealloc{T} <: PreallocResult{T} end

Base.view(p::PreallocResult, i::UnitRange) = p

"""Default preallocation function."""
prealloc(sim_method::SimulationMethod, backend::KA.Backend, obj::Phantom{T}, M::Mag{T}, max_block_length::Integer, groupsize) where {T<:Real} = DefaultPrealloc{T}()

include("BlochSimple/BlochSimple.jl")
include("Bloch/cpu/BlochCPU.jl")
include("Bloch/gpu/BlochGPU.jl")
include("BlochDict/BlochDict.jl")