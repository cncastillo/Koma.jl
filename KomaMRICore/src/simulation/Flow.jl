"""
    reset_magnetization!
"""
function reset_magnetization!(M::Mag{T}, Mxy::AbstractArray{Complex{T}}, motion::NoMotion{T}, t::AbstractArray{T}) where {T<:Real}
   return nothing
end

function reset_magnetization!(M::Mag{T}, Mxy::AbstractArray{Complex{T}}, motion::MotionVector{T}, t::AbstractArray{T}) where {T<:Real}
   for m in motion.motions
      reset_magnetization!(M, Mxy, m, t)
   end
   return nothing
end

function reset_magnetization!(M::Mag{T}, Mxy::AbstractArray{Complex{T}}, motion::Motion{T}, t::AbstractArray{T}) where {T<:Real}
   return nothing
end

function reset_magnetization!(M::Mag{T}, Mxy::AbstractArray{Complex{T}}, motion::FlowTrajectory{T}, t::AbstractArray{T}) where {T<:Real}
    itp = interpolate(motion.resetmag, Gridded(Constant{Previous}), Val(size(x,1)))
    flags = resample(itp, unit_time(t, motion.times))
    reset = any(flags; dims=2)
    flags = .!(cumsum(flags; dims=2) .>= 1)
    Mxy .*= flags
    M.z[reset] = p.ρ[reset]
   return nothing
end