using KernelAbstractions: @kernel, @Const, @index, @uniform, @localmem, @synchronize, @groupsize

#Used for getting spin coordinates inside precession and excitation kernels
@inline function get_spin_coordinates(x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}, i::Integer, t::Integer) where {T<:Real} 
    (x[i], y[i], z[i]) 
end
@inline function get_spin_coordinates(x::AbstractMatrix{T}, y::AbstractMatrix{T}, z::AbstractMatrix{T}, i::Integer, t::Integer) where {T<:Real} 
    (x[i, t], y[i, t], z[i, t]) 
end

# Returns the next least power of two starting from n, used to calculate remaining indexes in block-level reduction
@inline function next_least_power_of_two(n)
    return  n < 2 ? 1 :
            n < 4 ? 2 :
            n < 8 ? 4 :
            n < 16 ? 8 :
            n < 32 ? 16 :
            n < 64 ? 32 :
            n < 128 ? 64 :
            n < 256 ? 128 :
            n < 512 ? 256 :
            n < 1024 ? 512 :
            1024
end

@inline function reduce_block!(s_data1, s_data2, i, N, N_closest, R)
    if N == 1024u16
        if i <= 512u16
            s_data1[i] = s_data1[i] + s_data1[i + 512u16]
            s_data2[i] = s_data2[i] + s_data2[i + 512u16]
        end
    elseif N != N_closest
        #If workgroup size is not a power of two, need to take care of
        #remaining indices before entering next if statement
        if i <= R
            s_data1[i] = s_data1[i] + s_data1[i + N_closest]
            s_data2[i] = s_data2[i] + s_data2[i + N_closest]
        end
    end
    @synchronize()
    if N >= 512u16
        if i <= 256u16
            s_data1[i] = s_data1[i] + s_data1[i + 256u16]
            s_data2[i] = s_data2[i] + s_data2[i + 256u16]
        end
        @synchronize()
    end
    if N >= 256u16
        if i <= 128u16
            s_data1[i] = s_data1[i] + s_data1[i + 128u16]
            s_data2[i] = s_data2[i] + s_data2[i + 128u16]
        end
        @synchronize()
    end
    if N >= 128u16
        if i <= 64u16
            s_data1[i] = s_data1[i] + s_data1[i + 64u16]
            s_data2[i] = s_data2[i] + s_data2[i + 64u16]
        end
        @synchronize()
    end

    if N >= 64u16
        if i <= 32u16
            s_data1[i] = s_data1[i] + s_data1[i + 32u16]
            s_data2[i] = s_data2[i] + s_data2[i + 32u16]
        end
        @synchronize()
    end
    if N >= 32u16
        if i <= 16u16
            s_data1[i] = s_data1[i] + s_data1[i + 16u16]
            s_data2[i] = s_data2[i] + s_data2[i + 16u16]
        end
        @synchronize()
    end
    if N >= 16u16
        if i <= 8u16
            s_data1[i] = s_data1[i] + s_data1[i + 8u16]
            s_data2[i] = s_data2[i] + s_data2[i + 8u16]
        end
        @synchronize()
    end
    if N >= 8u16
        if i <= 4u16
            s_data1[i] = s_data1[i] + s_data1[i + 4u16]
            s_data2[i] = s_data2[i] + s_data2[i + 4u16]
        end
        @synchronize()
    end
    if N >= 4u16
        if i <= 2u16
            s_data1[i] = s_data1[i] + s_data1[i + 2u16]
            s_data2[i] = s_data2[i] + s_data2[i + 2u16]
        end
        @synchronize()
    end
    if N >= 2u16
        if i <= 1u16
            s_data1[i] = s_data1[i] + s_data1[i + 1u16]
            s_data2[i] = s_data2[i] + s_data2[i + 1u16]
        end
        @synchronize()
    end
end