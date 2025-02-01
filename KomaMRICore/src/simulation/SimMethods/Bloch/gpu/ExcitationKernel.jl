@kernel inbounds=true function excitation_kernel!(
    M_xy::AbstractVector{Complex{T}}, M_z, 
    @Const(p_x), @Const(p_y), @Const(p_z), @Const(p_ΔBz), @Const(p_T1), @Const(p_T2), @Const(p_ρ), 
    @Const(s_Gx), @Const(s_Gy), @Const(s_Gz), @Const(s_Δt), @Const(s_Δf), @Const(s_B1), 
    ::Val{MOTION}
) where {T, MOTION}
    
    i = @index(Global)
    
    @uniform N_Δt = length(s_Δt)

    x, y, z = get_spin_coordinates(p_x, p_y, p_z, i, 1)
    ΔBz = p_ΔBz[i]
    Mxy_r, Mxy_i = reim(M_xy[i])
    Mz = M_z[i]
    ρ = p_ρ[i]
    T1 = p_T1[i]
    T2 = p_T2[i]

    for t = 1:N_Δt
        if MOTION
            x, y, z = get_spin_coordinates(p_x, p_y, p_z, i, t)
        end
        
        Bz = (x * s_Gx[t] + y * s_Gy[t] + z * s_Gz[t]) + ΔBz - s_Δf[t] / T(γ)
        B1_r, B1_i = reim(s_B1[t])
        B = sqrt(B1_r^2 + B1_i^2 + Bz^2)
        Δt = s_Δt[t]
        φ = T(-π * γ) * B * Δt
        sin_φ, cos_φ = sincos(φ)
        α_r = cos_φ
        if iszero(B)
            α_i = -(Bz / (B + eps(T))) * sin_φ
            β_r = (B1_i / (B + eps(T))) * sin_φ
            β_i = -(B1_r / (B + eps(T))) * sin_φ
        else
            α_i = -(Bz / B) * sin_φ
            β_r = (B1_i / B) * sin_φ
            β_i = -(B1_r / B) * sin_φ
        end

        Mxy_new_r = 2 * (Mxy_i * (α_r * α_i - β_r * β_i) +
                    Mz * (α_i * β_i + α_r * β_r)) +
                    Mxy_r * (α_r^2 - α_i^2 - β_r^2 + β_i^2)
        
        Mxy_new_i = -2 * (Mxy_r * (α_r * α_i + β_r * β_i) -
                    Mz * (α_r * β_i - α_i * β_r)) +
                    Mxy_i * (α_r^2 - α_i^2 + β_r^2 - β_i^2)
        
        Mz_new =    Mz * (α_r^2 + α_i^2 - β_r^2 - β_i^2) -
                    2 * (Mxy_r * (α_r * β_r - α_i * β_i) +
                    Mxy_i * (α_r * β_i + α_i * β_r))
        
        ΔT1 = exp(-Δt / T1)
        ΔT2 = exp(-Δt / T2)
        Mxy_r = Mxy_new_r * ΔT2
        Mxy_i = Mxy_new_i * ΔT2
        Mz = Mz_new * ΔT1 + ρ * (1 - ΔT1)
    end

    M_xy[i] = complex(Mxy_r, Mxy_i)
    M_z[i] = Mz
end