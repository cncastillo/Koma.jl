@kernel inbounds=true function precession_kernel!(
    sig_output::AbstractMatrix{Complex{T}}, 
    M_xy, M_z, 
    @Const(p_x), @Const(p_y), @Const(p_z), @Const(p_ΔBz), @Const(p_T1), @Const(p_T2), @Const(p_ρ), 
    @Const(s_Gx), @Const(s_Gy), @Const(s_Gz), @Const(s_Δt), @Const(s_ADC), 
    ::Val{MOTION}, ::Val{N_SPINS}, ::Val{GROUPSIZE}, ::Val{GROUPSIZE_CLOSEST_POWER_OF_TWO}, ::Val{GROUPSIZE_REMAINDER}
) where {T, MOTION, N_SPINS, GROUPSIZE, GROUPSIZE_CLOSEST_POWER_OF_TWO, GROUPSIZE_REMAINDER}

    i = @index(Global, Linear)
    i_l = @index(Local, Linear)
    i_g = @index(Group, Linear)

    @uniform s_length = length(s_Δt)+1

    sig_group_r = @localmem T GROUPSIZE
    sig_group_i = @localmem T GROUPSIZE
    sig_group_r[i_l] = zero(T)
    sig_group_i[i_l] = zero(T)
    
    Mxy_r = zero(T)
    Mxy_i = zero(T)
    t = zero(T)
    ϕ = zero(T)
    ΔBz = zero(T)
    T2 = zero(T)
    x = zero(T)
    y = zero(T)
    z = zero(T)
    Bz_prev = zero(T)
    Bz_next = zero(T)

    if i <= N_SPINS
        Mxy_r, Mxy_i = reim(M_xy[i])
        sig_group_r[i_l] = Mxy_r
        sig_group_i[i_l] = Mxy_i
        ΔBz = p_ΔBz[i]
        T2 = p_T2[i]
        x, y, z = get_spin_coordinates(p_x, p_y, p_z, i, 1)
        Bz_prev = x * s_Gx[1] + y * s_Gy[1] + z * s_Gz[1] + ΔBz
    end

    ADC_idx = 1
    if s_ADC[1]
        @synchronize()
        reduce_block!(sig_group_r, sig_group_i, i_l, GROUPSIZE, GROUPSIZE_CLOSEST_POWER_OF_TWO, GROUPSIZE_REMAINDER)
        if i_l == 1u16
            sig_output[i_g, 1] = complex(sig_group_r[1u16], sig_group_i[1u16])
        end
        ADC_idx += 1
    end

    for s_idx = 2:s_length
        if i <= N_SPINS
            if MOTION
                x, y, z = get_spin_coordinates(p_x, p_y, p_z, i, s_idx)
            end

            Δt = s_Δt[s_idx-1]
            t += Δt
            Bz_next = x * s_Gx[s_idx] + y * s_Gy[s_idx] + z * s_Gz[s_idx] + ΔBz
            ϕ += (Bz_prev + Bz_next) * T(-π * γ) * Δt
        end

        if s_idx < s_length && s_ADC[s_idx]
            if i <= N_SPINS
                ΔT2 = exp(-t / T2)
                cis_ϕ_i, cis_ϕ_r = sincos(ϕ)
                sig_group_r[i_l] = ΔT2 * (Mxy_r * cis_ϕ_r - Mxy_i * cis_ϕ_i)
                sig_group_i[i_l] = ΔT2 * (Mxy_r * cis_ϕ_i + Mxy_i * cis_ϕ_r)
            end
            @synchronize()
            reduce_block!(sig_group_r, sig_group_i, i_l, GROUPSIZE, GROUPSIZE_CLOSEST_POWER_OF_TWO, GROUPSIZE_REMAINDER)
            if i_l == 1u16
                sig_output[i_g, ADC_idx] = complex(sig_group_r[1u16], sig_group_i[1u16]) 
            end
            ADC_idx += 1
        end

        if i <= N_SPINS
            Bz_prev = Bz_next
        end
    end

    if i <= N_SPINS
        ΔT2 = exp(-t / T2)
        cis_ϕ_i, cis_ϕ_r = sincos(ϕ)
        M_xy[i] = complex(ΔT2 * (Mxy_r * cis_ϕ_r - Mxy_i * cis_ϕ_i), ΔT2 * (Mxy_r * cis_ϕ_i + Mxy_i * cis_ϕ_r))
        ΔT1 = exp(-t / p_T1[i])
        M_z[i] = M_z[i] * ΔT1 + p_ρ[i] * (1 - ΔT1)
    end
end