"""
"""
function trapezoid(; duration=0, amplitude=0, area=0, flat_area=0,
    flat_time=0, rise_time=0, fall_time=0, delay=0,
    max_grad=Inf, max_slew=Inf, Δtgr=1e-3, sys=nothing)

    if !isnothing(sys)
        max_grad = sys.Gmax
        max_slew = sys.Smax
        Δtgr = sys.GR_Δt
    end

    if area == 0 && flat_area == 0 && amplitude == 0
        @error "trapezoid: invalid keywords. Must supply either 'area', 'flat_area' or 'amplitude'"
    end
    if fall_time > 0 && rise_time == 0
        @error "trapezoid: invalid keywords. Must always supply 'rise_time' if 'fall_time' is specified explicitly."
    end

    if flat_time > 0
        if amplitude == 0
            if flat_area == 0
                @error "trapezoid: invalid keyworks. When 'flat_time' is provided either 'flat_area' or 'amplitude' must be provided as well; you may consider providing 'duration', 'area' and optionally ramp times instead."
            end
            amplitude = flat_area / flat_time
        end
        if rise_time == 0
            rise_time = abs(amplitude) / max_slew;
            rise_time = ceil(rise_time / Δtgr) * Δtgr;
            if rise_time == 0
                rise_time = Δtgr
            end
        end
        if fall_time == 0
            fall_time = rise_time
        end
    elseif duration > 0
        if amplitude == 0
            if rise_time == 0
                dC = 1 / abs(2 * max_slew) + 1 / abs(2 * max_slew)
                possible = duration^2 > 4 * abs(area) * dC;
                @assert possible "Requested area is too large for this gradient. Minimum required duration (assuming triangle gradient can be realized) is $(round(sqrt(4 * abs(area) * dC) * 1e6)) us"
                amplitude = (duration - sqrt(duration^2 - 4 * abs(area) * dC)) / (2 * dC)
            else
                if fall_time == 0
                    fall_time = rise_time
                end
                amplitude = area / (duration - 0.5 * rise_time - 0.5 * fall_time)
                possible = duration > (rise_time + fall_time) && abs(amplitude) < max_grad
                @assert possible "Requested area is too large for this gradient. Probably amplitude is violated ($(round(abs(amplitude) / max_grad * 100))%)"
            end
        end
        if rise_time == 0
            rise_time = ceil(abs(amplitude) / max_slew / Δtgr) * Δtgr
            if rise_time == 0
                rise_time = Δtgr
            end
        end
        if fall_time == 0
            fall_time = rise_time
        end
        flat_time = duration - rise_time - fall_time
        if amplitude == 0
            # Adjust amplitude (after rounding) to achieve given area
            amplitude = area / (rise_time / 2 + fall_time / 2 + flat_time)
        end
    else
        if area == 0
            @error "trapezoid: invalid keywords. Must supply area or duration"
        else
            # find the shortest possible duration
            # first check if the area can be realized as a triangle
            # if not we calculate a trapezoid
            rise_time = ceil(sqrt(abs(area) / max_slew) / Δtgr) * Δtgr
            if rise_time < Δtgr  # the "area" was probably 0 or almost 0 ...
                rise_time = Δtgr;
            end
            amplitude = area / rise_time
            t_eff = rise_time
            if abs(amplitude) > max_grad
                t_eff = ceil(abs(area) / max_grad / Δtgr) * Δtgr
                amplitude = area / t_eff
                if rise_time == 0
                    rise_time = Δtgr
                end
            end
            flat_time = t_eff - rise_time
            fall_time = rise_time
        end
    end

    @assert abs(amplitude) <= max_grad "trapezoid: invalid amplitude. Amplitude violation ($(round(abs(amplitude) / max_grad * 100))%)"

    return Grad(amplitude, flat_time, rise_time, fall_time, delay)
end

"""
"""
function arbitrary_grad(waveform; delay=0,
    max_grad=Inf, max_slew=Inf, Δtgr=1e-3, sys=nothing)

    if !isnothing(sys)
        max_grad = sys.Gmax
        max_slew = sys.Smax
        Δtgr = sys.GR_Δt
    end

    slew = (waveform[2:end] - waveform[1:end-1]) / Δtgr
    if !isempty(slew)
        @assert maximum(abs.(slew)) <= max_slew "Slew rate violation ($(maximum(abs.(slew)) / max_slew * 100)%)"
    end
    @assert maximum(abs.(waveform)) <= max_grad "Gradient amplitude violation ($(maximum(abs.(waveform)) / max_grad * 100)%)"

    duration = (length(waveform)-1) * Δtgr
    return Grad(waveform, duration, 0, 0, delay)
end
