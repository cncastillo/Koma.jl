"""
"""
function make_adc(num_samles; Δtadc=0, duration=0, delay=0,
    freq_offset=0, phase_offset=0, dead_time=0, sys=nothing)

    if !isnothing(sys)
        dead_time = sys.ADC_dead_time_T
        Δtadc = sys.ADC_Δt
    end

    if (Δtadc == 0 && duration == 0) || (Δtadc > 0 && duration > 0)
        @error "Either dwell or duration must be defined"
    end
    if duration > 0
        Δtadc = duration / num_samles
    elseif Δtadc > 0
        duration = Δtadc * num_samles;
    end
    if dead_time > delay
        delay = dead_time; # adcDeadTime is added before the actual sampling (and also second time after the sampling period)
    end

    adc = ADC(num_samles, duration, delay, freq_offset, phase_offset)

    return adc
end
