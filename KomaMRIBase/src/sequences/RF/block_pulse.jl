
function block_pulse(flip_angle::FlipAngle, duration::Duration;
    phase_offset=0, freq_offset=0, delay=0, sys=Scanner())

    flip_angle, duration = flip_angle.val, duration.val
    amplitude = flip_angle / (2π * γ * duration) * exp(im * phase_offset)
    delay = max(delay, sys.RF_dead_time_T)
    block_duration = delay + duration + sys.RF_ring_down_T

    rf = RF(amplitude, duration, freq_offset, delay)
    return Sequence([Grad(0, 0);;], [rf;;], [ADC(0, 0)], block_duration)
end

function block_pulse(flip_angle::FlipAngle, bandwidth::Bandwidth; time_bw_product=0.25, kwargs...)
    duration = Duration(time_bw_product / bandwidth.val)
    return block_pulse(flip_angle, duration; kwargs...)
end

export block_pulse
