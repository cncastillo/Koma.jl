function sinc_pulse(flip_angle::FlipAngle, duration::Duration, time_bw_product::TimeBwProduct;
    phase_offset=0, freq_offset=0, delay=0, sys=Scanner(), apodization=0.5, centerpos=0.5)

    flip_angle, duration, time_bw_product = flip_angle.val, duration.val, time_bw_product.val

    dead_time = sys.RF_dead_time_T
    ring_down_time = sys.RF_ring_down_T
    Δtrf = sys.RF_Δt

    BW = time_bw_product / duration
    N = Integer(ceil(duration / Δtrf))
    t = range(0, duration; length=N)
    window = (1 - apodization) .+ apodization * cos.(2π * ((t  .- (centerpos * duration)) / duration))
    signal = window .* sinc.(BW * (t  .- (centerpos * duration)))
    flip = 0.5 * sum(signal[2:end] + signal[1:end-1]) * Δtrf * 2π
    signal = signal * flip_angle / flip * exp(im * phase_offset)
    delay = max(delay, dead_time)
    block_duration = delay + duration + ring_down_time

    rf = RF(signal, duration, freq_offset, delay)
    return Sequence([Grad(0, 0);;], [rf;;], [ADC(0, 0)], block_duration)
end

export sinc_pulse
