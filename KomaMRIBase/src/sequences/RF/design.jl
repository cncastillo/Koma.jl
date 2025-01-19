"""
"""
function sinc_pulse(flip_angle; duration=0, freq_offset=0, phase_offset=0,
    time_bw_product=0, apodization=0.5, centerpos=0.5, delay=0, slice_thickness=0,
    dead_time=0, ring_down_time=0, Δtrf=1e-5, Δtgr=1e-3, sys=nothing)

    if !isnothing(sys)
        dead_time = sys.RF_dead_time_T
        ring_down_time = sys.RF_ring_down_T
        Δtrf = sys.RF_Δt
        Δtgr = sys.GR_Δt
    end

    if duration <= 0
        @error "rf pulse duration must be positive"
    end
    if Δtrf <= 0
        @error "the Δtrf gradient raster time must be positive"
    end

    BW = time_bw_product / duration
    N = Integer(ceil(duration / Δtrf))
    t = range(0, duration; length=N)
    window = (1 - apodization) .+ apodization * cos.(2π * ((t  .- (centerpos * duration)) / duration))
    signal = window .* sinc.(BW * (t  .- (centerpos * duration)))
    flip = 0.5 * sum(signal[2:end] + signal[1:end-1]) * Δtrf * 2π
    signal = signal * flip_angle / flip * cis(phase_offset)
    if dead_time > delay
        delay = dead_time
    end
	rf = RF(signal, duration, freq_offset, delay)

    @assert slice_thickness > 0 "slice_thickness must be provided"

    amplitude = BW / slice_thickness
    area = amplitude * duration
    gz = trapezoid(; flat_time=duration, flat_area=area, sys=sys);
    gz_area = gz.A * (gz.T + gz.rise / 2 + gz.fall / 2)
    gzr_area = -area*(1 - centerpos) - 0.5*(gz_area - area)
    gzr = trapezoid(; sys=sys, area=gzr_area)
    if rf.delay > gz.rise
        gz.delay = ceil((rf.delay - gz.rise) / Δtgr) * Δtgr     # round-up to gradient raster
    end
    if rf.delay < gz.rise + gz.delay
        rf.delay = gz.rise + gz.delay   # these are on the grad raster already which is coarser
    end

    dly = Delay(0)
    if ring_down_time > 0
        dly = Delay(rf.delay + rf.T + ring_down_time)     # I NEED a review
    end

    return rf, gz, gzr, dly
end

"""
"""
function arbitrary_rf(signal, flip; freq_offset=0, phase_offset=0,
    time_bw_product=0, bandwidth=0, delay=0, slice_thickness=0,
    dead_time=0, ring_down_time=0, Δtrf=1e-5, Δtgr=1e-3, sys=nothing)

    if !isnothing(sys)
        dead_time = sys.RF_dead_time_T
        ring_down_time = sys.RF_ring_down_T
        Δtrf = sys.RF_Δt
        Δtgr = sys.GR_Δt
    end

    signal = signal / abs(sum(signal * Δtrf)) * flip / 2π * cis(phase_offset)

    N =  length(signal)
    duration = (N-1) * Δtrf
    t = range(0, duration; length=N)

    if dead_time > delay
        delay = dead_time;
    end

    rf = RF(signal, duration, freq_offset, delay)

    if time_bw_product > 0
        if bandwidth > 0
            @error "Both 'bandwidth' and 'time_bw_product' cannot be specified at the same time"
        else
            bandwidth = time_bw_product / duration
        end
    end

    @assert slice_thickness > 0 "SliceThickness must be provided"
    @assert bandwidth > 0 "Bandwidth of pulse must be provided"

    BW = bandwidth
    if time_bw_product > 0
        BW = time_bw_product / duration
    end

    amplitude = BW / slice_thickness
    area = amplitude * duration
    gz = trapezoid(; flat_time=duration, flat_area=area, sys=sys)
    gz_area = gz.A * (gz.T + gz.rise / 2 + gz.fall / 2)
    gzr_area = -area*(1 - KomaMRIBase.get_RF_center(rf) / rf.T) - 0.5*(gz_area - area)
    gzr = trapezoid(; sys=sys, area=gzr_area)


    if rf.delay > gz.rise
        gz.delay = ceil((rf.delay - gz.rise) / Δtgr) * Δtgr     # round-up to gradient raster
    end
    if rf.delay < gz.rise + gz.delay
        rf.delay = gz.rise + gz.delay   # these are on the grad raster already which is coarser
    end

    dly = Delay(0)
    if ring_down_time > 0
        dly = Delay(rf.delay + rf.T + ring_down_time)     # I NEED a review
    end

    return rf, gz, gzr, delay
end
