"""
Returns a boolean value indicating whether two vectors are approximately equal.
"""
function safe_approx(v1, v2)
    if length(v1) != length(v2)
        return false
    end
    return v1 ≈ v2
end

"""
Returns a boolean value indicating whether two vector of angles are equal.
"""
function isequal_angles(a1, a2)
    if length(a1) != length(a2)
        return false
    end
    return abs(sum(exp.(1im * 2π * (a1)) .* exp.(-1im * 2π * (a2)))) / length(a1) == 1
end

"""
Returns a boolean value indicating whether a vector is present in a list of vectors.
"""
function not_in_list(vec, vec_list)
    return all([!safe_approx(vec, arr) for arr in vec_list])
end

"""
Returns a boolean value indicating whether a vector is present in a list of vectors.
"""
function not_in_list_angles(angle, angle_list)
    return all([!isequal_angles(angle, phase) for phase in angle_list])
end

"""
Returns a boolean value indicating whether an event is "on".
"""
function is_event_on(event)
    return any([sum(abs.(getfield(event, key))) > 0 for key in fieldnames(typeof(event))])
end

"""
Returns the vector of "on" events (RF, Grad or ADC).
"""
function get_events_on(event_array)
    return [event for event in event_array if is_event_on(event)]
end

"""
Returns the a vector of vectors [obj, id] of unique events and IDs which are unique given
an input vector of events.
"""
function get_events_obj_id(events::Vector)
    events_obj_id, id_cnt = [], 1
    for obj in events
        if all([!(obj ≈ obj_unique) for (obj_unique, _) in events_obj_id])
            push!(events_obj_id, [obj, id_cnt]); id_cnt += 1
        end
    end
    return events_obj_id
end

"""
Returns the vector of vectors [blk, obj, id] for all the blocks `blk` of an array of events.
It is neccessary to add the input vector `events_obj_id` which contains the uniques objects
(RF, Grad, or ADC) with its repective ID.
"""
function get_events_blk_obj_id(event_array, events_obj_id::Vector)
    events_blk_obj_id = [[blk, obj, 0] for (blk, obj) in enumerate(event_array)]
    for boi in events_blk_obj_id
        for (obj, id) in events_obj_id
            if boi[2] ≈ obj
                boi[3] = id
            end
        end
    end
    return events_blk_obj_id
end

"""
Returns the unique shapes for the magnitude, angle and time of the "rfs_obj_id" vector.
Requires an initial integer counter "id_shape_cnt" to asign IDs incrementally.
"""
function get_rf_shapes(rfs_obj_id::Vector, id_shape_cnt::Integer, Δt_rf)
    # Find the unique shapes (magnitude, phase and time shapes) and assign IDs
    rfs_abs_id, rfs_ang_id, rfs_tim_id = [], [], []
    for (obj, _) in rfs_obj_id
        shape_abs = abs.(obj.A) / maximum(abs.(obj.A))
        if not_in_list(shape_abs, [shape for (shape, _) in rfs_abs_id])
            push!(rfs_abs_id, [shape_abs, id_shape_cnt])
            id_shape_cnt += 1
        end
        shape_ang = mod.(angle.(obj.A), 2π) / 2π
        ang = shape_ang .- shape_ang[1]
        list_ang_unique = [shape .- shape[1] for (shape, _) in rfs_ang_id]
        if not_in_list_angles(ang, list_ang_unique)
            push!(rfs_ang_id, [shape_ang, id_shape_cnt])
            id_shape_cnt += 1
        end
        if isa(obj.T, Vector{<:Number})
            shape_tim = cumsum([0; obj.T]) / Δt_rf
            if not_in_list(shape_tim, [shape for (shape, _) in rfs_tim_id])
                push!(rfs_tim_id, [shape_tim, id_shape_cnt])
                id_shape_cnt += 1
            end
        end
    end
    return rfs_abs_id, rfs_ang_id, rfs_tim_id, id_shape_cnt
end

"""
    gradunique_amp_id, gradunique_tim_id, id_shape_cnt = get_gradunique(gradunique_obj_id::Vector, id_shape_cnt::Integer, Δt_gr)

Returns the unique shapes for the amplitude and time of the "gradunique_obj_id" vector.
Requires an initial integer counter "id_shape_cnt" to asign IDs incrementally.
"""
function get_gradunique(gradunique_obj_id::Vector, id_shape_cnt::Integer, Δt_gr)
    # Find shapes for magnitude and time gradients
    gradunique_amp_id, gradunique_tim_id = [], []
    for (obj, _) in gradunique_obj_id
        shape_amp = obj.A / maximum(abs.(obj.A))
        if not_in_list(shape_amp, [shape for (shape, _) in gradunique_amp_id])
            push!(gradunique_amp_id, [shape_amp, id_shape_cnt])
            id_shape_cnt = id_shape_cnt + 1
        end
        if isa(obj.T, Vector{<:Number})
            shape_tim = cumsum([0; obj.T]) / Δt_gr
            if not_in_list(shape_tim, [shape for (shape, _) in gradunique_tim_id])
                push!(gradunique_tim_id, [shape_tim, id_shape_cnt])
                id_shape_cnt = id_shape_cnt + 1
            end
        end
    end
    return gradunique_amp_id, gradunique_tim_id, id_shape_cnt
end

"""
    write_seq(seq::Sequence, filename::String)

Writes a .seq file for a given sequence `seq` y the location `filename`
"""
function write_seq(seq::Sequence, filename)
    Δt_rf = seq.DEF["RadiofrequencyRasterTime"]
    Δt_gr = seq.DEF["GradientRasterTime"]
    # Get the unique objects (RF, Grad y ADC) and its IDs
    rfs_obj_id = get_events_obj_id(get_events_on(seq.RF))
    grunique_obj_id = get_events_obj_id(get_events_on(seq.GR))
    adcunique_obj_id = get_events_obj_id(get_events_on(seq.ADC))
    gradunique_obj_id = [[obj, id] for (obj, id) in grunique_obj_id if length(obj.A) != 1]
    trapunique_obj_id = [[obj, id] for (obj, id) in grunique_obj_id if length(obj.A) == 1]
    rfs_abs_id, rfs_ang_id, rfs_tim_id, id_shape_cnt = get_rf_shapes(rfs_obj_id, 1, Δt_rf)
    gradunique_amp_id, gradunique_tim_id, _ = get_gradunique(
        gradunique_obj_id, id_shape_cnt, Δt_gr
    )
    @warn "EXTENSIONS will not be handled"
    # [BLOCKS]: Define the table to be written in the [BLOCKS] section
    # Columns of table_blocks:
    # [blk, seq[blk], id_rf, id_gx, id_gy, id_gz, id_adc, id_ext]
    r = [id for (_, _, id) in get_events_blk_obj_id(seq.RF, rfs_obj_id)]
    x = [id for (_, _, id) in get_events_blk_obj_id(seq.GR.x, grunique_obj_id)]
    y = [id for (_, _, id) in get_events_blk_obj_id(seq.GR.y, grunique_obj_id)]
    z = [id for (_, _, id) in get_events_blk_obj_id(seq.GR.z, grunique_obj_id)]
    a = [id for (_, _, id) in get_events_blk_obj_id(seq.ADC, adcunique_obj_id)]
    table_blocks = [[b, s, 0, r[b], x[b], y[b], z[b], a[b], 0] for (b, s) in enumerate(seq)]
    for row in table_blocks
        blk = row[1]
        bd = seq.DUR[blk] / seq.DEF["BlockDurationRaster"]
        bdr = round(bd)
        if abs(bdr - bd) > 1e-6
            @warn "Block $blk duration rounded"
        end
        row[3] = bdr
    end
    # [RF]: Define the table to be written in the [RF] section
    # Columns of table_rf:
    # [id, rf_obj, amp, id_mag, id_phase, id_time, delay, freq, phase]
    table_rf = [[id, obj, 0, 0, 0, 0, 0, 0, 0] for (obj, id) in rfs_obj_id]
    for row in table_rf
        obj = row[2]
        row[3] = γ * maximum(abs.(obj.A))
        row[8] = obj.Δf
        shape_abs = abs.(obj.A) / maximum(abs.(obj.A))
        for (shape_abs_unique, id_abs) in rfs_abs_id
            if safe_approx(shape_abs, shape_abs_unique)
                row[4] = id_abs
            end
        end
        shape_ang = mod.(angle.(obj.A), 2π) / 2π
        ang = shape_ang .- shape_ang[1]
        for (shape_ang_unique, id_ang) in rfs_ang_id
            ang_unique = shape_ang_unique .- shape_ang_unique[1]
            if isequal_angles(ang, ang_unique)
                row[5] = id_ang
                row[9] = angle(
                    sum(exp.(1im * 2π * shape_ang) .* exp.(-1im * 2π * shape_ang_unique)) /
                    length(shape_ang),
                )
            end
        end
        if isa(obj.T, Vector{<:Number})
            shape_tim = cumsum([0; obj.T]) / Δt_rf
            for (shape_tim_unique, id_tim) in rfs_tim_id
                if safe_approx(shape_tim, shape_tim_unique)
                    row[6] = id_tim
                end
            end
        end
        delay_compensation_rf_koma = (row[6] == 0) * Δt_rf / 2
        row[7] = round((obj.delay - delay_compensation_rf_koma) / Δt_rf) * Δt_rf * 1e6
    end
    # Define the table to be written for the [GRADIENTS] section
    grad_idx_obj_amp_iamp_itim_delay = [
        [idx, obj, 0, 0, 0, 0] for (obj, idx) in gradunique_obj_id
    ]
    for ioamtd in grad_idx_obj_amp_iamp_itim_delay
        obj = ioamtd[2]
        ioamtd[3] = γ * maximum(abs.(obj.A))    # this always stores positive values, the waveform vector have the respective positive or negative values
        ioamtd[6] = round(1e6 * obj.delay)
        shape_amp = obj.A / maximum(abs.(obj.A))
        for (shape_amp_unique, id_amp) in gradunique_amp_id
            if safe_approx(shape_amp, shape_amp_unique)
                ioamtd[4] = id_amp
            end
        end
        if isa(obj.T, Vector{<:Number})
            shape_tim = cumsum([0; obj.T]) / Δt_gr
            for (shape_tim_unique, id_tim) in gradunique_tim_id
                if safe_approx(shape_tim, shape_tim_unique)
                    ioamtd[5] = id_tim
                end
            end
        end
    end
    # Define the table to be written for the [TRAP] section
    trap_idx_obj_amp_rise_flat_fall_delay = [
        [idx, obj, 0, 0, 0, 0, 0] for (obj, idx) in trapunique_obj_id
    ]
    for ioarfad in trap_idx_obj_amp_rise_flat_fall_delay
        obj = ioarfad[2]
        ioarfad[3] = γ * obj.A
        ioarfad[4] = 1e6 * obj.rise
        ioarfad[5] = 1e6 * obj.T
        ioarfad[6] = 1e6 * obj.fall
        ioarfad[7] = 1e6 * obj.delay
    end
    # Define the table to be written for the [ADC] section
    adc_idx_obj_num_dwell_delay_freq_phase = [
        [idx, obj, 0, 0, 0, 0, 0] for (obj, idx) in adcunique_obj_id
    ]
    for ionwdfp in adc_idx_obj_num_dwell_delay_freq_phase
        obj = ionwdfp[2]
        ionwdfp[3] = obj.N
        ionwdfp[4] = obj.T * 1e9 / (obj.N - 1)
        ionwdfp[5] = (obj.delay - 0.5 * obj.T / (obj.N - 1)) * 1e6
        ionwdfp[6] = obj.Δf
        ionwdfp[7] = obj.ϕ
    end
    # Define the table to be written for the [SHAPES] section
    shapefull_data_id = [
        shapeunique_data_id_i for shapeunique_data_id in [
            rfs_abs_id,
            rfs_ang_id,
            rfs_tim_id,
            gradunique_amp_id,
            gradunique_tim_id,
        ] for shapeunique_data_id_i in shapeunique_data_id
    ]
    shape_data_id_num = [
        (
            if length(compress_shape(data)[2]) == length(data)
                data
            else
                compress_shape(data)[2]
            end,
            id,
            length(data),
        ) for (data, id) in shapefull_data_id
    ]
    # Write the .seq file
    open(filename, "w") do fid
        @printf(fid, "# Pulseq sequence file\n")
        @printf(fid, "# Created by KomaMRI.jl \n\n") #TODO: add Koma version
        @printf(fid, "[VERSION]\n")
        @printf(fid, "major 1\n")
        @printf(fid, "minor 4\n")
        @printf(fid, "revision 1\n")
        @printf(fid, "\n")
        if !isempty(seq.DEF)
            @printf(fid, "[DEFINITIONS]\n")
            sorted_keys = sort(collect(keys(seq.DEF)))
            for key in sorted_keys
                val = seq.DEF[key]
                @printf(fid, "%s ", key)
                if isa(val, String)
                    @printf(fid, "%s ", val)
                else
                    if isa(val, Vector{<:Number})
                        for v in val
                            @printf(fid, "%.9g ", v)
                        end
                    else
                        @printf(fid, "%.9g ", val)
                    end
                end
                @printf(fid, "\n")
            end
            @printf(fid, "\n")
        end
        if !isempty(table_blocks)
            @printf(fid, "# Format of blocks:\n")
            @printf(fid, "# NUM DUR RF  GX  GY  GZ  ADC  EXT\n")
            @printf(fid, "[BLOCKS]\n")
            id_format_str = "%" * string(length(string(length(seq)))) * "d "
            fmt = Printf.Format(id_format_str * "%3d %3d %3d %3d %3d %2d %2d\n")
            for (blk, _, dur, rf, gx, gy, gz, adc, ext) in table_blocks
                Printf.format(fid, fmt, blk, dur, rf, gx, gy, gz, adc, ext)
            end
            @printf(fid, "\n")
        end
        if !isempty(table_rf)
            @printf(fid, "# Format of RF events:\n")
            @printf(fid, "# id amplitude mag_id phase_id time_shape_id delay freq phase\n")
            @printf(fid, "# ..        Hz   ....     ....          ....    us   Hz   rad\n")
            @printf(fid, "[RF]\n")
            fmt = Printf.Format("%d %12g %d %d %d %g %g %g\n")
            for (id, _, amp, mag_id, pha_id, time_id, delay, freq, pha) in table_rf
                Printf.format(fid, fmt, id, amp, mag_id, pha_id, time_id, delay, freq, pha)
            end
            @printf(fid, "\n")
        end
        if !isempty(grad_idx_obj_amp_iamp_itim_delay)
            @printf(fid, "# Format of arbitrary gradients:\n")
            @printf(
                fid,
                "#   time_shape_id of 0 means default timing (stepping with grad_raster starting at 1/2 of grad_raster)\n"
            )
            @printf(fid, "# id amplitude amp_shape_id time_shape_id delay\n") # do we need delay ???
            @printf(fid, "# ..      Hz/m       ..         ..          us\n")
            @printf(fid, "[GRADIENTS]\n")
            for (id, _, amp, ampid, timeid, delay) in grad_idx_obj_amp_iamp_itim_delay
                @printf(fid, "%d %12g %d %d %d\n", id, amp, ampid, timeid, delay)
            end
            @printf(fid, "\n")
        end
        if !isempty(trap_idx_obj_amp_rise_flat_fall_delay)
            @printf(fid, "# Format of trapezoid gradients:\n")
            @printf(fid, "# id amplitude rise flat fall delay\n")
            @printf(fid, "# ..      Hz/m   us   us   us    us\n")
            @printf(fid, "[TRAP]\n")
            for (id, _, amp, rise, flat, fall, delay) in
                trap_idx_obj_amp_rise_flat_fall_delay
                @printf(fid, "%2d %12g %3d %4d %3d %3d\n", id, amp, rise, flat, fall, delay)
            end
            @printf(fid, "\n")
        end
        if !isempty(adc_idx_obj_num_dwell_delay_freq_phase)
            @printf(fid, "# Format of ADC events:\n")
            @printf(fid, "# id num dwell delay freq phase\n")
            @printf(fid, "# ..  ..    ns    us   Hz   rad\n")
            @printf(fid, "[ADC]\n")
            for (id, _, num, dwell, delay, freq, phase) in
                adc_idx_obj_num_dwell_delay_freq_phase
                @printf(fid, "%d %d %.0f %.0f %g %g\n", id, num, dwell, delay, freq, phase)
            end
            @printf(fid, "\n")
        end
        if !isempty(shape_data_id_num)
            @printf(fid, "# Sequence Shapes\n")
            @printf(fid, "[SHAPES]\n\n")
            for (data, id, num) in shape_data_id_num
                @printf(fid, "shape_id %d\n", id)
                @printf(fid, "num_samples %d\n", num)
                [@printf(fid, "%.9g\n", datai) for datai in data]
                @printf(fid, "\n")
            end
        end
    end
    md5hash = bytes2hex(open(md5, filename))
    open(filename, "a") do fid
        @printf(fid, "\n[SIGNATURE]\n") # the preceding new line BELONGS to the signature (and needs to be sripped away to recalculate the signature)
        @printf(
            fid,
            "# This is the hash of the Pulseq file, calculated right before the [SIGNATURE] section was added\n"
        )
        @printf(
            fid,
            "# It can be reproduced/verified with md5sum if the file trimmed to the position right above [SIGNATURE]\n"
        )
        @printf(
            fid,
            "# The new line character preceding [SIGNATURE] BELONGS to the signature (and needs to be sripped away for recalculating/verification)\n"
        )
        @printf(fid, "Type md5\n")
        @printf(fid, "Hash %s\n", md5hash)
    end
end