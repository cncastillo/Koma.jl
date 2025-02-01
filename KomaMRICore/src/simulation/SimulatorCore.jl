abstract type SimulationMethod end #get all available types by using subtypes(KomaMRI.SimulationMethod)
abstract type SpinStateRepresentation{T<:Real} end #get all available types by using subtypes(KomaMRI.SpinStateRepresentation)

#Defined methods:
include("SimMethods/SimulationMethod.jl")  #Defines simulation methods

"""
    sim_params = default_sim_params(sim_params=Dict{String,Any}())

This function returns a dictionary containing default simulation parameters while also
allowing the user to define some of them.

# Arguments
- `sim_params`: (`::Dict{String,Any}`, `=Dict{String,Any}()`) user-defined dictionary with
    simulation parameters. The following lists its keys along with their possible values:
    * "return_type": defines the output of the [`simulate`](@ref) function. Possible values
        are `"raw"`, `"mat"`, and `"state"`, corresponding to outputting a MRIReco
        `RawAcquisitionData`, the signal values, and the last magnetization state of the
        simulation, respectively
    * "sim_method": defines the type of simulation. The default value is `Bloch()`, but you
        can alternatively use the `BlochDict()` simulation method. Moreover, you have the
        flexibility to create your own methods without altering the KomaMRI source code
    * "Δt": raster time for gradients
    * "Δt_rf": raster time for RFs
    * "precision": defines the floating-point simulation precision. You can choose between
        `"f32"` and `"f64"` to use `Float32` and `Float64` primitive types, respectively.
        It's important to note that, especially for GPU operations, using `"f32"` is
        generally much faster
    * "Nblocks": divides the simulation into a specified number of time blocks. This parameter
        is designed to conserve RAM resources, as **KomaMRI** computes a series of
        simulations consecutively, each with the specified number of blocks determined by
        the value of `"Nblocks"`
    * "Nthreads": divides the **Phantom** into a specified number of threads. Because spins
        are modeled independently of each other, **KomaMRI** can solve simulations in
        parallel threads, speeding up the execution time
    * "gpu": is a boolean that determines whether to use GPU or CPU hardware resources, as
        long as they are available on the host computer
    * "gpu_device": default value is 'nothing'. If set to integer or device instance, calls the
        corresponding function to set the device of the available GPU in the host computer 
        (e.g. CUDA.device!)

# Returns
- `sim_params`: (`::Dict{String,Any}`) dictionary with simulation parameters
"""
function default_sim_params(sim_params=Dict{String,Any}())
    sampling_params = KomaMRIBase.default_sampling_params()
    get!(sim_params, "gpu", true)
    get!(sim_params, "gpu_device", nothing)
    get!(sim_params, "gpu_groupsize_precession", DEFAULT_PRECESSION_GROUPSIZE)
    get!(sim_params, "gpu_groupsize_excitation", DEFAULT_EXCITATION_GROUPSIZE)
    get!(sim_params, "Nthreads", Threads.nthreads())
    get!(sim_params, "Nblocks", 20)
    get!(sim_params, "Δt", sampling_params["Δt"])
    get!(sim_params, "Δt_rf", sampling_params["Δt_rf"])
    get!(sim_params, "sim_method", Bloch())
    get!(sim_params, "precision", "f32")
    get!(sim_params, "return_type", "raw")
    return sim_params
end

"""
    sig, Xt = run_spin_precession_parallel(obj, seq, M; Nthreads)

Implementation in multiple threads for the simulation in free precession,
separating the spins of the phantom `obj` in `Nthreads`.

# Arguments
- `obj`: (`::Phantom`) Phantom struct
- `seq`: (`::Sequence`) Sequence struct

# Keywords
- `Nthreads`: (`::Int`, `=Threads.nthreads()`) number of process threads for
    dividing the simulation into different phantom spin parts

# Returns
- `sig`: (`Vector{ComplexF64}`) raw signal over time
- `Xt`: (`::Vector{Mag}`) final state of the Mag vector (or the initial state for the
    next simulation step (the next step can be another precession step or an excitation
    step))
"""
function run_spin_precession_parallel!(
    obj::Phantom{T},
    seq::DiscreteSequence{T},
    sig::AbstractArray{Complex{T}},
    Xt::SpinStateRepresentation{T},
    sim_method::SimulationMethod,
    groupsize::Integer,
    backend::KA.Backend,
    prealloc::PreallocResult;
    Nthreads=Threads.nthreads(),
) where {T<:Real}
    parts = kfoldperm(length(obj), Nthreads)
    dims = [Colon() for i in 1:(ndims(sig) - 1)] # :,:,:,... Ndim times

    ThreadsX.foreach(enumerate(parts)) do (i, p)
        run_spin_precession!(
            @view(obj[p]), seq, @view(sig[dims..., i]), @view(Xt[p]), sim_method, groupsize, backend, @view(prealloc[p])
        )
    end

    return nothing
end

"""
    M0 = run_spin_excitation_parallel(obj, seq, Xt; Nthreads)

It gives rise to a rotation of M0 with an angle given by the efective magnetic field
(including B1, gradients and off resonance) and with respect to a rotation axis. It uses
different number threads to excecute the process.

# Arguments
- `obj`: (`::Phantom`) Phantom struct
- `seq`: (`::Sequence`) Sequence struct

# Keywords
- `Nthreads`: (`::Int`, `=Threads.nthreads()`) number of process threads for
    dividing the simulation into different phantom spin parts

# Returns
- `M0`: (`::Vector{Mag}`) final state of the Mag vector after a rotation (or the initial
    state for the next precession simulation step)
"""
function run_spin_excitation_parallel!(
    obj::Phantom{T},
    seq::DiscreteSequence{T},
    sig::AbstractArray{Complex{T}},
    Xt::SpinStateRepresentation{T},
    sim_method::SimulationMethod,
    groupsize::Integer,
    backend::KA.Backend,
    prealloc::PreallocResult;
    Nthreads=Threads.nthreads(),
) where {T<:Real}
    parts = kfoldperm(length(obj), Nthreads)
    dims = [Colon() for i in 1:(ndims(sig) - 1)] # :,:,:,... Ndim times

    ThreadsX.foreach(enumerate(parts)) do (i, p)
        run_spin_excitation!(
            @view(obj[p]), seq, @view(sig[dims..., i]), @view(Xt[p]), sim_method, groupsize, backend, @view(prealloc[p])
        )
    end

    return nothing
end

"""
    S_interp, M0 = run_sim_time_iter(obj, seq, t, Δt; Nblocks, Nthreads, gpu, w)

Performs the simulation over the total time vector `t` by dividing the time into `Nblocks`
parts to reduce RAM usage and spliting the spins of the phantom `obj` into `Nthreads` to
take advantage of CPU parallel processing.

# Arguments
- `obj`: (`::Phantom`) Phantom struct
- `seq`: (`::Sequence`) Sequence struct
- `t`: (`::Vector{Float64}`, `[s]`) non-uniform time vector
- `Δt`: (`::Vector{Float64}`, `[s]`) delta time of `t`

# Keywords
- `Nblocks`: (`::Int`, `=16`) number of groups for spliting the simulation over time
- `Nthreads`: (`::Int`, `=Threads.nthreads()`) number of process threads for
    dividing the simulation into different phantom spin parts
- `gpu`: (`::Function`) function that represents the gpu of the host
- `w`: (`::Any`, `=nothing`) flag to regard a progress bar in the blink window UI. If
    this variable is differnet from nothing, then the progress bar is considered

# Returns
- `S_interp`: (`::Vector{ComplexF64}`) interpolated raw signal
- `M0`: (`::Vector{Mag}`) final state of the Mag vector
"""
function run_sim_time_iter!(
    obj::Phantom,
    seq::DiscreteSequence,
    sig::AbstractArray{Complex{T}},
    Xt::SpinStateRepresentation{T},
    sim_method::SimulationMethod,
    backend::KA.Backend;
    Nblocks=1,
    Nthreads=Threads.nthreads(),
    precession_groupsize=DEFAULT_PRECESSION_GROUPSIZE,
    excitation_groupsize=DEFAULT_EXCITATION_GROUPSIZE,
    parts=[1:length(seq)],
    excitation_bool=ones(Bool, size(parts)),
    w=nothing,
) where {T<:Real}
    # Simulation
    rfs = 0
    samples = 1
    progress_bar = Progress(Nblocks; desc="Running simulation...")
    prealloc_result = prealloc(sim_method, backend, obj, Xt, maximum(length.(parts))+1, precession_groupsize)

    for (block, p) in enumerate(parts)
        seq_block = @view seq[p]
        # Params
        # excitation_bool = is_RF_on(seq_block) #&& is_ADC_off(seq_block) #PATCH: the ADC part should not be necessary, but sometimes 1 sample is identified as RF in an ADC block
        Nadc = sum(seq_block.ADC)
        acq_samples = samples:(samples + Nadc - 1)
        dims = [Colon() for i in 1:(ndims(sig) - 1)] # :,:,:,... Ndim times
        # Simulation wrappers
        if excitation_bool[block]
            run_spin_excitation_parallel!(
                obj, seq_block, @view(sig[acq_samples, dims...]), Xt, sim_method, excitation_groupsize, backend, prealloc_result; Nthreads
            )
            rfs += 1
        else
            run_spin_precession_parallel!(
                obj, seq_block, @view(sig[acq_samples, dims...]), Xt, sim_method, precession_groupsize, backend, prealloc_result; Nthreads
            )
        end
        KA.synchronize(backend)
        samples += Nadc
        #Update progress
        next!(
            progress_bar;
            showvalues=[
                (:simulated_blocks, block), (:rf_blocks, rfs), (:acq_samples, samples - 1)
            ],
        )
        update_blink_window_progress!(w, block, Nblocks)
    end

    return nothing
end

"""Updates KomaUI's simulation progress bar."""
function update_blink_window_progress!(w::Nothing, block, Nblocks)
    return nothing
end

"""
Separates the discrete sequence into Nblocks, ensuring that each block has either
RF-on or RF-off. The function returns the ranges of the discrete sequence blocks along
with a boolean vector indicating whether each block has RF.
"""
function get_sim_ranges(seqd::DiscreteSequence; Nblocks)
    ranges = UnitRange{Int}[]
    ranges_bool = Bool[]
    start_idx_rf_block = 0
    start_idx_gr_block = 0
    #Split 1:N into Nblocks like kfoldperm
    N = length(seqd.Δt)
    k = min(N, Nblocks)
    n, r = divrem(N, k) #N >= k, N < k
    breaks = collect(1:n:(N + 1))
    for i in eachindex(breaks)
        breaks[i] += i > r ? r : i - 1
    end
    breaks = breaks[2:(end - 1)] #Remove borders,
    #Iterate over B1 values to decide the simulation UnitRanges
    for i in eachindex(seqd.Δt)
        if abs(seqd.B1[i]) > 1e-9 #TODO: This is needed as the function ⏢ in get_rfs is not very accurate
            if start_idx_rf_block == 0 #End RF block
                start_idx_rf_block = i
            end
            if start_idx_gr_block > 0 #End of GR block
                push!(ranges, start_idx_gr_block:(i - 1))
                push!(ranges_bool, false)
                start_idx_gr_block = 0
            end
        else
            if start_idx_gr_block == 0 #Start GR block
                start_idx_gr_block = i
            end
            if start_idx_rf_block > 0 #End of RF block
                push!(ranges, start_idx_rf_block:(i - 1))
                push!(ranges_bool, true)
                start_idx_rf_block = 0
            end
        end
        #More subdivisions
        if i in breaks
            if start_idx_rf_block > 0 #End of RF block
                if length(start_idx_rf_block:(i - 1)) > 1
                    push!(ranges, start_idx_rf_block:(i - 1))
                    push!(ranges_bool, true)
                    start_idx_rf_block = i
                end
            end
            if start_idx_gr_block > 0 #End of RF block
                if length(start_idx_gr_block:(i - 1)) > 1
                    push!(ranges, start_idx_gr_block:(i - 1))
                    push!(ranges_bool, false)
                    start_idx_gr_block = i
                end
            end
        end
    end
    #Finishing the UnitRange's
    if start_idx_rf_block > 0
        push!(ranges, start_idx_rf_block:N)
        push!(ranges_bool, true)
    end
    if start_idx_gr_block > 0
        push!(ranges, start_idx_gr_block:N)
        push!(ranges_bool, false)
    end
    #Output
    return ranges, ranges_bool
end

"""
    out = simulate(obj::Phantom, seq::Sequence, sys::Scanner; sim_params, w)

Returns the raw signal or the last state of the magnetization according to the value
of the `"return_type"` key of the `sim_params` dictionary. 

This is a wrapper function to `run_sim_time_iter`, which converts the inputs to the appropriate types and discretizes the sequence before simulation. The reported simulation time only considers `run_sim_time_iter`, as the preprocessing duration should be negligible compared to the simulation time (if this is not the case, please file a bug report). 

# Arguments
- `obj`: (`::Phantom`) Phantom struct
- `seq`: (`::Sequence`) Sequence struct
- `sys`: (`::Scanner`) Scanner struct

# Keywords
- `sim_params`: (`::Dict{String,Any}`, `=Dict{String,Any}()`) simulation parameter dictionary
- `w`: (`::Blink.AtomShell.Window`, `=nothing`) the window within which to display a
    progress bar in the Blink Window UI. If this variable is anything other than 'nothing',
    the progress bar will be considered

# Returns
- `out`: (`::Vector{Complex}` or `::SpinStateRepresentation` or `::RawAcquisitionData`) depending
    on whether "return_type" is "mat", "state" or "raw" (default), respectively

# Examples
```julia-repl
julia> seq_file = joinpath(dirname(pathof(KomaMRI)), "../examples/5.koma_paper/comparison_accuracy/sequences/EPI/epi_100x100_TE100_FOV230.seq");

julia> sys, obj, seq = Scanner(), brain_phantom2D(), read_seq(seq_file)

julia> raw = simulate(obj, seq, sys)

julia> plot_signal(raw)
```
"""
function simulate(
    obj::Phantom, seq::Sequence, sys::Scanner; sim_params=Dict{String,Any}(), w=nothing
)
    #Simulation parameter unpacking, and setting defaults if key is not defined
    sim_params = default_sim_params(sim_params)
    #Warn if user is trying to run on CPU without enabling multi-threading
    if (!sim_params["gpu"] && Threads.nthreads() == 1)
        @info """Simulation will be run on the CPU with only 1 thread. To enable multi-threading, start julia with --threads=auto
        """ maxlog=1
    end
    # Simulation init
    seqd = discretize(seq; sampling_params=sim_params, motion=obj.motion) # Sampling of Sequence waveforms
    parts, excitation_bool = get_sim_ranges(seqd; Nblocks=sim_params["Nblocks"]) # Generating simulation blocks
    t_sim_parts = [seqd.t[p[1]] for p in parts]
    append!(t_sim_parts, seqd.t[end])
    # Spins' state init (Magnetization, EPG, etc.), could include modifications to obj (e.g. T2*)
    Xt, obj = initialize_spins_state(obj, sim_params["sim_method"])
    # Signal init
    Ndims = sim_output_dim(obj, seq, sys, sim_params["sim_method"])
    backend = get_backend(sim_params["gpu"])
    sim_params["gpu"] &= backend isa KA.GPU
    if sim_params["gpu"]
        sim_params["Nthreads"] = 1
    end
    sig = zeros(ComplexF64, Ndims..., sim_params["Nthreads"])
    if !KA.supports_float64(backend) && sim_params["precision"] == "f64"
        sim_params["precision"] = "f32"
        @info """ Backend: '$(name(backend))' does not support 64-bit precision 
        floating point operations. Automatically converting to type Float32.
        (set sim_param["precision"] = "f32" to avoid seeing this message).
        """ maxlog=1
    end
    if sim_params["precision"] == "f64" && KA.supports_float64(backend)
        obj  = obj |> f64 #Phantom
        seqd = seqd |> f64 #DiscreteSequence
        Xt   = Xt |> f64 #SpinStateRepresentation
        sig  = sig |> f64 #Signal
    else
        #Precision  #Default
        obj  = obj |> f32 #Phantom
        seqd = seqd |> f32 #DiscreteSequence
        Xt   = Xt |> f32 #SpinStateRepresentation
        sig  = sig |> f32 #Signal
    end
    # Objects to GPU
    if backend isa KA.GPU
        isnothing(sim_params["gpu_device"]) || set_device!(backend, sim_params["gpu_device"])
        gpu_name = device_name(backend)
        obj = obj |> gpu #Phantom
        seqd = seqd |> gpu #DiscreteSequence
        Xt = Xt |> gpu #SpinStateRepresentation
        sig = sig |> gpu #Signal
    end

    # Simulation
    @info "Running simulation in the $(backend isa KA.GPU ? "GPU ($gpu_name)" : "CPU with $(sim_params["Nthreads"]) thread(s)")" koma_version =
        pkgversion(@__MODULE__) sim_method = sim_params["sim_method"] spins = length(obj) time_points = length(
        seqd.t
    ) adc_points = Ndims[1]
    @time timed_tuple = @timed run_sim_time_iter!(
        obj,
        seqd,
        sig,
        Xt,
        sim_params["sim_method"],
        backend;
        Nblocks=length(parts),
        Nthreads=sim_params["Nthreads"],
        precession_groupsize=sim_params["gpu_groupsize_precession"],
        excitation_groupsize=sim_params["gpu_groupsize_excitation"],
        parts,
        excitation_bool,
        w,
    )
    # Result to CPU, if already in the CPU it does nothing
    sig = sum(sig; dims=length(Ndims) + 1) |> cpu #Sum over threads
    sig .*= get_adc_phase_compensation(seq)
    Xt = Xt |> cpu
    # Output
    if sim_params["return_type"] == "state"
        out = Xt
    elseif sim_params["return_type"] == "mat"
        out = sig
    elseif sim_params["return_type"] == "raw"
        # To visually check the simulation blocks
        sim_params_raw = copy(sim_params)
        sim_params_raw["sim_method"] = string(sim_params["sim_method"])
        sim_params_raw["gpu"] = sim_params["gpu"]
        sim_params_raw["gpu_device"] = backend isa KA.GPU ? gpu_name : "nothing"
        sim_params_raw["Nthreads"] = sim_params["Nthreads"]
        sim_params_raw["t_sim_parts"] = t_sim_parts
        sim_params_raw["type_sim_parts"] = excitation_bool
        sim_params_raw["Nblocks"] = length(parts)
        sim_params_raw["sim_time_sec"] = timed_tuple.time
        sim_params_raw["allocations_bytes"] = timed_tuple.bytes

        out = signal_to_raw_data(
            sig, seq; phantom_name=obj.name, sys=sys, sim_params=sim_params_raw
        )
    end
    return out
end

"""
    mag = simulate_slice_profile(seq; z, sim_params)

Returns magnetization of spins distributed along `z` after running the Sequence struct.

# Arguments
- `seq`: (`::Sequence`) Sequence struct

# Keywords
- `z`: (`=range(-2e-2,2e-2,200)`) range for the z axis
- `sim_params`: (`::Dict{String, Any}`, `=Dict{String,Any}("Δt_rf"=>1e-6)`) dictionary with
    simulation parameters

# Returns
- `mag`: (`::SpinStateRepresentation`) final state of the magnetization vector
"""
function simulate_slice_profile(
    seq::Sequence; z=range(-2.e-2, 2.e-2, 200), sim_params=Dict{String,Any}("Δt_rf" => 1e-6)
)
    sim_params["return_type"] = "state"
    sys = Scanner()
    obj = Phantom{Float64}(; x=zeros(size(z)), z=Array(z))
    mag = simulate(obj, seq, sys; sim_params)
    return mag
end
