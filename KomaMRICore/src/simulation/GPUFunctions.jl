const LOADED_BACKENDS = Ref{Vector{KA.GPU}}([])
const BACKEND = Ref{Union{KA.Backend,Nothing}}(nothing)
const DEFAULT_PRECESSION_GROUPSIZE = 32
const DEFAULT_EXCITATION_GROUPSIZE = 256

device_name(backend) = @error "device_name called with invalid backend type $(typeof(backend))"
isfunctional(::KA.CPU) = true
isfunctional(x) = false
_print_devices(backend) = @error "_print_devices called with invalid backend type $(typeof(backend))"
_print_devices(::KA.CPU) = @info "CPU: $(length(Sys.cpu_info())) x $(Sys.cpu_info()[1].model)"
name(::KA.CPU) = "CPU"
set_device!(backend, val) = @error "set_device! called with invalid parameter types: '$(typeof(backend))', '$(typeof(val))'" 
set_device!(val) = set_device!(get_backend(true), val)

#Copied from AcceleratedKernels.jl: https://github.com/JuliaGPU/AcceleratedKernels.jl/blob/00a9c0e7a36d6e02a51dbd13032a9165caab7909/src/utils.jl#L34
struct TypeWrap{T} end
TypeWrap(T) = TypeWrap{T}()
Base.:*(x::Number, ::TypeWrap{T}) where T = T(x)

const u16 = TypeWrap(UInt16)

"""
    get_backend(use_gpu)

Gets the simulation backend to use. If use_gpu=false or there are no available GPU backends, 
returns CPU(), else, returns the GPU backend (currently either CUDABackend(), MetalBackend(), 
ROCBackend(), or oneAPIBackend()).

The GPU package for the corresponding backend (CUDA.jl, Metal.jl, AMDGPU.jl, or oneAPI.jl) must be
loaded and functional, otherwise KomaMRI will default to using the CPU.

# Arguments
- 'use_gpu': ('::Bool') If true, attempt to use GPU and check for available backends

# Returns
- 'backend': (::KernelAbstractions.backend) The backend to use
"""
function get_backend(use_gpu::Bool)
    if !isnothing(BACKEND[])
        # The backend can be set and still need to change based on the value of
        # use_gpu, e.g. if switching between CPU and GPU while running tests
        ((BACKEND[] isa KA.GPU) == use_gpu) && return BACKEND[]
    end

    if !use_gpu
        BACKEND[] = KA.CPU()
        return BACKEND[]
    end

    if isempty(LOADED_BACKENDS[])
        @info """ 
          The GPU functionality is being called but no GPU backend is loaded 
          to access it. Add 'using CUDA / Metal / AMDGPU / oneAPI' to your 
          code. Defaulting back to the CPU. (No action is required if you want
          to run on the CPU).
        """ maxlog=1
        BACKEND[] = KA.CPU()
        return BACKEND[]
    end

    functional_gpu_backends = []
    for backend in LOADED_BACKENDS[]
        if isfunctional(backend)
            push!(functional_gpu_backends, backend)
        else
            @warn "Loaded backend $(name(backend)) is not functional" maxlog=1
        end
    end
    
    if length(functional_gpu_backends) == 1
        BACKEND[] = functional_gpu_backends[1]
        @info """Using  backend: '$(name(BACKEND[]))'"""  maxlog = 1
        return BACKEND[]
    elseif length(functional_gpu_backends) == 0
        @info """Defaulting back to the CPU. (No action is required if you want to run on the CPU). """ maxlog = 1
        BACKEND[] = KA.CPU()
        return BACKEND[]
    else
        # Will probably never get here
        @info """
          Multiple functional backends have been loaded and KomaMRI does not 
          know which one to use. Ensure that your code contains only one 'using' 
          statement for the GPU backend you wish to use. Defaulting back to the 
          CPU. (No action is required if you want to run on the CPU).
        """ maxlog = 1
        BACKEND[] = KA.CPU()
        return BACKEND[]
    end
end

"""
    print_devices()

Simple function to print available devices. Calls internal get_backend() function to
get the appropriate GPU / CPU backend and prints device information.

# Arguments
- 'use_gpu':  ('::Bool') If true, check for loaded / functional GPU backends and print appropriate warnings if no GPU backends have been loaded
"""
function print_devices(use_gpu = true)
    backend = get_backend(use_gpu)
    _print_devices(backend)
end

export print_devices, set_device!
