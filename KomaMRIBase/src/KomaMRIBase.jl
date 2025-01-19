module KomaMRIBase

#IMPORT PACKAGES
import Base.*, Base.+, Base.-, Base./, Base.vcat, Base.size, Base.abs, Base.getproperty
#General
using Reexport
#Datatypes
using Parameters
#Simulation
using Interpolations
#Reconstruction
using MRIBase
@reexport using MRIBase:
    Profile, RawAcquisitionData, AcquisitionData, AcquisitionHeader, EncodingCounters, Limit
using MAT   # For loading example phantoms

const global γ = 42.5774688e6 # Hz/T gyromagnetic constant for H1, JEMRIS uses 42.5756 MHz/T

# Hardware
include("datatypes/Scanner.jl")
# Sequence
include("datatypes/sequence/Grad.jl")
include("datatypes/sequence/RF.jl")
include("datatypes/sequence/ADC.jl")
include("timing/KeyValuesCalculation.jl")
include("datatypes/Sequence.jl")
include("datatypes/sequence/Delay.jl")
# Motion
include("motion/MotionList.jl")
include("motion/NoMotion.jl")
# Phantom
include("datatypes/Phantom.jl")
# Simulator
include("datatypes/simulation/DiscreteSequence.jl")
include("timing/TimeStepCalculation.jl")
include("timing/TrapezoidalIntegration.jl")

# Main
export γ    # gyro-magnetic ratio [Hz/T]
export Scanner, Sequence, Phantom
export Grad, RF, ADC, Delay
export dur, get_block_start_times, get_samples
export DiscreteSequence
export discretize, get_adc_phase_compensation, get_adc_sampling_times
export is_Gx_on, is_Gy_on, is_Gz_on, is_RF_on, is_ADC_on
export times, ampls, freqs
# This are also used for simulation
export kfoldperm, trapz, cumtrapz
# Phantom
export brain_phantom2D, brain_phantom3D, pelvis_phantom2D, heart_phantom
# Motion
export MotionList, NoMotion, Motion
export Translate, TranslateX, TranslateY, TranslateZ
export Rotate, RotateX, RotateY, RotateZ 
export HeartBeat, Path, FlowPath
export TimeRange, Periodic, TimeCurve
export SpinRange, AllSpins
export get_spin_coords
# Secondary
export get_kspace, rotx, roty, rotz
# Additionals
export get_flip_angles, is_RF_on, is_GR_on, is_ADC_on
# Sequence related
export get_Mk, get_kspace, get_M0, get_M1, get_M2

# PulseDesigner submodule
struct FlipAngle val end
struct Duration val end
struct Bandwidth val end
struct TimeBwProduct val end
struct SliceThickness val end
export FlipAngle, Duration, Bandwidth, TimeBwProduct, SliceThickness

include("sequences/PulseDesigner.jl")
export PulseDesigner

end # module KomaMRIBase
