using KomaMRI, PlotlyJS, MAT

# 1. DEFINE MAIN PARAMETERS
# 1.1 Define lists for tissue parameters - this is just an example of possible values
T1list = Array(vcat(50:50:500, 505:5:1000, 1050:50:1500, 1600:100:2000))*1e-3
T2list = Array(vcat(4:1:80, 85:5:200, 220:20:400))*1e-3

# 1.2 Compute combinations (T1-T2 pairs)
params_combs = vec(collect(Iterators.product(T1list, T2list, B1list, T1rholist, Dlist)))
# get ordered lists for T1-T2 values (for consistency)
T1values = [x[1] for x in params_combs]
T2values = [x[2] for x in params_combs]
Ncombinations = length(params_combs)


# 2. READ/LOAD SEQUENCE
seq_path = "path/to/your_seq.seq"
seq = read_seq(seq_path)

# 2.1 Add heart rate variability - if needed
# read .mat file with RRs vector
# RRs_struct is a MATLAB struct with a field subject_RRs which contains an array with the measured RRs
# this is just a way to do it, it may be different depending on your application
RRs_struct = matread("MRF_simulations/RRs/HV8_RRs.mat")
desired_RRs = RRs_struct["subject_RRs"][:]*1e-3

# 2.2 Get triggered blocks from sequence
trigg_idx = findall(seq.DEF["extension"] .!= 0)
t0 = get_block_start_times(seq)[1:end-1][trigg_idx] # The 1:end-1 is because start time are [0, t0, t1, .., tn]
current_RRs = diff(t0)

# 2.3 Modify RRs to the actual value
seq.DUR[trigg_idx[1:end-1]] .+= (desired_RRs .- current_RRs)
t0 = get_block_start_times(seq)[1:end-1][trigg_idx] 
new_RRs = diff(t0)

## plot the sequence - if you want
using KomaMRIPlots.PlotlyJS
p = plot_seq(seq)
for i in eachindex(t0)
    add_trace!(p, KomaMRIPlots.scatter(x=[t0[i], t0[i]]*1e3, y=[-30, 30],
        marker=attr(color="red"), legendgroup="RRs", showlegend=i==1, name="RRs")) # we mark the RR timestamps
end
p

# 2.4 Only sample the center of the readout - you may need to modify this for sequences with echoes
# you could also add more samples but computation time increases significantly
for (i, adc) in enumerate(seq.ADC[is_ADC_on.(seq)])
    adc.N = 1
end


# 3. SIMULATION
# 3.1 Define main simulation parameters
include("path/to/T1T2_voxel_phantom.jl")
include("path/to/BlochVoxelDictSimulation.jl")
sim_params = KomaMRICore.default_sim_params()
sim_params["sim_method"] =  BlochVoxelDict(N_spins_per_voxel=NisoTotal) # N_spins_per_voxel depends on the spins defined in the voxelphantom 
sim_params["Nthreads"] = 8        # define number of threads - depends on your machine
sim_params["return_type"] = "mat" # format for the output signal
sim_params["gpu"] = false         # GPU not supported yet

# system struct instance
sys = Scanner()

# 3.2 Generate list of phantoms - each voxel (phantom) has a unique T1-T2 combination and a unique ID (n)
phantom_list = []
for n in range(1,Ncombinations)
    push!(phantom_list, voxel_phantom(T1values[n],T2values[n], n)) 
end

# 3.3 sum list of phantoms and run the simulation!
phantom = sum(phantom_list)
raw = simulate(phantom, seq, sys; sim_params)


# 4. PLOT STUFF
# maybe you want to plot the results, e.g. plot the magnitude of the first 5 entries of the dictionary
sig = abs.(raw[:,1:5,1])
plot(sig)


# 5. SAVE RESULTS
# reshape combinations lists as an array
lists_temp = [[1e3*k for k in x] for x in params_combs] # in ms
lists_combinations = mapreduce(permutedims, vcat, lists_temp)

# save MATLAB struct named KomaDict with fields
#                          KomaDict.raw
#                          KomaDict.combinations
# you can also add formatting for a more descriptive name
save_folder = "path/to/save/"
matwrite(join([save_folder, "KomaDict.mat"]), Dict("GenDict" => (Combinations = lists_combinations, dictOn = raw))) 
