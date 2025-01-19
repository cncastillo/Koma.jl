using KomaMRI


B0 = 0.55         # 0.55 T
Niso = 1          # number of spins in x and y directions 
Niso_z = 101      # number of spins in z direction
Δx_voxel = 2e-3   # x voxel dim 
Δy_voxel = 2e-3   # y voxel dim
Δz_voxel = 10e-3  # z voxel dim

NisoTotal = Niso*Niso*Niso_z

if Niso == 1
    dx = 0 
    dy = 0
else
    dx = Array(range(-Δx_voxel/2, Δx_voxel/2, Niso))
    dy = Array(range(-Δy_voxel/2, Δy_voxel/2, Niso))
end

dz = Array(range(-Δz_voxel/2, Δz_voxel/2, Niso_z))
coords = vec(collect(Iterators.product(dx, dy, dz)))

function voxel_phantom(T1v, T2v, idx)
    obj = Phantom{Float64}(x=[x[1] for x in coords], 
                           y=[x[2] for x in coords],
                           z=[x[3] for x in coords],
                           T1=T1v*ones(Niso*Niso*Niso_z), 
                           T2=T2v*ones(Niso*Niso*Niso_z),
                           Dλ1=idx*ones(Niso*Niso*Niso_z))
    return obj
end

