using KomaMRIPlots, KomaMRICore, Distributions

X = collect(1:128) 
Y = collect(1:128)

positions = [(32, 32), (64, 32), (96, 32), (32, 64), (64, 64), (96, 64), (32, 96), (64, 96), (96, 96)]

r = 11 # 22mm diameter per vial
M = zeros(UInt32, 128, 128, 9) 
for i in 1:9
    pos_x, pos_y = positions[i]
    M[:,:,i] .= i.*((X .- pos_x).^2 .+ (Y' .- pos_y).^2 .<= r.^2)
end

M = sum(M, dims=3)
M = convert(Array{UInt32, 3}, M)
class = Matrix{UInt32}(reshape(M, 128, 128))'

B0 = 0.55         # 0.55 T
Niso = 200        # 200 isochromats

# Define spin position arrays
Δx = (122/117)*.1e-2        # 10mm
Δz_voxel = 10e-3            # 10 mm
M, N = size(class)          # Number of spins in x and y
FOVx = (M-1)*Δx             # Field of view in x
FOVy = (N-1)*Δx             # Field of view in y
x = -FOVx/2:Δx:FOVx/2       # x spin coordinates vector
y = -FOVy/2:Δx:FOVy/2       # y spin coordinates vector
x, y = x .+ y'*0, x*0 .+ y' # x and y grid points


# Define the proton density array
ρ = (class.== 0)*1 .+    # Matrix
    (class.== 1)*1 .+    # vial 1
    (class.== 2)*1 .+    # vial 2
    (class.== 3)*1 .+    # vial 3 
    (class.== 4)*1 .+    # vial 4
    (class.== 5)*1 .+    # vial 5
    (class.== 6)*1 .+    # vial 6
    (class.== 7)*1 .+    # vial 7
    (class.== 8)*1 .+    # vial 8
    (class.== 9)*1       # vial 9

# Define the T1 decay array
import Random 
Random.seed!(1234)
T1 = (class.==0)*810 .+ (class.==0).*Random.rand(Uniform(-16,16),M,N) .+  # Matrix
     (class.==1)*424 .+ (class.==1).*Random.rand(Uniform(-4,4),M,N) .+    # vial 1
     (class.==2)*993 .+ (class.==2).*Random.rand(Uniform(-4,4),M,N) .+    # vial 2
     (class.==3)*450 .+ (class.==3).*Random.rand(Uniform(-5,5),M,N) .+    # vial 3
     (class.==4)*543 .+ (class.==4).*Random.rand(Uniform(-3,3),M,N) .+    # vial 4
     (class.==5)*1185 .+ (class.==5).*Random.rand(Uniform(-7,7),M,N) .+   # vial 5
     (class.==6)*1460 .+ (class.==6).*Random.rand(Uniform(-15,15),M,N) .+ # vial 6
     (class.==7)*299 .+ (class.==7).*Random.rand(Uniform(-3,3),M,N) .+    # vial 7
     (class.==8)*751 .+ (class.==8).*Random.rand(Uniform(-4,4),M,N) .+    # vial 8
     (class.==9)*264 .+ (class.==9).*Random.rand(Uniform(-3,3),M,N)       # vial 9

# Define the T2 decay array
T2 = (class.==0)*125 .+ (class.==0).*Random.rand(Uniform(-20,20),M,N) .+  # Matrix
     (class.==1)*47 .+ (class.==1).*Random.rand(Uniform(-1,1),M,N) .+     # vial 1
     (class.==2)*53 .+ (class.==2).*Random.rand(Uniform(-1.5,1.5),M,N) .+ # vial 2
     (class.==3)*196 .+ (class.==3).*Random.rand(Uniform(-5,5),M,N) .+    # vial 3
     (class.==4)*49 .+ (class.==4).*Random.rand(Uniform(-1,1),M,N) .+     # vial 4
     (class.==5)*53 .+ (class.==5).*Random.rand(Uniform(-1,1),M,N) .+     # vial 5
     (class.==6)*238 .+ (class.==6).*Random.rand(Uniform(-5,5),M,N) .+    # vial 6
     (class.==7)*47 .+ (class.==7).*Random.rand(Uniform(-1,1),M,N) .+     # vial 7
     (class.==8)*53 .+ (class.==8).*Random.rand(Uniform(-1,1),M,N) .+     # vial 8
     (class.==9)*174 .+ (class.==9).*Random.rand(Uniform(-3,3),M,N)       # vial 9

# Time scaling factor
T1 = T1*1e-3
T2 = T2*1e-3

# Define the phantom object
obj = Phantom{Float64}(
    name = "custom_T1MES_lowfield",
	x = x[ρ.!=0],
	y = y[ρ.!=0],
	z = 0*x[ρ.!=0],
	ρ = ρ[ρ.!=0],
	T1 = T1[ρ.!=0],
	T2 = T2[ρ.!=0],
)


