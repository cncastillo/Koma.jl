using TestItems, TestItemRunner

@run_package_tests filter=t_start->!(:skipci in t_start.tags)&&(:base in t_start.tags) #verbose=true

@testitem "Sequence" tags=[:base] begin
    @testset "Init" begin
        sys = Scanner()
        B1 = sys.B1; durRF = π/2/(2π*γ*B1) #90-degree hard excitation pulse
        EX = PulseDesigner.RF_hard(B1, durRF, sys; G=[0.0,0.0,0.0])
        @test dur(EX) ≈ durRF #RF length matches what is supposed to be

        #ACQ construction
        N = 101
        FOV = 23e-2
        EPI = PulseDesigner.EPI(FOV, N, sys)
        TE = 30e-3
        d1 = TE-dur(EPI)/2-dur(EX)
        d1 = d1 > 0 ? d1 : 0.0
        if d1 > 0 DELAY = Delay(d1) end

        #Sequence construction
        seq = d1 > 0 ? EX + DELAY + EPI : EX + EPI #Only add delay if d1 is positive (enough space)
        seq.DEF["TE"] = round(d1 > 0 ? TE : TE - d1, digits=4)*1e3
        @test dur(seq) ≈ dur(EX) + d1 + dur(EPI) #Sequence duration matches what is supposed to be
    end

    @testset "Rot_and_Concat" begin
        # Rotation 2D case
        A1, A2, A3, T, t = rand(5)
        s = Sequence([Grad(A1,T);
                      Grad(A2,T);
                      Grad(A3,T);;])
        θ = π*t
        R = rotz(θ)
        s2 = R*s #Matrix-Matrix{Grad} multiplication
        GR2 = R*s.GR.A #Matrix-vector multiplication
        @test s2.GR.A    ≈ GR2
        @test s.GR.T     == s2.GR.T
        @test s.GR.delay == s2.GR.delay
        @test s.GR.rise  == s2.GR.rise
        @test s.GR.fall  == s2.GR.fall
        # Rotation 3D case
        T, t1, t2, t3 = rand(4)
        N = 100
        GR = [Grad(rand(),T) for i=1:3, j=1:N]
        s = Sequence(GR)
        α, β, γ = π*t1, π*t2, π*t3
        Rx = rotx(α)
        Ry = roty(β)
        Rz = rotz(γ)
        R = Rx*Ry*Rz
        s2 = R*s #Matrix-Matrix{Grad} multiplication
        GR2 = R*s.GR.A #Matrix-vector multiplication
        @test s2.GR.A    ≈ GR2
        @test s.GR.T     == s2.GR.T
        @test s.GR.delay == s2.GR.delay
        @test s.GR.rise  == s2.GR.rise
        @test s.GR.fall  == s2.GR.fall
        # Concatenation of sequences
        A1, A2, A3, T1 = rand(4)
        s1 = Sequence([Grad(A1,T1);
                    Grad(A2,T1)],
                        [RF(A3,T1)])
        B1, B2, B3, T2 = rand(4)
        s2 = Sequence([Grad(B1,T2);
                    Grad(B2,T2)],
                        [RF(B3,T2)])
        s = s1 + s2
        @test s.GR.A ≈ [s1.GR.A s2.GR.A]
        @test s.RF.A ≈ [s1.RF.A s2.RF.A]
        @test s.ADC.N ≈ [s1.ADC.N ; s2.ADC.N]
    end

    @testset "Grad" begin
        #Testing gradient concatenation, breakes in some Julia versions
        A1, A2, T = rand(3)
        g1, g2 = Grad(A1,T), Grad(A2,T)
        GR = [g1;g2;;]
        GR2 = reshape([g1;g2],:,1)
        @test GR.A ≈ GR2.A

        #Sanity checks of contructors (A [T], T[s], rise[s], fall[s], delay[s])
        A, T = 0.1, 1e-3
        grad = Grad(A, T)

        A, T = rand(2)
        g1, g2 = Grad(A,T), Grad(A,T,0.0,0.0,0.0)
        @test g1 ≈ g2

        A, T, ζ = rand(3)
        g1, g2 = Grad(A,T,ζ), Grad(A,T,ζ,ζ,0.0)
        @test g1 ≈ g2

        A, T, delay, ζ = rand(4)
        g1, g2 = Grad(A,T,ζ,delay), Grad(A,T,ζ,ζ,delay)
        @test g1 ≈ g2

        # Test construction with shape function
        T, N = 1e-3, 100
        f = t -> sin(π*t / T)
        gradw = Grad(f, T, N)
        @test gradw.A ≈ f.(range(0.0, T; length=N))

        # Test Grad operations
        α = 3
        gradt = α * grad
        @test gradt.A ≈ α * grad.A
        gradt = grad * α
        @test gradt.A ≈ α * grad.A
        gradt = grad / α
        @test gradt.A ≈ grad.A / α
        grads = grad + gradt
        @test grads.A ≈ grad.A + gradt.A
        A1, A2, A3 = 0.1, 0.2, 0.3
        v1 = [Grad(A1,T); Grad(A2,T); Grad(A3,T)]
        v2 = [Grad(A2,T); Grad(A3,T); Grad(A1,T)]
        v3 = v1 + v2
        @test [v3[i].A for i=1:length(v3)] ≈ [v1[i].A + v2[i].A for i=1:length(v1)]
        gradr = grad - gradt
        @test gradr.A ≈ grad.A - gradt.A
        gradt = -grad
        @test gradt.A ≈ -grad.A
        vc = vcat(v1, v2)
        @test [vc[1,j].A for j=1:length(v1)] ≈ [v1[i].A for i=1:length(v1)]
        @test [vc[2,j].A for j=1:length(v2)] ≈ [v2[i].A for i=1:length(v2)]
        vc = vcat(v1, v2, v3)
        @test [vc[1,j].A for j=1:length(v1)] ≈ [v1[i].A for i=1:length(v1)]
        @test [vc[2,j].A for j=1:length(v2)] ≈ [v2[i].A for i=1:length(v2)]
        @test [vc[3,j].A for j=1:length(v3)] ≈ [v3[i].A for i=1:length(v3)]
        delay, rise, T, fall = 1e-6, 2e-6, 10e-3, 3e-6
        gr = Grad(A, T, rise, fall, delay)
        @test dur(gr) ≈ delay + rise + T + fall
        T1, T2, T3 = 1e-3, 2e-3, 3e-3
        vt = [Grad(A1,T1); Grad(A2,T2); Grad(A3,T3)]
        @test dur(vt) ≈ [maximum([T1, T2, T3])]

        # Just checking to ensure that show() doesn't get stuck and that it is covered
        show(IOBuffer(), "text/plain", grad)
        @test true

    end

    @testset "RF" begin
        #Testing gradient concatenation, breakes in some Julia versions
        A1, A2, T = rand(3)
        r1, r2 = RF(A1,T), RF(A2,T)
        R = [r1;r2;;]
        R2 = reshape([r1;r2],:,1)
        @test R.A ≈ R2.A

        #Sanity checks of constructors (A [T], T [s], Δf[Hz], delay [s])
        A, T = rand(2)
        r1, r2 = RF(A,T), RF(A,T,0.0,0.0)
        @test r1 ≈ r2

        A, T, Δf = rand(3)
        r1, r2 = RF(A,T,Δf), RF(A,T,Δf,0.0)
        @test r1 ≈ r2

        # Just checking to ensure that show() doesn't get stuck and that it is covered
        show(IOBuffer(), "text/plain", r1)
        @test true

        # Test Grad operations
        B1x, B1y, T = rand(3)
        A = B1x + im*B1y
        α = Complex(rand())
        rf = RF(A, T)
        rft = α * rf
        @test size(rf, 1) == 1
        @test rft.A ≈ α * rf.A
        @test dur(rf) ≈ rf.T
        B1x, B1y, B2x, B2y, B3x, B3y, T1, T2, T3 = rand(9)
        rf1, rf2, rf3 = RF(B1x + im*B1y, T1), RF(B1x + im*B1y, T2), RF(B3x + im*B3y, T3)
        rv = [rf1; rf2; rf3 ;;]
        @test dur(rv) ≈ maximum(dur.(rv); dims=1)

    end

    @testset "Delay" begin

        # Test delay construction
        T = 1e-3
        delay = Delay(T)
        @test delay.T ≈ T

        # Test delay construction error for negative values
        err = Nothing
        try Delay(-T) catch err end
        @test err isa ErrorException

        # Just checking to ensure that show() doesn't get stuck and that it is covered
        show(IOBuffer(), "text/plain", delay)
        @test true

        # Test addition of a delay to a sequence
        seq = Sequence([Grad(0.0, 0.0)])
        ds = delay + seq
        @test dur(ds[1]) ≈ delay.T && dur(ds[2]) ≈ .0
        sd = seq + delay
        @test dur(sd[1]) ≈ .0 && dur(sd[2]) ≈ delay.T

    end
    @testset "ADC" begin

        # Test ADC construction
        N, T, delay, Δf, ϕ  = 64, 1e-3, 2e-3, 1e-6, .25*π
        adc = ADC(N, T, delay, Δf, ϕ)

        adc1, adc2 = ADC(N, T), ADC(N,T,0,0,0)
        @test adc1 ≈ adc2

        adc1, adc2 = ADC(N, T, delay), ADC(N, T, delay, 0, 0)
        @test adc1 ≈ adc2

        adc1, adc2 = ADC(N, T, delay, Δf, ϕ), ADC(N, T, delay, Δf, ϕ)
        @test adc1 ≈ adc2

        # Test ADC construction errors for negative values
        err = Nothing
        try ADC(N, -T) catch err end
        @test err isa ErrorException
        try ADC(N, -T,  delay) catch err end
        @test err isa ErrorException
        try ADC(N,  T, -delay) catch err end
        @test err isa ErrorException
        try ADC(N, -T, -delay) catch err end
        @test err isa ErrorException
        try ADC(N, -T,  delay, Δf, ϕ) catch err end
        @test err isa ErrorException
        try ADC(N,  T, -delay, Δf, ϕ) catch err end
        @test err isa ErrorException
        try ADC(N, -T, -delay, Δf, ϕ) catch err end
        @test err isa ErrorException

        # Test ADC getproperties
        Nb, Tb, delayb, Δfb, ϕb  = 128, 2e-3, 4e-3, 2e-6, .125*π
        adb = ADC(Nb, Tb, delayb, Δfb, ϕb)
        adcs = [adc, adb]
        @test adcs.N ≈ [adc.N, adb.N] && adcs.T ≈ [adc.T, adb.T] && adcs.delay ≈ [adc.delay, adb.delay]
        @test adcs.Δf ≈ [adc.Δf, adb.Δf] && adcs.ϕ ≈ [adc.ϕ, adb.ϕ] && adcs.dur ≈ [adc.T + adc.delay, adb.T + adb.delay]

    end

    @testset "DiscreteSequence" begin
        path = joinpath(@__DIR__, "test_files")
        seq = PulseDesigner.EPI_example()
        sampling_params = KomaMRIBase.default_sampling_params()
        t, Δt = KomaMRIBase.get_variable_times(seq; Δt=sampling_params["Δt"], Δt_rf=sampling_params["Δt_rf"])
        seqd = KomaMRIBase.discretize(seq)
        i1, i2 = rand(1:Int(floor(0.5*length(seqd)))), rand(Int(ceil(0.5*length(seqd))):length(seqd))
        @test seqd[i1].t ≈ [t[i1]]
        @test seqd[i1:i2-1].t ≈ t[i1:i2]

        T, N = 1.0, 4
        seq = RF(1.0e-6, 1.0)
        seq += Sequence([Grad(1.0e-3, 1.0)])
        seq += ADC(N, 1.0)
        sampling_params = KomaMRIBase.default_sampling_params()
        sampling_params["Δt"], sampling_params["Δt_rf"] = T/N, T/N
        seqd1 = KomaMRIBase.discretize(seq[1]; sampling_params)
        seqd2 = KomaMRIBase.discretize(seq[2]; sampling_params)
        seqd3 = KomaMRIBase.discretize(seq[3]; sampling_params)
        # Block 1
        @test is_RF_on(seq[1]) == is_RF_on(seqd1)
        @test is_GR_on(seq[1]) == is_GR_on(seqd1)
        @test is_ADC_on(seq[1]) == is_ADC_on(seqd1)
        # Block 2
        @test is_RF_on(seq[2]) == is_RF_on(seqd2)
        @test is_GR_on(seq[2]) == is_GR_on(seqd2)
        @test is_ADC_on(seq[2]) == is_ADC_on(seqd2)
        # Block 3
        @test is_RF_on(seq[3]) == is_RF_on(seqd3)
        @test is_GR_on(seq[3]) == is_GR_on(seqd3)
        @test is_ADC_on(seq[3]) == is_ADC_on(seqd3)
        @test KomaMRIBase.is_GR_off(seqd) ==  !KomaMRIBase.is_GR_on(seqd)
        @test KomaMRIBase.is_RF_off(seqd) ==  !KomaMRIBase.is_RF_on(seqd)
        @test KomaMRIBase.is_ADC_off(seqd) == !KomaMRIBase.is_ADC_on(seqd)
    end

    @testset "SequenceFunctions" begin
        path = joinpath(@__DIR__, "test_files")
        seq = PulseDesigner.EPI_example()
        t, Δt = KomaMRIBase.get_variable_times(seq; Δt=1)
        t_adc =  KomaMRIBase.get_adc_sampling_times(seq)
        M2, M2_adc = KomaMRIBase.get_slew_rate(seq)
        M2eddy, M2eddy_adc = KomaMRIBase.get_eddy_currents(seq)
        Gx, Gy, Gz = KomaMRIBase.get_grads(seq, t)
        Gmx, Gmy, Gmz = KomaMRIBase.get_grads(seq, reshape(t, 1, :))
        @test reshape(Gmx, :, 1) ≈ Gx && reshape(Gmy, :, 1) ≈ Gy && reshape(Gmz, :, 1) ≈ Gz
        @test is_ADC_on(seq) == is_ADC_on(seq, t)
        @test is_RF_on(seq) == is_RF_on(seq, t)
        @test KomaMRIBase.is_Delay(seq) == !(is_GR_on(seq) || is_RF_on(seq) || is_ADC_on(seq))
        @test size(M2, 1) == length(Δt) && size(M2_adc, 1) == length(t_adc)
        @test size(M2eddy, 1) == length(Δt) && size(M2eddy_adc, 1) == length(t_adc)

        # Just checking to ensure that show() doesn't get stuck and that it is covered
        show(IOBuffer(), "text/plain", seq)
        @test true

        α = rand()
        c = α + im*rand()
        x = seq
        y = PulseDesigner.EPI_example()
        z = x + y
        @test z.GR.A ≈ [x.GR  y.GR].A && z.RF.A ≈ [x.RF y.RF].A && z.ADC.N ≈ [x.ADC; y.ADC].N
        z = x - y
        @test z.GR.A ≈ [x.GR  -y.GR].A
        z = -x
        @test z.GR.A ≈ -x.GR.A
        z = x * α
        @test z.GR.A ≈ α*x.GR.A
        z = α * x
        @test z.GR.A ≈ α*x.GR.A
        z = x * c
        @test z.RF.A ≈ c*x.RF.A
        z = c * x
        @test z.RF.A ≈ c*x.RF.A
        z = x / α
        @test z.GR.A ≈ x.GR.A/α
        @test size(y) == size(y.GR[1,:])
        z = x + x.GR[3,1]
        @test z.GR.A[1, end] ≈ x.GR[3,1].A
        z = x.GR[3,1] + x
        @test z.GR.A[1, 1] ≈ x.GR[3,1].A
        z = x + x.RF[1,1]
        @test z.RF.A[1, end] ≈ x.RF[1,1].A
        z = x.RF[1,1] + x
        @test z.RF.A[1, 1] ≈ x.RF[1,1].A
        z = x + x.ADC[3,1]
        @test z.ADC.N[end] ≈ x.ADC[3,1].N
        z = x.ADC[3,1] + x
        @test z.ADC.N[1] ≈ x.ADC[3,1].N
    end

end

@testitem "PulseDesigner" tags=[:base] begin
    @testset "RF_sinc" begin
        sys = Scanner()
        B1 = 23.4e-6 # For 90 deg flip angle
        Trf = 1e-3
        rf = PulseDesigner.RF_sinc(B1, Trf, sys; TBP=4)
        @test round(KomaMRIBase.get_flip_angles(rf)[1]) ≈ 90
    end
    @testset "Spiral" begin
        sys = Scanner()
        sys.Smax = 150    # [mT/m/ms]
        sys.Gmax = 500e-3 # [T/m]
        sys.GR_Δt = 4e-6  # [s]
        FOV = 0.2       # [m]
        N = 80          # Reconstructed image N×N
        Nint = 8
        λ = 2.1
        spiral = PulseDesigner.spiral_base(FOV, N, sys; λ=λ, BW=120e3, Nint)
        # Look at the k_space generated
        @test spiral(0).DEF["λ"] ≈ λ
    end
    @testset "Radial" begin
        sys = Scanner()
        N = 80
        Nspokes = ceil(Int64, π/2 * N ) #Nyquist in the radial direction
        FOV = 0.2
        spoke = PulseDesigner.radial_base(FOV, N, sys)
        @test spoke.DEF["Δθ"] ≈ π / Nspokes
    end
    @testset "design" begin
        # trapezoid
        duration = 10
        amplitude = 5
        grad = PulseDesigner.trapezoid(; duration, amplitude)
        seq = Sequence()
        seq += grad
        @test true

        # arbitrary_grad
        t = range(0, 1; length=20)
        waveform = sin.(2π * t)
        grad = PulseDesigner.arbitrary_grad(waveform)
        seq = Sequence()
        seq += grad
        @test true

        # block_pulse
        flip_angle = π / 2
        duration = 0.001
        rf, delay = PulseDesigner.block_pulse(flip_angle; duration)
        seq = Sequence()
        seq += rf
        @test true

        # sinc_pulse
        flip_angle = π / 2
        duration = 0.001
        slice_thickness = 0.001
        time_bw_product = 4
        rf, gz, gzr, dly = PulseDesigner.sinc_pulse(flip_angle; duration, time_bw_product, slice_thickness)
        seq = Sequence([Grad(0,0); Grad(0,0); gz;;], [rf;;])
        seq += Sequence([Grad(0,0); Grad(0,0); gzr;;])
        @test true

        # arbitrary_rf
        flip = π / 2
        slice_thickness = 0.001
        time_bw_product = 4
        t = range(0, 0.5; length=20)
        signal = sin.(2π * t)
        rf, gz, gzr, dly = PulseDesigner.arbitrary_rf(signal, flip; time_bw_product, slice_thickness)
        seq = Sequence([Grad(0,0); Grad(0,0); gz;;], [rf;;])
        seq += Sequence([Grad(0,0); Grad(0,0); gzr;;])
        @test true

        # make_adc
        num_samles = 10
        duration = 0.001
        adc = PulseDesigner.make_adc(num_samles; duration)
        seq = Sequence()
        seq += adc
        @test true
    end
end

@testitem "Motion" tags=[:base] begin
    @testset "Constructors" begin
        action = Rotate(10.0, 20.0, 40.0)
        spins = AllSpins()
        # TimeCurve constructors
        time = TimeRange(t_start=0.0, t_end=1.0)
        time = Periodic(period=1.0, asymmetry=0.5)
        time = TimeCurve([0.0, eps()], [0.0, 1.0])
        # Motion constructors
        m = Motion(action, time, spins)
        @test Motion(action) == m
        @test Motion(action, time) == m
        @test Motion(action, spins) == m
        # MotionList
        @test MotionList() == NoMotion()
        @test MotionList(m) == m
    end
    @testset "Subset" begin
        rotate = Rotate(10.0, 20.0, 40.0)
        translate = Translate(0.1, 0.2, 0.3)
        time = TimeRange(0.0, eps())
        spins = AllSpins()
        rng = rng = 1:2:5
        # NoMotion
        nm = NoMotion()
        @test nm[rng] == nm
        # Motion
        m = Motion(rotate, time, spins)
        @test m[rng] == m
        # MotionList 
        ml = MotionList(
            Motion(rotate, time, spins),
            Motion(translate, time, spins)
        )
        @test ml[rng] == ml 
    end
    # Spin Positions
    @testset "NoMotion" begin
        ph = Phantom(x=[1.0, 2.0], y=[1.0, 2.0])
        t_start=0.0; t_end=1.0 
        t = collect(range(t_start, t_end, 11))
        xt, yt, zt = get_spin_coords(ph.motion, ph.x, ph.y, ph.z, t')
        @test xt == ph.x
        @test yt == ph.y
        @test zt == ph.z
    end
    @testset "Translate" begin
        ph = Phantom(x=[1.0], y=[1.0])
        t_start=0.0; t_end=1.0 
        t = collect(range(t_start, t_end, 11))
        dx, dy, dz = [1.0, 0.0, 0.0]
        vx, vy, vz = [dx, dy, dz] ./ (t_end - t_start)
        translation = Translate(dx, dy, dz, TimeRange(t_start, t_end))
        xt, yt, zt = get_spin_coords(translation, ph.x, ph.y, ph.z, t')
        @test xt == ph.x .+ vx.*t'
        @test yt == ph.y .+ vy.*t'
        @test zt == ph.z .+ vz.*t'
    end
    @testset "PeriodicTranslate" begin
        ph = Phantom(x=[1.0], y=[1.0])
        t_start=0.0; t_end=1.0 
        t = collect(range(t_start, t_end, 11))
        period = 2.0
        asymmetry = 0.5
        dx, dy, dz = [1.0, 0.0, 0.0]
        vx, vy, vz = [dx, dy, dz] ./ (t_end - t_start)
        periodictranslation = Translate(dx, dy, dz, Periodic(period=period, asymmetry=asymmetry))
        xt, yt, zt = get_spin_coords(periodictranslation, ph.x, ph.y, ph.z, t')
        @test xt == ph.x .+ vx.*t'
        @test yt == ph.y .+ vy.*t'
        @test zt == ph.z .+ vz.*t'
    end
    @testset "Rotate" begin
        ph = Phantom(x=[1.0], y=[1.0])
        t_start=0.0; t_end=1.0 
        t = collect(range(t_start, t_end, 11))
        pitch = 45.0
        roll = 0.0
        yaw = 45.0
        rotation = Rotate(pitch, roll, yaw, TimeRange(t_start, t_end))
        xt, yt, zt = get_spin_coords(rotation, ph.x, ph.y, ph.z, t')
        r = vcat(ph.x, ph.y, ph.z)
        R = rotz(π*yaw/180) * roty(π*roll/180) * rotx(π*pitch/180)
        rot_x, rot_y, rot_z = R*r
        @test xt[end ,end] ≈ rot_x
        @test yt[end ,end] ≈ rot_y
        @test zt[end ,end] ≈ rot_z
    end
    @testset "PeriodicRotation" begin
        ph = Phantom(x=[1.0], y=[1.0])
        t_start=0.0; t_end=1.0 
        t = collect(range(t_start, t_end, 11))
        period = 2.0
        asymmetry = 0.5
        pitch = 45.0
        roll = 0.0
        yaw = 45.0
        periodicrotation = Rotate(pitch, roll, yaw, Periodic(period=period, asymmetry=asymmetry))
        xt, yt, zt = get_spin_coords(periodicrotation, ph.x, ph.y, ph.z, t')
        r = vcat(ph.x, ph.y, ph.z)
        R = rotz(π*yaw/180) * roty(π*roll/180) * rotx(π*pitch/180)
        rot_x, rot_y, rot_z = R*r
        @test xt[end ,end] ≈ rot_x
        @test yt[end ,end] ≈ rot_y
        @test zt[end ,end] ≈ rot_z
    end
    @testset "HeartBeat" begin
        ph = Phantom(x=[1.0], y=[1.0])
        t_start=0.0; t_end=1.0 
        t = collect(range(t_start, t_end, 11))
        circumferential_strain = -0.1
        radial_strain = 0.0
        longitudinal_strain = -0.1
        heartbeat = HeartBeat(circumferential_strain, radial_strain, longitudinal_strain, TimeRange(t_start, t_end))
        xt, yt, zt = get_spin_coords(heartbeat, ph.x, ph.y, ph.z, t')
        r = sqrt.(ph.x .^ 2 + ph.y .^ 2)
        θ = atan.(ph.y, ph.x)
        @test xt[:,end] == ph.x .* (1 .+ circumferential_strain * maximum(r) .* cos.(θ))
        @test yt[:,end] == ph.y .* (1 .+ circumferential_strain * maximum(r) .* sin.(θ))
        @test zt[:,end] == ph.z .* (1 .+ longitudinal_strain)
    end
    @testset "PeriodicHeartBeat" begin
        ph = Phantom(x=[1.0], y=[1.0])
        t_start=0.0; t_end=1.0 
        t = collect(range(t_start, t_end, 11))
        period = 2.0
        asymmetry = 0.5
        circumferential_strain = -0.1
        radial_strain = 0.0
        longitudinal_strain = -0.1
        periodicheartbeat = HeartBeat(circumferential_strain, radial_strain, longitudinal_strain, Periodic(period=period, asymmetry=asymmetry))
        xt, yt, zt = get_spin_coords(periodicheartbeat, ph.x, ph.y, ph.z, t')
        r = sqrt.(ph.x .^ 2 + ph.y .^ 2)
        θ = atan.(ph.y, ph.x)
        @test xt[:,end] == ph.x .* (1 .+ circumferential_strain * maximum(r) .* cos.(θ))
        @test yt[:,end] == ph.y .* (1 .+ circumferential_strain * maximum(r) .* sin.(θ))
        @test zt[:,end] == ph.z .* (1 .+ longitudinal_strain)
    end
    @testset "Path" begin
        # 1 spin
        ph = Phantom(x=[1.0], y=[1.0])
        Ns = length(ph)
        t_start = 0.0
        t_end = 1.0
        Nt = 10
        dx = rand(Ns, Nt)
        dy = rand(Ns, Nt)
        dz = rand(Ns, Nt)
        arbitrarymotion = Path(dx, dy, dz, TimeRange(t_start, t_end))
        t = range(t_start, t_end, Nt)
        xt, yt, zt = get_spin_coords(arbitrarymotion, ph.x, ph.y, ph.z, t')
        @test xt == ph.x .+ dx
        @test yt == ph.y .+ dy
        @test zt == ph.z .+ dz
        # More than 1 spin
        ph = Phantom(x=[1.0, 2.0], y=[1.0, 2.0])
        Ns = length(ph)
        t_start = 0.0
        t_end = 1.0
        Nt = 10
        dx = rand(Ns, Nt)
        dy = rand(Ns, Nt)
        dz = rand(Ns, Nt)
        arbitrarymotion = Path(dx, dy, dz, TimeRange(t_start, t_end))
        t = range(t_start, t_end, Nt)
        xt, yt, zt = get_spin_coords(arbitrarymotion, ph.x, ph.y, ph.z, t')
        @test xt == ph.x .+ dx
        @test yt == ph.y .+ dy
        @test zt == ph.z .+ dz
    end
    @testset "FlowPath" begin
        # 1 spin
        ph = Phantom(x=[1.0], y=[1.0])
        Ns = length(ph)
        t_start = 0.0
        t_end = 1.0
        Nt = 10
        dx = rand(Ns, Nt)
        dy = rand(Ns, Nt)
        dz = rand(Ns, Nt)
        arbitrarymotion = FlowPath(dx, dy, dz, Bool.(zeros(Ns, Nt)), TimeRange(t_start, t_end))
        t = range(t_start, t_end, Nt)
        xt, yt, zt = get_spin_coords(arbitrarymotion, ph.x, ph.y, ph.z, t')
        @test xt == ph.x .+ dx
        @test yt == ph.y .+ dy
        @test zt == ph.z .+ dz
        # More than 1 spin
        ph = Phantom(x=[1.0, 2.0], y=[1.0, 2.0])
        Ns = length(ph)
        t_start = 0.0
        t_end = 1.0
        Nt = 10
        dx = rand(Ns, Nt)
        dy = rand(Ns, Nt)
        dz = rand(Ns, Nt)
        arbitrarymotion = FlowPath(dx, dy, dz, Bool.(zeros(Ns, Nt)), TimeRange(t_start, t_end))
        t = range(t_start, t_end, Nt)
        xt, yt, zt = get_spin_coords(arbitrarymotion, ph.x, ph.y, ph.z, t')
        @test xt == ph.x .+ dx
        @test yt == ph.y .+ dy
        @test zt == ph.z .+ dz
    end
    @testset "Translate + Rotate" begin
        ph = Phantom(x=[1.0], y=[1.0])
        t_start=0.0; t_end=1.0 
        t = collect(range(t_start, t_end, 11))
        # Translate
        dx, dy, dz = [1.0, 0.0, 0.0]
        vx, vy, vz = [dx, dy, dz] ./ (t_end - t_start)
        translation = Translate(dx, dy, dz, TimeRange(t_start, t_end))
        # Rotate
        pitch, roll, yaw = [45.0, 0.0, 45.0]
        rotation = Rotate(pitch, roll, yaw, TimeRange(t_start, t_end))
        r = vcat(ph.x, ph.y, ph.z)
        R = rotz(π*yaw/180) * roty(π*roll/180) * rotx(π*pitch/180)
        rot_x, rot_y, rot_z = R*r
        # Combination into a MotionList
        motion = MotionList(translation, rotation)
        xt, yt, zt = get_spin_coords(motion, ph.x, ph.y, ph.z, t')
        @test xt[end ,end] ≈ rot_x + vx*t[end]
        @test yt[end ,end] ≈ rot_y + vy*t[end]
        @test zt[end ,end] ≈ rot_z + vz*t[end]        
    end
end

@testitem "Phantom" tags = [:base] begin
    using Suppressor
    # Phantom Struct Fields
    name = "Bulks"
    x = [-2e-3; -1e-3; 0.0; 1e-3; 2e-3]
    y = [-4e-3; -2e-3; 0.0; 2e-3; 4e-3]
    z = [-6e-3; -3e-3; 0.0; 3e-3; 6e-3]
    ρ = [0.2; 0.4; 0.6; 0.8; 1.0]
    T1 = [0.9; 0.9; 0.5; 0.25; 0.4]
    T2 = [0.09; 0.05; 0.04; 0.07; 0.005]
    T2s = [0.1; 0.06; 0.05; 0.08; 0.015]
    Δw = [-2e-6; -1e-6; 0.0; 1e-6; 2e-6]
    Dλ1 = [-4e-6; -2e-6; 0.0; 2e-6; 4e-6]
    Dλ2 = [-6e-6; -3e-6; 0.0; 3e-6; 6e-6]
    Dθ = [-8e-6; -4e-6; 0.0; 4e-6; 8e-6]
    # Motion
    Ns = length(x)
    Nt = 3
    t_start = 0.0
    t_end = 1.0
    translate = Translate(0.05, 0.05, 0.0, Periodic(period=0.5, asymmetry=0.5))
    rotate = Rotate(0.0, 0.0, 90.0, TimeRange(t_start=0.05, t_end=0.5), SpinRange(1:3))
    path = Path(0.01 .* rand(Ns, Nt), 0.01 .* rand(Ns, Nt), 0.01 .* rand(Ns, Nt), TimeRange(t_start, t_end), SpinRange(2:2:4))
    @testset "Comparison" begin
        obj1 = Phantom(name=name, x=x, y=y, z=z, ρ=ρ, T1=T1, T2=T2, T2s=T2s, Δw=Δw, Dλ1=Dλ1, Dλ2=Dλ2, Dθ=Dθ, motion=MotionList(translate, rotate))
        obj2 = Phantom(name=name, x=x, y=y, z=z, ρ=ρ, T1=T1, T2=T2, T2s=T2s, Δw=Δw, Dλ1=Dλ1, Dλ2=Dλ2, Dθ=Dθ, motion=MotionList(translate, rotate))
        @test obj1 == obj2
        obj2.x .+= 1e-10
        @test obj1 ≈ obj2
    end
    @testset "Size and Length" begin
        obj1 = Phantom(name=name, x=x, y=y, z=z, ρ=ρ, T1=T1, T2=T2, T2s=T2s, Δw=Δw, Dλ1=Dλ1, Dλ2=Dλ2, Dθ=Dθ, motion=MotionList(translate, rotate))
        @test size(obj1) == size(ρ)
        @test length(obj1) == length(ρ)
    end
    @testset "Subset" begin 
        motion = MotionList(translate, rotate)
        obj1 = Phantom(name, x, y, z, ρ, T1, T2, T2s, Δw, Dλ1, Dλ2, Dθ, motion)
        rng = 1:2:5
        obj2 = Phantom(
            name, x[rng], y[rng], z[rng], 
            ρ[rng], T1[rng], T2[rng], T2s[rng], 
            Δw[rng], Dλ1[rng], Dλ2[rng], Dθ[rng], 
            motion[rng]
        )
        # Phantom subset
        @test obj1[rng] == obj2
        @test @view(obj1[rng]) == obj2
        # Phantom view
        obj_view = @view(obj1[rng])
        obj_view.ρ .= 0.0
        @test obj_view.ρ == obj1[rng].ρ
        # BitVector range
        obj3 = copy(obj1)
        rng = obj1.x .> 0
        obj1.motion = Translate(5e-4, 6e-4, 7e-4, TimeRange(0.0, 1.0), SpinRange(rng))
        obj3.motion = Translate(5e-4, 6e-4, 7e-4, TimeRange(0.0, 1.0), SpinRange(1:length(obj3)))
        @test obj1[rng] == obj3[rng]
        @test obj1[rng].motion == obj3.motion[rng]
    end
    @testset "Addition" begin
        obj1 = Phantom(name=name, x=x, y=y, z=z, ρ=ρ, T1=T1, T2=T2, T2s=T2s, Δw=Δw, Dλ1=Dλ1, Dλ2=Dλ2, Dθ=Dθ)
        rng = 1:2:5
        obj2 = obj1[rng]
        oba = Phantom(
            name, [x; x[rng]], [y; y[rng]], [z; z[rng]], 
            [ρ; ρ[rng]], [T1; T1[rng]], [T2; T2[rng]], [T2s; T2s[rng]], 
            [Δw; Δw[rng]], [Dλ1; Dλ1[rng]], [Dλ2; Dλ2[rng]], [Dθ; Dθ[rng]], 
            vcat(obj1.motion, obj2.motion, length(obj1), length(obj2))
        )
        # NOTE: these vcat methods must be simplified once the Vector{<:Motion} approach is accomplished: 
        # https://github.com/JuliaHealth/KomaMRI.jl/issues/480
        # NoMotion + NoMotion
        @test obj1 + obj2 == oba
        # NoMotion + MotionList
        obj2.motion = MotionList(translate, rotate)
        oba.motion = vcat(obj1.motion, obj2.motion, length(obj1), length(obj2))
        @test obj1 + obj2 == oba
        # MotionList + NoMotion
        obj1.motion = MotionList(translate, rotate)
        obj2.motion = NoMotion()
        oba.motion = vcat(obj1.motion, obj2.motion, length(obj1), length(obj2))
        @test obj1 + obj2 == oba
        # NoMotion + Motion
        obj1.motion = NoMotion()
        obj2.motion = translate
        oba.motion = vcat(obj1.motion, obj2.motion, length(obj1), length(obj2))
        @test obj1 + obj2 == oba
        # Motion + NoMotion
        obj1.motion = translate
        obj2.motion = NoMotion()
        oba.motion = vcat(obj1.motion, obj2.motion, length(obj1), length(obj2))
        @test obj1 + obj2 == oba
        # MotionList + MotionList
        obj1.motion = MotionList(translate, rotate)
        obj2.motion = MotionList(translate, rotate)
        oba.motion = vcat(obj1.motion, obj2.motion, length(obj1), length(obj2))
        @test obj1 + obj2 == oba
        # Motion + Motion
        obj1.motion = translate
        obj2.motion = rotate
        oba.motion = vcat(obj1.motion, obj2.motion, length(obj1), length(obj2))
        @test obj1 + obj2 == oba
        # Motion + MotionList
        obj1.motion = translate
        obj2.motion = MotionList(translate, rotate)
        oba.motion = vcat(obj1.motion, obj2.motion, length(obj1), length(obj2))
        @test obj1 + obj2 == oba
        # MotionList + Motion
        obj1.motion = MotionList(translate, rotate)
        obj2.motion = translate
        oba.motion = vcat(obj1.motion, obj2.motion, length(obj1), length(obj2))
        @test obj1 + obj2 == oba
    end 
    @testset "Scalar multiplication" begin
        obj1 = Phantom(name=name, x=x, y=y, z=z, ρ=ρ, T1=T1, T2=T2, T2s=T2s, Δw=Δw, Dλ1=Dλ1, Dλ2=Dλ2, Dθ=Dθ, motion=MotionList(translate, rotate))
        c = 7
        obc = Phantom(name=name, x=x, y=y, z=z, ρ=c*ρ, T1=T1, T2=T2, T2s=T2s, Δw=Δw, Dλ1=Dλ1, Dλ2=Dλ2, Dθ=Dθ, motion=MotionList(translate, rotate))
        @test c * obj1 == obc
    end
    @testset "Brain Phantom 2D" begin
        ph = brain_phantom2D()
        @test ph.name == "brain2D_axial"
        @test KomaMRIBase.get_dims(ph) == Bool[1, 1, 0]
    end
    @testset "Brain Phantom 3D" begin
        ph = brain_phantom3D()
        @test ph.name == "brain3D"
        @test KomaMRIBase.get_dims(ph) == Bool[1, 1, 1]
    end
    @testset "Pelvis Phantom" begin
        ph = pelvis_phantom2D()
        @test ph.name == "pelvis2D"
        @test KomaMRIBase.get_dims(ph) == Bool[1, 1, 0]
    end
    @testset "Heart Phantom" begin
        ph = heart_phantom()
        @test ph.name == "LeftVentricle"
        @test KomaMRIBase.get_dims(ph) == Bool[1, 1, 0]
    end
end

@testitem "Scanner" tags=[:base] begin
    B0, B1, Gmax, Smax = 1.5, 10e-6, 60e-3, 500
    ADC_Δt, seq_Δt, GR_Δt, RF_Δt = 2e-6, 1e-5, 1e-5, 1e-6
    RF_ring_down_T, RF_dead_time_T, ADC_dead_time_T = 20e-6, 100e-6, 10e-6
    sys = Scanner(B0, B1, Gmax, Smax, ADC_Δt, seq_Δt, GR_Δt, RF_Δt, RF_ring_down_T, RF_dead_time_T, ADC_dead_time_T)
    @test sys.B0 ≈ B0 && sys.B1 ≈ B1 && sys.Gmax ≈ Gmax && sys.Smax ≈ Smax
end

@testitem "TrapezoidalIntegration" tags=[:base] begin
    dt = Float64[1 1 1 1]
    x  = Float64[0 1 2 1 0]
    @test KomaMRIBase.trapz(dt, x)[1] ≈ 4 #Triangle area = bh/2, with b = 4 and h = 2
    @test KomaMRIBase.cumtrapz(dt, x) ≈ [0.5 2 3.5 4]
end
