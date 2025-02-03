using TestItems, TestItemRunner

@run_package_tests filter=ti->!(:skipci in ti.tags)&&(:koma in ti.tags) #verbose=true

# include("../KomaMRICore/test/runtests.jl")
# include("../KomaMRIPlots/test/runtests.jl")

@testitem "MRIReco recon" tags=[:koma] begin
    #Sanity check 1
    A = rand(5,5,3)
    B = KomaMRI.fftc(KomaMRI.ifftc(A))
    @test A ≈ B
    
    #Sanity check 2
    B = KomaMRI.ifftc(KomaMRI.fftc(A))
    @test A ≈ B

    #MRIReco.jl
    path = @__DIR__
    fraw = ISMRMRDFile(path*"/test_files/Koma_signal.mrd")
    raw = RawAcquisitionData(fraw)
    acq = AcquisitionData(raw)

    @testset "MRIReco_direct" begin
        Nx, Ny = raw.params["reconSize"][1:2]
        recParams = Dict{Symbol,Any}(:reco=>"direct", :reconSize=>(Nx,Ny), :densityWeighting=>true)
        img = reconstruction(acq, recParams)
        @test true                #If the previous line fails the test will fail
    end

    #Test MRIReco regularized recon (with a λ)
    @testset "MRIReco_standard" begin
        #???
    end

end

@testitem "KomaUI" tags=[:koma] begin

    using Blink
    # Opens KomaUI

    @unsafe_blink
    w = KomaUI(return_window=true)
    @testset "Open UI" begin
        @test "index" == @js w document.getElementById("content").dataset.content
    end

    @testset "PulsesGUI" begin
        @js w document.getElementById("button_pulses_seq").click()
        @test "sequence" == @js w document.getElementById("content").dataset.content

        @js w document.getElementById("button_pulses_kspace").click()
        @test "kspace" == @js w document.getElementById("content").dataset.content

        @js w document.getElementById("button_pulses_M0").click()
        @test "m0" == @js w document.getElementById("content").dataset.content

        @js w document.getElementById("button_pulses_M1").click()
        @test "m1" == @js w document.getElementById("content").dataset.content

        @js w document.getElementById("button_pulses_M2").click()
        @test "m2" == @js w document.getElementById("content").dataset.content
    end

    @testset "PhantomGUI" begin
        @js w document.getElementById("button_phantom").click()
        @test "phantom" == @js w document.getElementById("content").dataset.content
    end

    @testset "ParamsGUI" begin
        @js w document.getElementById("button_scanner").click()
        @test "scanneparams" == @js w document.getElementById("content").dataset.content

        @js w document.getElementById("button_sim_params").click()
        @test "simparams" == @js w document.getElementById("content").dataset.content

        @js w document.getElementById("button_rec_params").click()
        @test "recparams" == @js w document.getElementById("content").dataset.content
    end

    @testset "Simulation" begin
        @js w document.getElementById("simulate!").click()
        @test "sig" == @js w document.getElementById("content").dataset.content
    end

    @testset "SignalGUI" begin
        @js w document.getElementById("button_sig").click()
        @test "sig" == @js w document.getElementById("content").dataset.content
    end

    @testset "Reconstruction" begin
        @js w document.getElementById("recon!").click()
        @test "absi" == @js w document.getElementById("content").dataset.content
    end

    @testset "ReconGUI" begin
        @js w document.getElementById("button_reconstruction_absI").click()
        @test "absi" == @js w document.getElementById("content").dataset.content

        @js w document.getElementById("button_reconstruction_angI").click()
        @test "angi" == @js w document.getElementById("content").dataset.content

        @js w document.getElementById("button_reconstruction_absK").click()
        @test "absk" == @js w document.getElementById("content").dataset.content
    end

    @testset "ExportToMAT" begin
        @js w document.getElementById("button_matfolder").click()
        @test "matfolder" == @js w document.getElementById("content").dataset.content

        @js w document.getElementById("button_matfolderseq").click()
        @test "matfolderseq" == @js w document.getElementById("content").dataset.content

        @js w document.getElementById("button_matfolderpha").click()
        @test "matfolderpha" == @js w document.getElementById("content").dataset.content

        @js w document.getElementById("button_matfoldersca").click()
        @test "matfoldersca" == @js w document.getElementById("content").dataset.content

        @js w document.getElementById("button_matfolderraw").click()
        @test "matfolderraw" == @js w document.getElementById("content").dataset.content

        @js w document.getElementById("button_matfolderima").click()
        @test "matfolderima" == @js w document.getElementById("content").dataset.content
    end

    if !isnothing(w)
        close(w)
    end

end
