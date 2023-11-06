include("ui/ExportMATFunctions.jl")

"""
    w = setup_blink_window(path::String)

Returns the blink window with some custom styles and js logic.
"""
function setup_blink_window(path::String; darkmode=true, frame=true, dev_tools=false, blink_show=true)
    ## ASSETS
    assets = AssetRegistry.register(joinpath(dirname(path), "assets"))
    scripts = AssetRegistry.register(dirname(path*"/ui/scripts/"))
    css = AssetRegistry.register(dirname(path*"/ui/css/"))
    # Assets
    logo = joinpath(assets, "logo-dark.svg")
    icon = joinpath(assets, "icon.svg")
    # Apparently Blink requires an assets folder in a chiled route of where is launched
    icon_png = path*"/ui/assets/logo-icon.png"
    if !isfile(icon_png)
        cp(joinpath([dirname(path), "assets", "logo-icon.png"]), path*"/ui/assets/logo-icon.png")
    end
    # JS
    bsjs = joinpath(scripts, "bootstrap.bundle.min.js") #this already has Popper
    bscss = joinpath(css,"bootstrap.min.css")
    bsiconcss = joinpath(css,"bootstrap-icons.css")
    jquery = joinpath(scripts,"jquery-3.4.1.slim.min.js")
    # mathjaxsetup = joinpath(scripts, "mathjaxsetup.js")
    # KaTeX
    katexrender = joinpath(scripts, "auto-render.min.js")
    katexjs = joinpath(scripts,"katex.min.js")
    katexcss = joinpath(css,"katex.min.css")
    # User defined JS and CSS
    customcss = joinpath(css,"custom.css")
    customjs = joinpath(scripts,"custom.js")
    customjs2 = joinpath(scripts,"custom2.js")
    sidebarcss = joinpath(css,"sidebars.css")
    # Custom icons
    icons = joinpath(css,"icons.css")
    ## WINDOW
    w = Blink.Window(Dict(
        "title"=>"KomaUI",
        "autoHideMenuBar"=>true,
        "frame"=>frame, #removes title bar
        "node-integration" => true,
        :icon=>icon_png,
        "width"=>1200,
        "height"=>800,
        "webPreferences" => Dict("devTools" => dev_tools),
        :show=>blink_show
        ),async=false);
    ## NAV BAR
    sidebar = open(f->read(f, String), path*"/ui/html/sidebar.html")
    sidebar = replace(sidebar, "LOGO"=>logo)
    ## CONTENT
    index = open(f->read(f, String), path*"/ui/html/index.html")
    index = replace(index, "ICON"=>icon)
    #index = replace(index, "BACKGROUND_IMAGE"=>background)
    ## LOADING
    #loading = open(f->read(f, String), path*"/ui/html/loading.html")
    ## CSS
    loadcss!(w, bscss)
    loadcss!(w, bsiconcss)
    loadcss!(w, customcss)
    loadcss!(w, sidebarcss)
    # KATEX
    loadcss!(w, katexcss)
    loadjs!(w, katexjs)
    loadjs!(w, katexrender)
    # JQUERY, BOOSTRAP JS
    loadjs!(w, customjs)    #must be before jquery
    loadjs!(w, jquery)
    loadjs!(w, bsjs)        #after jquery
    loadjs!(w, customjs2)   #must be after jquery
    # LOAD ICONS
    loadcss!(w, icons)

    #Update GUI's home
    w = body!(w, *(sidebar, index), async=false)
    if darkmode
        @js_ w document.getElementById("main").style="background-color:rgb(13,16,17);"
    end

    # Return the Blink window
    return w, index
end

"""
    sys = setup_scanner()

Return the default scanner used by the UI and print some information about it.
"""
function setup_scanner()

    # Print information and get the default Scanner struct
    @info "Loading Scanner (default)"
    sys = Scanner()
    println("B0 = $(sys.B0) T")
    println("Gmax = $(round(sys.Gmax*1e3, digits=2)) mT/m")
    println("Smax = $(sys.Smax) mT/m/ms")

    # Return the default Scanner struct
    return sys
end

"""
    seq = setup_sequence(sys::Scanner)

Return the default sequence used by the UI and print some information about it.
"""
function setup_sequence(sys::Scanner)

    # Print information and get the default Sequence struct
    @info "Loading Sequence (default) "
    seq = PulseDesigner.EPI_example(; sys)

    # Return the default Sequence struct
    return seq
end

"""
    seq = setup_sequence(sys::Scanner)

Return the default sequence used by the UI and print some information about it.
"""
function setup_phantom(; phantom_mode="2D")

    # Print information and get the default Phantom struct
    @info "Loading Phantom (default)"
    obj = phantom_mode == "3D" ? brain_phantom3D() : brain_phantom2D()
    obj.Δw .*= 0
    println("Phantom object \"$(obj.name)\" successfully loaded!")

    # Return the default Phantom struct
    return obj
end


"""
"""
function display_loading!(w::Window, msg::String)
    path = @__DIR__
    loading = replace(open((f) -> read(f, String), path*"/ui/html/loading.html"), "LOADDES"=>msg)
    content!(w, "div#content", loading)
end

"""
"""
function callback_filepicker_seq(filename::String, w::Window, seq::Sequence)
    if filename != ""
        filename_extension = splitext(filename)[end]
        if filename_extension ==".seqk" #Koma
            seq = JLD2.load(FileIO.File{FileIO.DataFormat{:JLD2}}(filename),"seq")
        elseif filename_extension == ".seq" #Pulseq
            seq = read_seq(filename) #Pulseq read
        end
        @js_ w (@var name = $(basename(filename));
        document.getElementById("seqname").innerHTML = "<abbr title='" + name + "'>" + name + "</abbr>";
        Toasty("0", "Loaded <b>" + name + "</b> successfully", """
        <ul>
            <li>
                <button class="btn btn-dark btn-circle btn-circle-sm m-1" onclick="Blink.msg('pulses_seq', 1)"><i class="fa fa-search"></i></button>
                Updating <b>Sequence</b> plots ...
            </li>
            <li>
                <button class="btn btn-primary btn-circle btn-circle-sm m-1" onclick="Blink.msg('simulate', 1)"><i class="bi bi-caret-right-fill"></i></button>
                Ready to <b>simulate</b>?
            </li>
        </ul>
        """))
        return seq
    end
    return seq
end

"""
"""
function display_ui!(seq::Sequence, w::Window; type="sequence", darkmode=true)
    # Add loading progress and then a plot to the UI depending on type of the plot
    if type == "sequence"
        display_loading!(w, "Plotting sequence ...")
        widget_plot = plot_seq(seq; darkmode, range=[0 30])
        content!(w, "div#content", dom"div"(widget_plot))
        @js_ w document.getElementById("content").dataset.content = "sequence"
    elseif type == "kspace"
        display_loading!(w, "Plotting kspace ...")
        widget_plot = plot_kspace(seq; darkmode)
        content!(w, "div#content", dom"div"(widget_plot))
        @js_ w document.getElementById("content").dataset.content = "kspace"
    elseif type == "moment0"
        display_loading!(w, "Plotting moment 0 ...")
        widget_plot = plot_M0(seq; darkmode)
        content!(w, "div#content", dom"div"(widget_plot))
        @js_ w document.getElementById("content").dataset.content = "m0"
    elseif type == "moment1"
        display_loading!(w, "Plotting moment 1 ...")
        widget_plot = plot_M1(seq; darkmode)
        content!(w, "div#content", dom"div"(widget_plot))
        @js_ w document.getElementById("content").dataset.content = "m1"
    elseif type == "moment2"
        display_loading!(w, "Plotting moment 2 ...")
        widget_plot = plot_M2(seq; darkmode)
        content!(w, "div#content", dom"div"(widget_plot))
        @js_ w document.getElementById("content").dataset.content = "m2"
    end
end


"""
This launches the Koma's UI
"""
function KomaUI(; darkmode=true, frame=true, phantom_mode="2D", sim=Dict{String,Any}(), rec=Dict{Symbol,Any}(), dev_tools=false, blink_show=true)

    # Setup the Blink window
    path = @__DIR__
    w, index = setup_blink_window(path; darkmode, frame, dev_tools, blink_show)

    ## LOADING BAR
    buffericon = """<div class="spinner-border spinner-border-sm text-light" role="status"></div>"""
    progressbar = """
    <div class="progress" style="background-color: #27292d;">
      <div id="simul_progress" class="progress-bar" role="progressbar" style="width: 0%; transition:none;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">0%</div>
    </div>
    """

    # Setup default simulation inputs
    sys = setup_scanner()
    seq = setup_sequence(sys)
    global phantom = setup_phantom(; phantom_mode)

    ## BOOLEAN TO INDICATE FIRST TIME PRECOMPILING
    global ISFIRSTSIM = true
    global ISFIRSTREC = true

#Init
global raw_ismrmrd = RawAcquisitionData(Dict(
    "systemVendor" => "",
    "encodedSize" => [2,2,1],
    "reconSize" => [2,2,1],
    "number_of_samples" => 4,
    "encodedFOV" => [100.,100.,1],
    "trajectory" => "other"),
    [KomaMRICore.Profile(AcquisitionHeader(trajectory_dimensions=2, sample_time_us=1),
        [0. 0. 1 1; 0 1 1 1]./2, reshape([0.; 0im; 0; 0],4,1))])
global rawfile = ""
global image =  [0.0im 0.; 0. 0.]
global kspace = [0.0im 0.; 0. 0.]
global matfolder = tempdir()

#Reco
default = Dict{Symbol,Any}(:reco=>"direct") #, :iterations=>10, :λ=>1e-5,:solver=>"admm",:regularization=>"TV")
global recParams = merge(default, rec)
#Simulation
default = Dict{String,Any}()
global sim_params = merge(default, sim)
#GPUs
@info "Loading GPUs"
KomaMRICore.print_gpus()

    #OBERSVABLES
    obs_sys = Observable{Scanner}(sys)
    obs_seq = Observable{Sequence}(seq)
global pha_obs = Observable{Phantom}(phantom)
global sig_obs = Observable{RawAcquisitionData}(raw_ismrmrd)
global img_obs = Observable{Any}(image)
global mat_obs = Observable{Any}(matfolder)

#
handle(w, "matfolder") do args...
    str_toast = export_2_mat(obs_seq[], phantom, sys, raw_ismrmrd, recParams, image, matfolder; type="all", matfilename="")
    @js_ w (@var msg = $str_toast; Toasty("1", "Saved .mat files" , msg);)
    @js_ w document.getElementById("content").dataset.content = "matfolder"
end
handle(w, "matfolderseq") do args...
    str_toast = export_2_mat(obs_seq[], phantom, sys, raw_ismrmrd, recParams, image, matfolder; type="sequence")
    @js_ w (@var msg = $str_toast; Toasty("1", "Saved .mat files" , msg);)
    @js_ w document.getElementById("content").dataset.content = "matfolderseq"
end
handle(w, "matfolderpha") do args...
    str_toast = export_2_mat(obs_seq[], phantom, sys, raw_ismrmrd, recParams, image, matfolder; type="phantom")
    @js_ w (@var msg = $str_toast; Toasty("1", "Saved .mat files" , msg);)
    @js_ w document.getElementById("content").dataset.content = "matfolderpha"
end
handle(w, "matfoldersca") do args...
    str_toast = export_2_mat(obs_seq[], phantom, sys, raw_ismrmrd, recParams, image, matfolder; type="scanner")
    @js_ w (@var msg = $str_toast; Toasty("1", "Saved .mat files" , msg);)
    @js_ w document.getElementById("content").dataset.content = "matfoldersca"
end
handle(w, "matfolderraw") do args...
    str_toast = export_2_mat(obs_seq[], phantom, sys, raw_ismrmrd, recParams, image, matfolder; type="raw")
    @js_ w (@var msg = $str_toast; Toasty("1", "Saved .mat files" , msg);)
    @js_ w document.getElementById("content").dataset.content = "matfolderraw"
end
handle(w, "matfolderima") do args...
    str_toast = export_2_mat(obs_seq[], phantom, sys, raw_ismrmrd, recParams, image, matfolder; type="image")
    @js_ w (@var msg = $str_toast; Toasty("1", "Saved .mat files" , msg);)
    @js_ w document.getElementById("content").dataset.content = "matfolderima"
end

    # Handle "View" sidebar buttons
    handle(w, "index") do _
        content!(w, "div#content", index)
        @js_ w document.getElementById("content").dataset.content = "index"
    end
    handle(w, "pulses_seq") do _
        display_ui!(obs_seq[], w; type="sequence", darkmode)
    end
    handle(w, "pulses_kspace") do _
        display_ui!(obs_seq[], w; type="kspace", darkmode)
    end
    handle(w, "pulses_M0") do _
        display_ui!(obs_seq[], w; type="moment0", darkmode)
    end
    handle(w, "pulses_M1") do _
        display_ui!(obs_seq[], w; type="moment1", darkmode)
    end
    handle(w, "pulses_M2") do _
        display_ui!(obs_seq[], w; type="moment2", darkmode)
    end

handle(w, "phantom") do args...
    loading = replace(open(f->read(f, String), path*"/ui/html/loading.html"), "LOADDES"=>"Plotting phantom ...")
    content!(w, "div#content", loading)
    include(path*"/ui/PhantomGUI.jl")
end

handle(w, "sig") do args...
    loading = replace(open(f->read(f, String), path*"/ui/html/loading.html"), "LOADDES"=>"Plotting raw signal ...")
    content!(w, "div#content", loading)
    include(path*"/ui/SignalGUI.jl")
    @js_ w document.getElementById("content").dataset.content = "sig"
end
handle(w, "reconstruction_absI") do args...
    loading = replace(open(f->read(f, String), path*"/ui/html/loading.html"), "LOADDES"=>"Plotting image magnitude ...")
    content!(w, "div#content", loading)
    include(path*"/ui/ReconGUI_absI.jl")
    @js_ w document.getElementById("content").dataset.content = "absi"
end
handle(w, "reconstruction_angI") do args...
    loading = replace(open(f->read(f, String), path*"/ui/html/loading.html"), "LOADDES"=>"Plotting image phase ...")
    content!(w, "div#content", loading)
    include(path*"/ui/ReconGUI_angI.jl")
end
handle(w, "reconstruction_absK") do args...
    loading = replace(open(f->read(f, String), path*"/ui/html/loading.html"), "LOADDES"=>"Plotting image k ...")
    content!(w, "div#content", loading)
    include(path*"/ui/ReconGUI_absK.jl")
end
handle(w, "scanner") do args...
    loading = replace(open(f->read(f, String), path*"/ui/html/loading.html"), "LOADDES"=>"Displaying scanner parameters ...")
    content!(w, "div#content", loading)
    include(path*"/ui/ScannerParams_view.jl")
end
handle(w, "sim_params") do args...
    loading = replace(open(f->read(f, String), path*"/ui/html/loading.html"), "LOADDES"=>"Displaying simulation parameters ...")
    content!(w, "div#content", loading)
    include(path*"/ui/SimParams_view.jl")
end
handle(w, "rec_params") do args...
    loading = replace(open(f->read(f, String), path*"/ui/html/loading.html"), "LOADDES"=>"Displaying reconstruction parameters ...")
    content!(w, "div#content", loading)
    include(path*"/ui/RecParams_view.jl")
end
handle(w, "simulate") do args...
    strLoadingMessage = "Running simulation ..."
    if ISFIRSTSIM
        strLoadingMessage = "Precompiling and running simulation functions ..."
        ISFIRSTSIM = false
    end
    loading = replace(open(f->read(f, String), path*"/ui/html/loading.html"), "LOADDES"=>strLoadingMessage)
    content!(w, "div#content", loading)
    @js_ w document.getElementById("simulate!").setAttribute("disabled", true); #Disable button during SIMULATION
    @js_ w (@var progressbar = $progressbar; document.getElementById("simulate!").innerHTML=progressbar)
    #To SequenceGUI
    global raw_ismrmrd = simulate(phantom, obs_seq[], sys; sim_params, w)
    #After simulation go to RECON
    @js_ w document.getElementById("simulate!").innerHTML="Simulate!"
    #EXPORT to ISMRMRD -> To SignalGUI
    global rawfile = tempdir()*"/Koma_signal.mrd"
    @info "Exporting to ISMRMRD file: $rawfile"
    sig_obs[] = raw_ismrmrd
    fout = ISMRMRDFile(rawfile)
    KomaMRICore.save(fout, raw_ismrmrd)
    #Message
    sim_time = raw_ismrmrd.params["userParameters"]["sim_time_sec"]
    @js_ w (@var sim_time = $sim_time;
    @var name = $(phantom.name);
    document.getElementById("rawname").innerHTML="Koma_signal.mrd";
    Toasty("1", """Simulation successfull<br>Time: <a id="sim_time"></a> s""" ,"""
    <ul>
        <li>
            <button class="btn btn-dark btn-circle btn-circle-sm m-1" onclick="Blink.msg('sig', 1)"><i class="fa fa-search"></i></button>
            Updating <b>Raw signal</b> plots ...
        </li>
        <li>
            <button class="btn btn-primary btn-circle btn-circle-sm m-1" onclick="Blink.msg('recon', 1)"><i class="bi bi-caret-right-fill"></i></button>
            Ready to <b>reconstruct</b>?
        </li>
    </ul>
    """);
    document.getElementById("sim_time").innerHTML=sim_time;
    )
    loading = replace(open(f->read(f, String), path*"/ui/html/loading.html"), "LOADDES"=>"Plotting raw signal ...")
    content!(w, "div#content", loading)
    include(path*"/ui/SignalGUI.jl")
    @js_ w document.getElementById("content").dataset.content = "simulation"
    @js_ w document.getElementById("simulate!").removeAttribute("disabled"); #Re-enable button
    @js_ w document.getElementById("recon!").removeAttribute("disabled");
end
handle(w, "recon") do args...
    strLoadingMessage = "Running reconstruction ..."
    if ISFIRSTREC
        strLoadingMessage = "Precompiling and running reconstruction functions ..."
        ISFIRSTREC = false
    end
    loading = replace(open(f->read(f, String), path*"/ui/html/loading.html"), "LOADDES"=>strLoadingMessage)
    content!(w, "div#content", loading)
    # Update loading icon for button
    @js_ w (@var buffericon = $buffericon; document.getElementById("recon!").innerHTML=buffericon)
    #IMPORT ISMRMRD raw data
    raw_ismrmrd.profiles = raw_ismrmrd.profiles[getproperty.(getproperty.(raw_ismrmrd.profiles, :head), :flags) .!= 268435456] #Extra profile in JEMRIS simulations
    acqData = AcquisitionData(raw_ismrmrd)
    acqData.traj[1].circular = false #Removing circular window
    acqData.traj[1].nodes = acqData.traj[1].nodes[1:2,:] ./ maximum(2*abs.(acqData.traj[1].nodes[:])) #Normalize k-space to -.5 to .5 for NUFFT
    Nx, Ny = raw_ismrmrd.params["reconSize"][1:2]
    recParams[:reconSize] = (Nx, Ny)
    recParams[:densityWeighting] = true
    #Reconstruction
    @info "Running reconstruction of $rawfile"
    aux = @timed reconstruction(acqData, recParams)
    global image  = reshape(aux.value.data,Nx,Ny,:)
    global kspace = fftc(reshape(aux.value.data,Nx,Ny,:))
    # global img_obs[] = image
    #After Recon go to Image
    recon_time = aux.time
    @js_ w document.getElementById("recon!").innerHTML="Reconstruct!"
    @js_ w (@var recon_time = $recon_time;
    Toasty("2", """Reconstruction successfull<br>Time: <a id="recon_time"></a> s""" ,"""
    <ul>
        <li>
            <button class="btn btn-dark btn-circle btn-circle-sm m-1" onclick="Blink.msg('reconstruction_absI', 1)"><i class="fa fa-search"></i></button>
            Updating <b>Reconstruction</b> plots ...
        </li>
    </ul>
    """
    );
    document.getElementById("recon_time").innerHTML=recon_time;
    )
    loading = replace(open(f->read(f, String), path*"/ui/html/loading.html"), "LOADDES"=>"Plotting image magnitude ...")
    content!(w, "div#content", loading)
    include(path*"/ui/ReconGUI_absI.jl")
    @js_ w document.getElementById("content").dataset.content = "reconstruction"
end
handle(w, "close") do args...
    global phantom = nothing

    global raw_ismrmrd = nothing
    global rawfile = nothing

    global image = nothing
    global kspace = nothing

    global sim_params = nothing
    global recParams = nothing

    global pha_obs = nothing
    global sig_obs = nothing
    global img_obs = nothing
    global mat_obs = nothing
    close(w)
end

    # Define functionality of sequence filepicker widget
    widget_filepicker_seq = filepicker(".seq (Pulseq)/.seqk (Koma)"; accept=".seq,.seqk")
    map!((filename) -> callback_filepicker_seq(filename, w, obs_seq[]), obs_seq, widget_filepicker_seq)
    # Update on change of the sequence observable (when respective filepicker changes it)
    on((seq) -> display_ui!(seq, w; type="sequence", darkmode), obs_seq)

    # Add filepicker widgets to the UI
    content!(w, "#seqfilepicker", widget_filepicker_seq, async=false)

#Phantom observable
load_pha = filepicker(".phantom (Koma)/.h5 (JEMRIS)"; accept=".phantom,.h5")
map!(f->if f!="" #Assigning function of data when load button (filepicker) is changed
            if splitext(f)[end]==".phantom" #Koma
                global phantom = JLD2.load(FileIO.File{FileIO.DataFormat{:JLD2}}(f),"phantom")
            elseif splitext(f)[end]==".h5" #JEMRIS
                global phantom = read_phantom_jemris(f)
            end
            @js_ w (@var name = $(basename(f));
            document.getElementById("phaname").innerHTML="<abbr title='"+name+"'>"+name+"</abbr>";;
            Toasty("0", "Loaded <b>"+name+"</b> successfully", """
            <ul>
                <li>
                    <button class="btn btn-dark btn-circle btn-circle-sm m-1" onclick="Blink.msg('phantom', 1)"><i class="fa fa-search"></i></button>
                    Updating <b>Phantom</b> plots ...
                </li>
                <li>
                    <button class="btn btn-primary btn-circle btn-circle-sm m-1" onclick="Blink.msg('simulate', 1)"><i class="bi bi-caret-right-fill"></i></button>
                    Ready to <b>simulate</b>?
                </li>
            </ul>
            """))
            phantom
        else
            phantom #default
        end
    , pha_obs, load_pha)
w = content!(w, "#phafilepicker", load_pha, async=false)

#Signal observable
load_sig = filepicker(".h5/.mrd (ISMRMRD)"; accept=".h5,.mrd")
map!(f->if f!="" #Assigning function of data when load button (filepicker) is changed
            fraw = ISMRMRDFile(f)
            global rawfile = f
            global raw_ismrmrd = RawAcquisitionData(fraw)

            not_Koma = raw_ismrmrd.params["systemVendor"] != "KomaMRI.jl"
            if not_Koma
                @warn "ISMRMRD files generated externally could cause problems during the reconstruction. We are currently improving compatibility."
            end

            @js_ w (@var name = $(basename(f));
            document.getElementById("rawname").innerHTML="<abbr title='"+name+"'>"+name+"</abbr>";;
            Toasty("0", "Loaded <b>"+name+"</b> successfully", """
            <ul>
                <li>
                    <button class="btn btn-dark btn-circle btn-circle-sm m-1" onclick="Blink.msg('sig', 1)"><i class="fa fa-search"></i></button>
                    Updating <b>Raw data</b> plots ...
                </li>
                <li>
                    <button class="btn btn-success btn-circle btn-circle-sm m-1" onclick="Blink.msg('recon', 1)"><i class="bi bi-caret-right-fill"></i></button>
                    Ready to <b>reconstruct</b>?
                </li>
            </ul>
            """))
            raw_ismrmrd
        else
            raw_ismrmrd #default
        end
    , sig_obs, load_sig)
w = content!(w, "#sigfilepicker", load_sig, async=false)

    # Update Koma version on UI
    version = string(KomaMRICore.__VERSION__)
    content!(w, "#version", version, async=false)
    @info "Currently using KomaMRICore v$version"

    # Return the Blink Window or not
    (dev_tools) && return w
    return nothing
end

"""
Auxiliary function to updates KomaUI's simulation progress bar.
"""
function update_blink_window_progress!(w::Window, block, Nblocks)
    progress = string(floor(Int, block / Nblocks * 100))
    @js_ w (@var progress = $progress;
    document.getElementById("simul_progress").style.width = progress + "%";
    document.getElementById("simul_progress").innerHTML = progress + "%";
    document.getElementById("simul_progress").setAttribute("aria-valuenow", progress))
    return nothing
end
