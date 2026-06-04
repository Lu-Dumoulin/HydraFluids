### A Pluto.jl notebook ###
# v0.20.28

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 1ead4d39-e5e7-4117-9e14-dfdcd92be319
begin
	using DelimitedFiles, CSV, DataFrames, PlutoUI, PlutoTeachingTools
	try
		include("common/utils/UI_utils.jl")
		@info "Module UI_utils is loaded"
	catch
		try 
			UI_utils.parse_values("1")
			@info "Module UI_utils is already loaded"
		catch
			@error "Error trying to load `utils/UI_utils.jl`. Try to restart the notebook."
		end
	end
	try
		include("common/utils/DF_utils.jl")
		@info "Module DF_utils is loaded"
	catch
		try 
			DF_utils.isloaded()
			@info "Module SSH_utils is already loaded"
		catch
			@error "Error trying to load `utils/DF_utils.jl`. Try to restart the notebook."
		end
	end

	const ENUM_ORI    = ["none", "nematic", "polar", "nematopolar"];
	const ENUM_INI   = ["Homogeneous", "Polarized", "Loop"];

	TableOfContents()
end |> WideCell

# ╔═╡ f30515e4-6f66-4209-8e6a-12c5808487ad
let
notebook_path= joinpath(@__DIR__, "../App.jl")

Markdown.parse("""You can return to the main page using [this link](./open?path=$notebook_path)""")
end |> WideCell

# ╔═╡ 6c39efc0-de1c-43f3-912f-fb887c0c2632
WideCell(md"""
## 1. Generate a dataframe with the input parameters
""")

# ╔═╡ cbb0ecab-37d1-4123-807e-14fabba14115
WideCell(md"""
#### System size and Discretization
""")

# ╔═╡ 33dc1510-fa01-44cf-a7e6-b026b18ceb41
WideCell(TwoColumn(
md"""
##### Size
Suqare lattice length, multiple of 16, N = 16 × $(@bind N_str TextField(default="32, 64"))  

##### Discretization
Spacial discretization: Δx = Δy = Δ = 2 ^(- $(@bind dx_str TextField(default="6")))

Time discretization: Maximal Δt = $(@bind dt_max_str TextField(default="0.01")),
	
Initial Δt = $(@bind dt_ini_str TextField(default="0.001"))
""",
md"""
##### Simulation time
Adapt Δt every $(@bind t_check_str TextField(default="0.1")) time units

Save data every $(@bind t_print_str TextField(default="2")) time units

Duration of simulation in time units: $(@bind t_end_str TextField(default="100"))
"""
))

# ╔═╡ 7bcf11f2-e188-4313-a2f7-b0ded82878ca
begin
	N = 16 .* Int.(UI_utils.parse_values(N_str))
	dx = 2.0 .^ (.- Int.(UI_utils.parse_values(dx_str)))
	
	listtab1, listname1 = UI_utils.@named_parse [N, dx, dt_max_str, dt_ini_str, t_check_str, t_print_str, t_end_str]

	UI_utils.print_list(listname1, listtab1)
end |> WideCell

# ╔═╡ 20908dc0-3394-4841-9ef0-42c638e00c03
WideCell(md"""
#### Orientation fields and Initialisation states
""")

# ╔═╡ 7af3d65d-5329-4456-8acd-a2f5169e5eb2
WideCell(
TwoColumnWideLeft(md"""
##### Initialisation:
Type of initialisation:
$(@bind initialisation MultiCheckBox(ENUM_INI, default=["Homogeneous", "Loop"]))
				   
Noise amplitude: ``\eta_\rho`` = $(@bind eta_rho_str TextField((5,1), default="0.001")), ``\eta_p`` = $(@bind eta_p_str TextField((5,1), default="0.001")),  ``\eta_Q`` = $(@bind eta_q_str TextField((5,1), default="0.001")) 
with seed(s) : $(@bind seed_str TextField((5,1),default="1"))
"""
,
md"""
##### Orientation Field(s):
$(@bind orientation MultiCheckBox(ENUM_ORI, default=["none", "polar", "nematopolar"]))
"""
))

# ╔═╡ bb39ce96-d8c8-4a86-bcf6-0e2a2bbf7e00
begin
	listtab2, listname2 = UI_utils.@named_parse [initialisation, eta_rho_str, eta_p_str, eta_q_str, seed_str, orientation];

	UI_utils.print_list(listname2, listtab2)
end |> WideCell

# ╔═╡ 3fc109bc-b40d-422d-8974-af8367eb4510
WideCell(md"""
#### Physical Parameters
""")

# ╔═╡ e3a46d44-4cb9-4ec5-bec2-70bdbac96a14
WideCell(
	md"""
First you can enter the parameters relative to density, then polar order, nematic order and eventually coupling between both fields.
```math
\mathcal{F} = \int {\rm d}^2 [ f_\rho + f_p + f_Q + f_{pQ} ]
```
##### Density related parameters:
"""
)

# ╔═╡ 42c15eb5-9572-44f6-85ad-728440da5465
WideCell(TwoColumn(
md"""
Free energy:
``
  f_\rho = \dfrac{a}{4} \rho^4
``
    
``a`` = $(@bind a_str TextField(default="1"))
    
Isotropic active stress:
``
\sigma^{\text{act},\rho}_{\alpha\beta} = \zeta\Delta\mu \rho^3\delta_{\alpha\beta}
``
    
 ``\zeta`` = $(@bind zeta_str TextField((8,1),default="0.8")) 

""",
md"""
Continuity equation: 
```math
\partial_t \rho = - \partial_\beta ( v_\beta \rho - D \partial_{\beta} {\rho} ) + \tau^{-1}(\rho_0 - \rho)
```
Diffusion coefficient: ``D`` = $(@bind D_str TextField((15,1),default="0.0001")) 

Renewal time: ``\tau`` = $(@bind tau_str TextField((25,1),default="1")) 

Target density: ``\rho_0`` = $(@bind rho0_str TextField((25,1),default="0.4:0.05:1.0"))
"""))

# ╔═╡ eca8b556-c151-46e3-853c-ba9a012e0e44
begin
	listtab_rho, listname_rho = UI_utils.@named_parse [a_str, zeta_str, D_str, tau_str, rho0_str]
	UI_utils.print_list(listname_rho, listtab_rho)
end |> WideCell

# ╔═╡ b854b1b1-586b-461a-b69e-d5625c779389
WideCell(md"""
		 ##### Polar parameters""")

# ╔═╡ d5a11f5d-5e72-4312-a3be-8dff912ce9cb
WideCell(TwoColumn(
md"""
Free energy:
	```math
  f_p = \rho^2 \left[  -\frac{\alpha_p}{2}\frac{\rho}{\rho_0} p^2 + \frac{\beta_p}{4} p^4 + \frac{\kappa_p}{2} (\partial_\alpha p_\beta )(\partial_\alpha p_\beta)\right]
```
``\alpha_p`` = $(@bind alpha_p_str TextField((8,1), default="0,0.1")), 
``\beta_p`` = $(@bind beta_p_str TextField((8,1), default="0.1")), 
``\kappa_p`` = $(@bind kappa_p_str TextField((8,1), default="0.0001"))
	
Active polar stress: 
	``\sigma^{\text{act},p}_{\alpha\beta} = \rho\zeta_p\Delta\mu p_\alpha p_\beta + \rho\tilde\zeta_p\Delta\mu p_\gamma p_\gamma \delta_{\alpha\beta}``

``\zeta_p`` = $(@bind zeta_p_str TextField((8,1),default="0.0")) , ``\tilde\zeta_p`` = $(@bind zeta_p2_str TextField((8,1),default="0.0")) 
""",
md"""
Continuity equation:
```math
\partial_t p_\alpha = - v_\gamma \partial_\gamma p_\alpha - \Omega_{\alpha\gamma} p_{\gamma} + \Gamma_p^{-1} h_\alpha -  \nu  p_\beta v_{\alpha\beta} - \bar{\nu} p_\alpha  v_{\gamma\gamma} + \varepsilon_p\Delta\mu
```
Rotational viscosity: ``\Gamma_p`` = $(@bind gamma_p_str TextField((25,1),default="1.0"))
	
Flow alignement 1: ``\nu`` = $(@bind nu_str TextField((25,1),default="0.0"))
	
Flow alignement 2: ``\bar\nu`` = $(@bind nu2_str TextField((25,1),default="0.0"))

Active extensil: ``\varepsilon_p`` = $(@bind epsi_p_str TextField((25,1),default="0.0"))
"""
)
)

# ╔═╡ 314e1b98-83e7-41c3-a30d-3360775c28ab
begin
	listtab_p, listname_p = UI_utils.@named_parse [alpha_p_str, beta_p_str, kappa_p_str, zeta_p_str, zeta_p2_str, gamma_p_str, nu_str, nu2_str, epsi_p_str]
	UI_utils.print_list(listname_p, listtab_p)
end |> WideCell

# ╔═╡ 321a1c3d-6fcc-46d1-ab5a-bf101b35cf8a
WideCell(md"""
		 ##### Nematic parameters
		 """)

# ╔═╡ 7f94634c-474d-470f-9bf1-e6de504fdc26
WideCell(TwoColumn(
md"""
Free energy:
```math
  f_Q = \rho^2 \left[  -\frac{\alpha_Q}{2}\frac{\rho}{\rho_0} Q^2 + \frac{\beta_Q}{2} Q^4 + \frac{\kappa_Q}{2} (\partial_\alpha Q_{\beta\gamma})(\partial_\alpha Q_{\beta\gamma})\right]
```
``\alpha_Q`` = $(@bind alpha_q_str TextField((5,1), default="0.1"))
``\beta_Q`` = $(@bind beta_q_str TextField((5,1), default="0.1"))
``\kappa_Q`` = $(@bind kappa_q_str TextField((5,1), default="0.0001"))

Active nematic stress: 
	``\sigma^{\text{act},Q}_{\alpha\beta} = \rho\zeta_Q\Delta\mu Q_{\alpha\beta}``

``\zeta_Q`` = $(@bind zeta_q_str TextField((8,1),default="0.0"))
""",
md"""
```math
\partial_t Q_{\alpha\beta} = - v_\gamma \partial_\gamma Q_{\alpha\beta} - \Omega_{\alpha\gamma} Q_{\gamma\beta} + \Gamma_Q^{-1} H_{\alpha\beta} - 2 \lambda \left( v_{\alpha\beta} - \frac{1}{2}  v_{\gamma\gamma} \right)
```
``\quad\quad\quad\quad + \varepsilon_Q\Delta\mu ``
	
Rotational viscosity: ``\Gamma_Q`` = $(@bind gamma_q_str TextField((25,1),default="1.0"))
	
Flow alignement 1: ``\lambda`` = $(@bind lambda_str TextField((25,1),default="0.0"))

Active extensil: ``\varepsilon_Q`` = $(@bind epsi_q_str TextField((25,1),default="0.0"))
"""
)
)

# ╔═╡ dd59581c-4c15-45fc-b515-0b7a0367e942
begin
	listtab_q, listname_q = UI_utils.@named_parse [alpha_q_str, beta_q_str, kappa_q_str, zeta_q_str, gamma_q_str, lambda_str, epsi_q_str]
	UI_utils.print_list(listname_q, listtab_q)
end |> WideCell

# ╔═╡ 4165eac0-7ad0-49d0-afb9-8c5c51c5652a
WideCell(md"""
		 ##### NematoPolar coupling""")

# ╔═╡ 54c99ab3-99fc-46e1-8141-ef08b4f2ace6
WideCell(md"""
```math
f_{pQ} = - \frac{\chi}{2} \rho^2 Q_{\alpha\beta} \left(p_\alpha p_\beta - \frac{p_\gamma p_\gamma}{2} \delta_{\alpha\beta} \right) = \frac{\chi}{2} \rho^2 \left[ Q_1(p_y^2 - p_x^2) - 2 Q_2 p_x p_y \right]. 
```
``\chi`` = $(@bind chi_str TextField(default="0.1"))	 
 """)

# ╔═╡ 531ef45f-800a-4a5f-a05a-d47c6ec3e5f4
begin
	listtab_pq, listname_pq = UI_utils.@named_parse [chi_str]
	UI_utils.print_list(listname_pq, listtab_pq)
end |> WideCell

# ╔═╡ 2bc2c9d4-7381-4d89-8b9d-8546e73d8354
WideCell(
    begin
        listtab = vcat(listtab1, listtab2, listtab_rho, listtab_p, listtab_q, listtab_pq)
        listname = vcat(listname1, listname2, listname_rho, listname_p, listname_q, listname_pq)
        any(isempty, listtab)
        df = DF_utils.generate_dataframe(listname, listtab)
        # remove value of P parameters and loop if no P:
        # df[(df.orientation .== "none"),  listname_p] .= 0.0
        df[map(x->!x, contains.(df.orientation, "polar")),  vcat(listname_p, listname_pq, ["eta_p"])] .= 0.0
        df[map(x->!x, contains.(df.orientation, "nemat")),  vcat(listname_q, listname_pq, ["eta_q"])] .= 0.0
        df[(map(x->!x, contains.(df.orientation, "nematopolar")) .& contains.(df.initialisation, "Loop")),  :initialisation] .= "Homogeneous"
        unique!(df)
        Nsim = isempty(df) ? 0 : nrow(df)
        DataFrames.insertcols!(df, 1, :fn => string.(1:Nsim))
    	# println()
    	println("Number of Simulations: $Nsim")
        df
    end
)

# ╔═╡ b34e3630-8b7d-450f-9594-3474939a66bf
# ╠═╡ disabled = true
#=╠═╡
# function apply_constraints!(df::DataFrame)
#     isempty(df) && return df
#     df[df.orientation .== "none",     [:K, :mu, :la, :Darp]] .= 0.0
#     df[df.arp_state .== "reaction", :Darp]                  .= 0.0
#     df[df.formin_state .== "none",     [:kp, :Dformin]] .= 0.0
#     df[df.formin_state .== "reaction", :Dformin]        .= 0.0
#     df[df.perturbation .== "none", [:seed, :perturbation_time]]  .= 0.0
#     unique!(df)
#     insertcols!(df, 1, :fn => 1:nrow(df))
#     return df
# end
  ╠═╡ =#

# ╔═╡ 829af64f-7846-439f-86ba-7325ef189585
begin
csv_path = joinpath(normpath(@__DIR__,"sim/"), "DF.csv")

md"""
If you continue the DataFrame will be saved in:
$(csv_path)

Switch to continue: $(@bind continue_csv Switch()).
"""
end |> WideCell

# ╔═╡ 07b32a31-1537-4e19-b842-7840863d7606
if continue_csv
    CSV.write(csv_path, df)
    println("DataFrame saved: $csv_path")
end

# ╔═╡ 237320e6-6b38-4f95-969f-6c8f64a3ee16
let
	notebook_path= joinpath(@__DIR__, "common/RunSimulations.jl")

	Markdown.parse("Now that your parameters are generated, you can run the simulation using [this page](./open?path=$notebook_path)")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
CSV = "~0.10.16"
DataFrames = "~1.8.2"
PlutoTeachingTools = "~0.4.7"
PlutoUI = "~0.7.82"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "b3928f466c9a5862af449c1bb3e04fc6054cebe8"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "8d8e0b0f350b8e1c91420b5e64e5de774c2f0f4d"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.16"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "5fab31e2e01e70ad66e3e24c968c264d1cf166d6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.2"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c0c9b76f3520863909825cbecdef58cd63de705a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.5+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "5d5e0a78e971354b1c7bff0655d11fdc1b0e12c8"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.4"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "90b41ced6bacd8c01bd05da8aed35c5458891749"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.7"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "0ecd70a51c13e150266e76a865f10a64a7f178a3"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.82"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "624de6279ab7d94fc9f672f0068107eb6619732c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.3.2"

    [deps.PrettyTables.extensions]
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "084c47c7c5ce5cfecefa0a98dff69eb3646b5a80"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.10"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "d05693d339e37d6ab134c5ab53c29fce5ee5d7d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.4"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "0716e01c3b40413de5dedbc9c5c69f27cddfddfc"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.3"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"
"""

# ╔═╡ Cell order:
# ╟─1ead4d39-e5e7-4117-9e14-dfdcd92be319
# ╟─f30515e4-6f66-4209-8e6a-12c5808487ad
# ╟─6c39efc0-de1c-43f3-912f-fb887c0c2632
# ╟─cbb0ecab-37d1-4123-807e-14fabba14115
# ╟─33dc1510-fa01-44cf-a7e6-b026b18ceb41
# ╟─7bcf11f2-e188-4313-a2f7-b0ded82878ca
# ╟─20908dc0-3394-4841-9ef0-42c638e00c03
# ╟─7af3d65d-5329-4456-8acd-a2f5169e5eb2
# ╟─bb39ce96-d8c8-4a86-bcf6-0e2a2bbf7e00
# ╟─3fc109bc-b40d-422d-8974-af8367eb4510
# ╟─e3a46d44-4cb9-4ec5-bec2-70bdbac96a14
# ╟─42c15eb5-9572-44f6-85ad-728440da5465
# ╟─eca8b556-c151-46e3-853c-ba9a012e0e44
# ╟─b854b1b1-586b-461a-b69e-d5625c779389
# ╟─d5a11f5d-5e72-4312-a3be-8dff912ce9cb
# ╟─314e1b98-83e7-41c3-a30d-3360775c28ab
# ╟─321a1c3d-6fcc-46d1-ab5a-bf101b35cf8a
# ╟─7f94634c-474d-470f-9bf1-e6de504fdc26
# ╟─dd59581c-4c15-45fc-b515-0b7a0367e942
# ╟─4165eac0-7ad0-49d0-afb9-8c5c51c5652a
# ╟─54c99ab3-99fc-46e1-8141-ef08b4f2ace6
# ╟─531ef45f-800a-4a5f-a05a-d47c6ec3e5f4
# ╟─2bc2c9d4-7381-4d89-8b9d-8546e73d8354
# ╟─b34e3630-8b7d-450f-9594-3474939a66bf
# ╟─829af64f-7846-439f-86ba-7325ef189585
# ╟─07b32a31-1537-4e19-b842-7840863d7606
# ╟─237320e6-6b38-4f95-969f-6c8f64a3ee16
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
