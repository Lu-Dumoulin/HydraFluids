# Type of nucleator dynamics if the nucleator exist
abstract type OrientationTrait end
struct IsNone    <: OrientationTrait end  # No orientation field
struct IsPolar  <: OrientationTrait end  # Only Polar
struct IsNematic  <: OrientationTrait end  # Only Nematic
struct IsNematoPolar <: OrientationTrait end  # NematoPolar

@inline has_orientation(::IsNone)  = false
@inline has_orientation(::Any)     = true
@inline is_polar(::Union{IsPolar, IsNematoPolar})    = true
@inline is_polar(::Any)       = false
@inline is_nematic(::Union{IsNematic, IsNematoPolar})    = true
@inline is_nematic(::Any)         = false


# Map user symbols to our Trait Types
function get_trait(choice::Symbol)
    if choice == :none      return IsNone()
    elseif choice == :polar    return IsPolar()
    elseif choice == :nematic     return IsNematic()
    elseif choice == :nematopolar  return IsNematoPolar()
    else
        error("Unknown state: $choice. Choose from :none, :polar, :nematic, :nematopolar")
    end
end


# abstract type Perturbation end
# struct None   <: Perturbation end
# struct Cut    <: Perturbation end  # Perturbation by cutting
# struct Noise  <: Perturbation end  # Perturbation by noise

#  # Map user symbols to our Perturbation Trait Types
# function get_trait_perturbation(choice::Symbol)
#     if choice == :none      return None()
#     elseif choice == :cut    return Cut()
#     elseif choice == :noise     return Noise()
#     else
#         error("Unknown state: $choice. Choose from :none, :cut, :noise")
#     end
# end

# Function to save data
@inline function prepare_and_add!(dict, label, data_host, data_device)
    copyto!(data_host, data_device)
    dict[label] = data_host
    return nothing
end
@inline prepare_and_add!(dict, label, data_host::Nothing, data_device::Nothing) = nothing

function save_simulation_state(filename, ρ, V, P, Q, ρ_host, V_host, P_host, Q_host)
    # We use a dictionary to store only the fields that actually exist
    data_to_save = Dict{Symbol, Any}()

    # These calls are specialized. If arp is nothing, 
    # the compiler removes the copyto! and the dict insertion.
    prepare_and_add!(data_to_save, :C,    ρ_host,    ρ)
    prepare_and_add!(data_to_save, :V,    V_host,    V)
    prepare_and_add!(data_to_save, :P,    P_host,    P)
    prepare_and_add!(data_to_save, :Q,    Q_host,    Q)

    # Save the collected data
    JLD2.jldsave(filename; data_to_save...)
end