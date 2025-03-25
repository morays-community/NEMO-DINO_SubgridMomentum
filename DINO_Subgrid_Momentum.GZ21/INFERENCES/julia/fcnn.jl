using Flux: Conv, relu, Chain
using Flux.Optimise: softplus

function activation(x; precision_indices=3:4, min_value=0.0015)
    out = copy(x) # If we want to avoid inplace modification
    out[:, :, precision_indices, :] .= softplus.(x[:, :, precision_indices, :]) .+ min_value
    return out
end

function get_model(weight_path=nothing) 
    # Define the network structure
    model = Chain(
        Conv((5, 5), 2 => 128, pad = (2, 2)), relu,
        Conv((5, 5), 128 => 64, pad = (2, 2)), relu,
        Conv((3, 3), 64 => 32, pad = (1, 1)), relu,
        Conv((3, 3), 32 => 32, pad = (1, 1)), relu,
        Conv((3, 3), 32 => 32, pad = (1, 1)), relu,
        Conv((3, 3), 32 => 32, pad = (1, 1)), relu,
        Conv((3, 3), 32 => 32, pad = (1, 1)), relu,
        Conv((3, 3), 32 => 4, pad = (1, 1)))
    if !isnothing(weight_path) 
        @info "Loading model : ", weight_path
        model_state = JLD2.load(weight_path, "model_state");
        Flux.loadmodel!(model, model_state);
    end
    return model
end

function subgrid_forcing(u, v; u_scale=10.0, v_scale=10.0, Su_scale=1e-7, Sv_scale=1e-7, sampling=true)
    """
    parameters :
        u (i, j, k) : zonal velocity
        v (i, j, k) : meridional velocity
        sampling (bool) : indicates if we add noise to the subgrid forcing. If False takes the MLE

    return 
        Su (i, j, k) : zonal subgrid_forcing
        Sv (i, j, k) : meridional subgrid_forcing

    Values from scale : https://github.com/chzhangudel/Forpy_CNN_GZ21/blob/smartsim/testNN.py
    """
    out = model(stack([u .* u_scale, v .* v_scale], dims=3)) #(w, h, 2, k) - Here we consider depth layers as batch as they are processed independently
    out = activation(out)
    Su, Sv, Spu, Spv = out[:,:,1,:], out[:,:,2,:], out[:,:,3,:], out[:,:,4,:]
    Su = Su_scale * ( Su + sqrt.(1 ./Spu).*randn(size(Spu)) * sampling )
    Sv = Sv_scale * ( Sv + sqrt.(1 ./Spv).*randn(size(Spv)) * sampling )
    return Su, Sv
end



