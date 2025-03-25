using PyCall
using Flux
using Flux, JLD2, Statistics, HDF5

base = "/Users/emeunier/Desktop/NEMO-DINO/DINO.GZ21/INFERENCES/"
pushfirst!(pyimport("sys")."path", base)
pushfirst!(pyimport("sys")."path", "$base/gz21_ocean_momentum/src/gz21_ocean_momentum/")

ca(torch_array) = convert(Array{Float32}, torch_array.detach().numpy())
pa(array) = Array(PermutedDimsArray(array, (3, 4, 2, 1)))

# Torch Model 
ml_models = pyimport("ml_models")
torch = pyimport("torch")

net = ml_models.model_loading("$base/weights/gz21_huggingface/low-resolution/files/trained_model.pth")
state_dict = net.state_dict()


# Flux Model + Import Weights
model=get_model()

p = Flux.params(model);

for i in 0:7 
    @info i*2+1, "$(i*2).weight", i*2+2, "$(i*2).bias"
    p[i*2+1] .= reverse(pa(ca(state_dict["$(i*2).weight"])), dims=(1, 2))
    p[i*2+2] .= ca(state_dict["$(i*2).bias"])
end

jldsave("/Users/emeunier/Desktop/NEMO-DINO/DINO.GZ21/INFERENCES/weights/gz21_huggingface/low-resolution/files/model_weights.jld2";  model_state=Flux.state(model))

