using Flux, JLD2, Statistics
include("fcnn.jl")

model=get_model("DINO.GZ21/INFERENCES/julia/model_weights.jld2")


input = randn(Float32, 200, 128, 2, 10); # i, j, c (zonal, meridional), k (depth levels)


out = activation(model(input));
out = activation(out);
@info mean(out, dims=(1,2,4))
