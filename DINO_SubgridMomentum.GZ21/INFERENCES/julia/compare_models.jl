using PyCall
using Flux
using Flux, JLD2, Statistics, HDF5

# Torch Model 
py"""
import torch 
import torch.nn as nn
import sys

b, c, i, j = 1, 2, 10, 20
base = "/Users/emeunier/Desktop/NEMO-DINO/DINO.GZ21/INFERENCES/"
random_mat = False

sys.path.append(f'{base}/models/')
import ml_models


net = ml_models.model_loading(f"{base}/weights/gz21_huggingface/low-resolution/files/trained_model.pth")

if random_mat :
    inp = torch.randn(b,c,i, j)
else : 
    def function_mat_python(b, c, i, j) : 
        return b*0.7 + c*0.1 + i*0.827 + j*0.193

    def create_mat_python(b, c, i, j) : 
        inp = torch.zeros((b,c, i ,j))
        for bi in range(0,b):
            for ci in range(0,c) :
                for ii in range(0,i):
                    for ji in range(0,j) :
                        inp[bi, ci, ii, ji] = function_mat_python(bi,ci,ii,ji)
        return inp

    inp = create_mat_python(b,c,i,j)

print('input', inp[0,0, 0,0], inp[0, 0, -1,-1], inp[0,1, 0,0], inp[0, 1, -1,-1])


out_torch = net(inp).detach().numpy(); # b, c, w, h

inp = inp.numpy()

print('output', out_torch[0,0, 0,0], out_torch[0, 0, -1,-1], out_torch[0,1, 0,0], out_torch[0, 1, -1,-1], out_torch[0,2, 0,0], out_torch[0, 2, -1,-1], out_torch[0,3, 0,0], out_torch[0, 3, -1,-1])
"""


# Julia Model 

include("fcnn.jl")
base = py"base"
b, c, i, j = py"b",py"c",py"i",py"j"
random_mat = py"random_mat"


model=get_model("$base/julia/model_weights.jld2");

if random_mat
    inp_julia = py"inp"
else 
    func_mat(b, c, i, j) = b*0.7 + c*0.1 + i*0.827 + j*0.193

    function create_mat(b,c,i,j)
        x = zeros(Float32, b, c, i, j)
        for bi in 1:b 
            for ci in 1:c 
                for ii  in 1:i 
                    for ji  in 1:j 
                        x[bi,ci,ii,ji] = func_mat(bi-1,ci-1,ii-1,ji-1)
                    end
                end
            end
        end
        x
    end

    inp_julia = create_mat(b,c,i,j);
end

@info("input", inp_julia[1,1,1,1], inp_julia[1,1,end,end], inp_julia[1,2,1,1], inp_julia[1,2,end,end])


inp_julia = PermutedDimsArray(inp_julia, (3, 4, 2, 1)); # w h c b

out_julia = activation(model(inp_julia)); 
Su_julia, Sv_julia = subgrid_forcing(inp_julia[:,:,1,:], inp_julia[:,:,2,:]; sampling=false)

out_julia = PermutedDimsArray(out_julia, (4, 3, 1, 2)); 

out_torch = py"out_torch"
@info("output", out_julia[1,1,1,1], out_julia[1,2,1,1], out_julia[1,3,1,1], out_julia[1,4,1,1])
@info("output", out_torch[1,1,1,1], out_torch[1,2,1,1], out_torch[1,3,1,1], out_torch[1,4,1,1])

@info("diff mean" , mean(abs.(out_torch - out_julia)))
@info("diff max" , maximum(abs.(out_torch - out_julia)))
