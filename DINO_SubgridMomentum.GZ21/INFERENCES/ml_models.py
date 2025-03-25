import numpy as np
import torch, einops
import sys
sys.path.append('gz21_ocean_momentum/src')

from gz21_cnn import FullyCNN
import transforms as transforms


#       Utils 
# -----------------
def Is_None(*inputs):
    """ Test presence of at least one None in inputs """
    return any(item is None for item in inputs)


@torch.no_grad()
def model_loading(weights_path='weights/gz21_huggingface/low-resolution/files/trained_model.pth', device='cpu', padding='init_circular') :
    net = FullyCNN(padding=padding)
    try: # in-repo test or in local deployed config dir
        model_weights = torch.load('trained_model.pth', map_location=device)
    except:
        model_weights = torch.load(weights_path , map_location=device)
    transformation = transforms.SoftPlusTransform()
    transformation.indices = [2, 3] # Careful if model change
    net.final_transformation = transformation
    net.load_state_dict(model_weights)
    net.eval()
    net.to(device)
    return net


#       Main Model Routines
# ------------------------------
# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')


# Load model
net = model_loading(padding='init_zeros' , device=device)
# From https://github.com/chzhangudel/Forpy_CNN_GZ21/blob/smartsim/testNN.py
u_scale= 10
v_scale= 10
Su_scale= 1e-7
Sv_scale= 1e-7

# Predictions    
@torch.no_grad()
def momentum_cnn(u, v, mask_u, mask_v, sampling=True):
    """ Take as input u and v fields and return forcing fields using GZ (2021)

    Param :
        u (i, j, k)
        v (i, j, k)
        mask_u (i j k)
        mask_v (i j k)  
        sampling (bool) : to add random noise or not
    Out : 
        Su (i j k)
        Sv (i j k)
    """
    if Is_None([u, v]):
        return None
    else:
        global net, u_scale, v_scale, Su_scale, Sv_scale, device
        # pack inputs
        inp = einops.rearrange( [ torch.tensor(u_scale*u.astype(np.float32)*mask_u.astype(np.float32)).to(device) ,
                                  torch.tensor(v_scale*v.astype(np.float32)*mask_v.astype(np.float32)).to(device) ], 'c i j k -> k c i j' )
        # preds
        r = net(inp)
        Su_mu, Sv_mu, Su_p, Sv_p = r[:, 0], r[:, 1], r[:, 2], r[:, 3] # k i j

        # subgrid forcing terms
        u = Su_scale * ( Su_mu + torch.sqrt(1/Su_p)*torch.randn_like(Su_p)*sampling)
        v = Sv_scale * ( Sv_mu + torch.sqrt(1/Sv_p)*torch.randn_like(Sv_p)*sampling)
        if device.type == 'cuda':
            u = u.cpu()
            v = v.cpu()
        u = einops.rearrange(u, 'k i j -> i j k').numpy()
        v = einops.rearrange(v, 'k i j -> i j k').numpy()

        return u*mask_u , v*mask_v
    

if __name__ == '__main__' : 

    b, c, i, j = 1, 2, 100, 200
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
    
    u = inp[:,0].permute(1,2,0).numpy()
    v = inp[:,1].permute(1,2,0).numpy()
    mask_u = np.ones_like(u).astype('float32')
    mask_v = np.ones_like(v).astype('float32')

    n_u, n_v = momentum_cnn(u, v, mask_u, mask_v, sampling=False)
    print(f'Returned n_u : {n_u.shape} n_v : {n_v.shape}')
    print('Max diff n_u', np.max(np.abs(n_u - n_u)),'- max diff n_v', np.max(np.abs(n_v - n_v)))
    print(f'Test successful')
