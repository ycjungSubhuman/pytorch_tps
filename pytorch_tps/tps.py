'''
Calculate warped image using control point manipulation on a thin plate (TPS)
Based on Herve Lombaert's 2006 web article
"Manual Registration with Thin Plates" 
(https://profs.etsmtl.ca/hlombaert/thinplates/)

Implementation by Yucheol Jung <ycjung@postech.ac.kr>
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class WarpTPS(nn.Module):
    '''
    Warps image using TPS
    
    Note)

    Thin plate spline surface height f is defined for every (x,y),

    f(x,y) = a[0] + a[1]*x + a[2]*y + \sum w_i*U(|P_i - (x,y)|_2)
    
    We construct two thin plates, dx plate and dy plate.
    Each plate contains dx values and dy values as height component.
    
    Args)
    normalize  		if true, normalize coordinates of control points
    			to [-1, 1]
    '''

    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, im, pts_before, pts_after):
        '''
        Deforms image according to movement of pts_before and pts_after

        Args)
        im		torch.Tensor object of size NxCxHxW
        pts_before	torch.Tensor object of size NxTx2 (T is # control pts)
        pts_after	torch.Tensor object of size NxTx2 (T is # control pts)
        '''
        # check input requirements
        assert(4 == im.dim())
        assert(3 == pts_after.dim())
        assert(3 == pts_before.dim())
        N = im.size()[0]
        assert(N == pts_after.size()[0] and N == pts_before.size()[0])
        assert(2 == pts_after.size()[2] and 2 == pts_before.size()[2])
        T = pts_after.size()[1]
        assert(T == pts_before.size()[1])
        H = im.size()[2]
        W = im.size()[3]

        if self.normalize:
            pts_after = pts_after.clone()
            pts_after[:,:,0] /= 0.5*W
            pts_after[:,:,1] /= 0.5*H
            pts_after -= 1
            pts_before = pts_before.clone()
            pts_before[:,:,0] /= 0.5*W
            pts_before[:,:,1] /= 0.5*H
            pts_before -= 1

        def construct_P():
            '''
            Consturcts matrix P of size NxTx3 where 
            P[n,i,0] := 1
            P[n,i,1:] := pts_after[n]
            '''
            # Create matrix P with same configuration as 'pts_after'
            P = pts_after.new_zeros((N, T, 3))
            P[:,:,0] = 1
            P[:,:,1:] = pts_after

            return P

        def calc_U(pt1, pt2):
            '''
            Calculate distance U between pt1 and pt2

            U(r) := r**2 * log(r)
            where
                r := |pt1 - pt2|_2

            Args)
            pt1 	torch.Tensor object, last dim is always 2
            pt2 	torch.Tensor object, last dim is always 2
            '''
            assert(2 == pt1.size()[-1])
            assert(2 == pt2.size()[-1])

            diff = pt1 - pt2
            sq_diff = diff**2
            sq_diff_sum = sq_diff.sum(-1)
            r = sq_diff_sum.sqrt()

            # Adds 1e-6 for numerical stability
            return (r**2) * torch.log(r + 1e-6)

        def construct_K():
            '''
            Consturcts matrix K of size NxTxT where 
            K[n,i,j] := U(|pts_after[n,i] - pts_after[n,j]|_2)
            '''

            # Assuming the number of control points are small enough,
            # We just use for-loop for easy-to-read code

            #   Create matrix K with same configuration as 'pts_after'
            K = pts_after.new_zeros((N, T, T))
            for i in range(T):
                for j in range(T):
                    K[:,i,j] = calc_U(pts_after[:,i,:], pts_after[:,j,:])

            return K

        def construct_L():
            '''
            Consturcts matrix L of size Nx(T+3)x(T+3) where 
            L[n] = [[ K[n]    P[n] ]]
                   [[ P[n]^T    0  ]]
            '''
            P = construct_P()
            K = construct_K()

            # Create matrix L with same configuration as 'K'
            L = K.new_zeros((N, T+3, T+3))

            # Fill L matrix
            L[:,:T,:T] = K
            L[:,:T,T:(T+3)] = P
            L[:,T:(T+3),:T] = P.transpose(1, 2)

            return L

        def construct_uv_grid():
            '''
            Returns H x W x 2 tensor uv with UV coordinate as its elements
            uv[:,:,0] is H x W grid of x values
            uv[:,:,1] is H x W grid of y values
            '''
            u_range = torch.arange(
                start=-1.0, end=1.0, step=2.0/W, device=im.device)
            assert(W == u_range.size()[0])
            u = u_range.new_zeros((H,W))
            u[:] = u_range

            v_range = torch.arange(
                start=-1.0, end=1.0, step=2.0/H, device=im.device)
            assert(H == v_range.size()[0])
            vt = v_range.new_zeros((W,H))
            vt[:] = v_range
            v = vt.transpose(0,1)

            return torch.stack([u,v], dim=2)

        L = construct_L()
        VT = pts_before.new_zeros((N, T+3, 2))
        # Use delta x and delta y as known heights of the surface
        VT[:,:T,:] = pts_before - pts_after

        # Solve Lx = VT
        #   x is of shape (N, T+3, 2)
        #   x[:,:,0] represents surface parameters for dx surface
        #       (dx values as surface height (z))
        #   x[:,:,1] represents surface parameters for dy surface
        #       (dy values as surface height (z))
        x, _ = torch.gesv(VT, L)

        uv = construct_uv_grid()
        uv_batch = uv.repeat((N,1,1,1))
        def calc_dxdy():
            '''
            Calculate surface height for each uv coordinate
            
            Returns NxHxWx2 tensor
            '''

            # control points of size NxTxHxWx2
            cp = uv.new_zeros((H,W,N,T,2))
            cp[:,:,:] = pts_after
            cp = cp.permute([2,3,0,1,4])

            U = calc_U(uv, cp) # U value matrix of size NxTxHxW
            w, a = x[:,:T,:], x[:,T:,:] # w is of size NxTx2, a is of size Nx3x2
            w_x, w_y = w[:,:,0], w[:,:,1] # NxT each
            a_x, a_y = a[:,:,0], a[:,:,1] # Nx3 each
            dx = (
                a_x[:,0].repeat((H,W,1)).permute(2,0,1) +
                torch.einsum('nhwd,nd->nhw', uv_batch, a_x[:,1:]) +
                torch.einsum('nthw,nt->nhw', U, w_x)) # dx values of NxHxW
            dy = (
                a_y[:,0].repeat((H,W,1)).permute(2,0,1) +
                torch.einsum('nhwd,nd->nhw', uv_batch, a_y[:,1:]) +
                torch.einsum('nthw,nt->nhw', U, w_y)) # dy values of NxHxW

            return torch.stack([dx, dy], dim=3)

        dxdy = calc_dxdy()
        flow_field = uv + dxdy

        return F.grid_sample(im, flow_field)


