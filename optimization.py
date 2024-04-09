import matplotlib.pyplot as plt
import tifffile
import torch
from tqdm import trange
import torch.nn.functional as F
from crip.preprocess import *
from utils import *
import os


class GCC(torch.nn.Module):
    """
    global normalized cross correlation (sqrt)
    """

    def __init__(self):
        super(GCC, self).__init__()

    def forward(self, I, J):
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I * J
        # average value
        I_ave, J_ave = I.mean(), J.mean()
        I2_ave, J2_ave = I2.mean(), J2.mean()
        IJ_ave = IJ.mean()

        cross = IJ_ave - I_ave * J_ave
        I_var = I2_ave - I_ave.pow(2)
        J_var = J2_ave - J_ave.pow(2)

        cc = cross / (I_var.sqrt() * J_var.sqrt() + torch.finfo(float).eps)  # 1e-5

        return -1.0 * cc + 1


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion = GCC()  # similarity measure

    def forward(self, proj_0, proj_m, V_0, U_0, V_m, U_m):
        grid = torch.stack((U_0 / nu * 2, V_0 / nv * 2), 2).unsqueeze(0)  # (1, nv, nu, 2)  <= (N, H, W, 2)
        inputv = proj_0.reshape(1, 1, nv, nu)  # (1, 1, nv, nu)  <= (N, C, H, W)
        output = F.grid_sample(inputv, grid).squeeze()

        img1 = (output).unsqueeze(0).unsqueeze(0)
        img2 = (proj_m[V_m, U_m]).unsqueeze(0).unsqueeze(0)

        return self.criterion(img1, img2)


def imgROI(img):
    H, W = img.shape
    mask = torch.zeros_like(img)
    H_PC, W_PC = 0.8, 0.8
    mask[int(H * (1-H_PC)/2): int(H * (1+H_PC)/2),
         int(W * (1-W_PC)/2): int(W * (1+W_PC)/2)] = 1

    return img * mask


if __name__ == '__main__':
    
    Views = 400
    du = dv = 0.4
    nu = nv = 750
    TotalScanAngle = 200  # inverse direction, same as config file
    ImageRotation = 0  # rotate again, commonly don't change
    SID = 750
    SDD = 1060

    ### Main
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    proj_0_volume = torch.tensor(tifffile.imread('./proj/proj_head1_ideal.tif'), dtype=torch.float32)  # ideal
    proj_m_volume = torch.tensor(tifffile.imread('./proj/proj_head1_measured.tif'), dtype=torch.float32)  # measured

    u_idx_vec = torch.arange(0, nu, 1) + 1 - (nu + 1) / 2  # u <- u - uc
    v_idx_vec = torch.arange(0, nv, 1) + 1 - (nv + 1) / 2  # v <- v - vc

    V, U = torch.meshgrid(u_idx_vec, v_idx_vec, indexing='ij')
    V = V.to(device); U = U.to(device)
    U_m = (U + (nu + 1) / 2 - 1).round().type(torch.long)
    V_m = (V + (nv + 1) / 2 - 1).round().type(torch.long)

    u_length = - U * du  # length of u direction
    v_length = + V * dv  # length of v direction

    R_s = SID  # SID
    R_d = SDD - SID  # SDD - SID
    th3 = np.sqrt(2 * du**2 - 2 * du * du * np.cos(np.deg2rad(1))) / (R_s + R_d)
    loss0_list, loss1_list, loss2_list = [], [], []
    params_list_pred = [[] for _ in range(7)]  # empty

    for v in trange(Views):

        proj_0 = proj_0_volume[v].to(device)
        proj_m = proj_m_volume[v].to(device)

        IV_7 = torch.zeros(7, dtype=torch.float32).to(device)
        IV_7.requires_grad = True
        optimizer = torch.optim.Rprop([IV_7], lr=0.001, etas=(0.3, 1.2), step_sizes=(1e-04, 50))

        criterion = Loss()

        ### Optimize for one view
        for i in range(50):
            delta_u = IV_7[0] + IV_7[1] * u_length / (R_s + R_d)  # delta_u in length
            delta_v = IV_7[5] + IV_7[1] * v_length / (R_s + R_d)  # delta_v in length
            U_0 = U - delta_u / du
            V_0 = V + delta_v / dv

            loss = criterion(proj_0, proj_m, V_0, U_0, V_m, U_m)
            loss0_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for i in range(20):
            delta_u = IV_7[0] + IV_7[1] * u_length / (R_s + R_d) + IV_7[2]*1 * v_length  # delta_u in length
            delta_v = IV_7[5] + IV_7[1] * v_length / (R_s + R_d) + IV_7[6]*1 * u_length  # delta_v in length
            U_0 = U - delta_u / du
            V_0 = V + delta_v / dv

            loss = criterion(proj_0, proj_m, V_0, U_0, V_m, U_m)
            loss1_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for i in range(30):

            # with torch.no_grad():
            #     torch.clamp_(IV_7[3], min=-th3, max=th3)
            #     torch.clamp_(IV_7[4], min=-th3, max=th3)

            delta_u = IV_7[0] + IV_7[1] * u_length / (R_s + R_d) + IV_7[2] * 1 * v_length \
                      + IV_7[3] / (R_s + R_d) * u_length ** 2 + IV_7[4] / (R_s + R_d) * u_length * v_length  # delta_u in length
            delta_v = IV_7[5] + IV_7[1] * v_length / (R_s + R_d) + IV_7[6] * 1 * u_length \
                      + IV_7[4] / (R_s + R_d) * v_length ** 2 + IV_7[3] / (R_s + R_d) * u_length * v_length  # delta_v in length
            U_0 = U - delta_u / du
            V_0 = V + delta_v / dv

            loss = criterion(proj_0, proj_m, V_0, U_0, V_m, U_m)
            loss2_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ### Pred
        params_pred = IV_7.cpu().detach().numpy()

        for i, params in enumerate(params_list_pred):
            params.append(params_pred[i])

    params_list_pred = np.array(params_list_pred)

    ### direction
    append_params = np.array([-1, 1, -du, -du, du, 1, -du])
    for i in range(7):
        params_list_pred[i] = params_list_pred[i] * append_params[i]

    ### plot
    for i in range(7):
        plt.plot(range(Views), params_list_pred[i], '--.')
        plt.legend([f'pred param{i}']), plt.xlabel('view'), plt.ylabel('mm')
        plt.show()

    ## Save
    pred_path = r'./params_pred'
    os.makedirs(pred_path, exist_ok=True)
    for i, pNum in enumerate([0,1,2,3,4,5,7]):
        param_list = save_paramsFile(params_list_pred[i], f'{pred_path}/param{pNum}_list.jsonc')

    ### generate pMatrix
    anglePerView = TotalScanAngle / Views
    
    params_list_pred = np.array(params_list_pred)
    x_d_list, e_u_list, e_v_list = [], [], []
    for v in range(Views):
        beta = np.radians(anglePerView * v + ImageRotation)
        vec_r = np.array([np.cos(beta),   np.sin(beta),   0])
        vec_b = np.array([-np.sin(beta),  np.cos(beta),   0])
        vec_a = np.array([0,              0,              1])
    
        params_pred = params_list_pred[:, v]
        x_d = vec_b * params_pred[0] + vec_r * params_pred[1] + vec_a * params_pred[5]
        e_u = - du * (vec_r * params_pred[3] + vec_a * params_pred[6])
        e_v = + dv * (vec_r * params_pred[4] + vec_b * params_pred[2])
    
        x_d_list.append(x_d), e_u_list.append(e_u), e_v_list.append(e_v)

    pred_pmatrix_path = r'./params_pmatrix'
    os.makedirs(pred_pmatrix_path, exist_ok=True)
    gen_pMatrix(projOffsetD_list=x_d_list,
                projOffsetS_list=np.zeros(Views),
                projOffsetEU_list=e_u_list,
                projOffsetEV_list=e_v_list,
                SID=SID, SDD=SDD, du=du, dv=dv, nu=nu, nv=nv,
                Views=Views, TotalScanAngle=TotalScanAngle,
                postfix='_predOPT',
                path=pred_pmatrix_path)
