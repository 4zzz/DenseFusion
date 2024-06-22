import argparse
import torch
from torch.autograd import Variable
from lib.network import PoseNet, PoseRefineNet
from datasets.bin_dataset.dataset import BinDataset
from lib.transformations import quaternion_matrix, quaternion_from_matrix
import numpy as np
from matplotlib import pyplot as plt
import copy
import json

parser = argparse.ArgumentParser()
parser.add_argument('--estimator_weights', type=str, help='estimator model weights')
parser.add_argument('--refiner_weights', type=str, help='estimator model weights')
parser.add_argument('--output_json', type=str, help='path where to save results')
parser.add_argument('--num_points', type=int, default=1000, help='number of points on the input pointcloud')
parser.add_argument('--num_obj', type=int, default=8, help='number of object classes in the dataset')
opt = parser.parse_args()

num_obj = opt.num_obj # 21
num_points =  opt.num_points # 1000
np.random.seed(40)
test_dataset = BinDataset(
    dataset_root='datasets/bin_dataset/Gajdosech_etal_2021_dataset/',
    mode='test',
    num_points=num_points,
    width=256,
    height=256,
    preload=False,
    return_transform_mat=True
)

dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

estimator = PoseNet(num_points = num_points, num_obj = num_obj)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.estimator_weights))
estimator.eval()

refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
refiner.cuda()
refiner.load_state_dict(torch.load(opt.refiner_weights))
refiner.eval()


def to_batch(t):
    return tuple([i.unsqueeze(0) for i in t])


def save_json(obj, file):
    json_object = json.dumps(obj, indent=4)
    with open(file, "w") as outfile:
        outfile.write(json_object)


def rotation_error(R1, R2):
  # params:
  # R1 - (3, 3) array representing a rotation matrix
  # R2 - (3, 3) array representing a rotation matrix
  # returns:
  # angle - angle between the two rotations in degrees
  R2a = R2
  R2b = np.matrix.copy(R2)
  R2b[:, :2] *= -1

  R_d = R1.T @ R2a
  error1 = np.arccos((np.trace(R_d) - 1)/2) * 180 / np.pi

  R_d = R1.T @ R2b
  error2 = np.arccos((np.trace(R_d) - 1)/2) * 180 / np.pi
  return min(error1, error2)

exr_files = []
estimated_r_err = []
estimated_t_err = []
refined_r_err = []
refined_t_err = []

results = []

ll = len(test_dataset)
for i in range(ll):
    cloud, choose, img, target, model_points, idx, target_mat, exr_file = test_dataset[i]
    cloud, choose, img, target, model_points, idx = to_batch((cloud, choose, img, target, model_points, idx))
    cloud, choose, img, target, model_points, idx = Variable(cloud).cuda(), \
                                                    Variable(choose).cuda(), \
                                                    Variable(img).cuda(), \
                                                    Variable(target).cuda(), \
                                                    Variable(model_points).cuda(), \
                                                    Variable(idx).cuda()
    exr_files.append(exr_file)
    pred_r, pred_t, pred_c, emb = estimator(img, cloud, choose, idx)

    bs = 1
    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

    pred_c = pred_c.view(bs, num_points)
    how_max, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(bs * num_points, 1, 3)
    points = cloud.view(bs * num_points, 1, 3)

    my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
    my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
    my_pred = np.append(my_r, my_t)

    est_mat = quaternion_matrix(my_r)
    est_mat[0:3, 3] = my_t

    est_t = my_t
    target_t = target_mat[0:3, 3] / 1000

    print('target_mat', target_mat)
    print('estimated trans', est_t, 'target_t', target_t)
    #exit()

    #print('estimated mat err', rotation_error(est_mat[0:3, 0:3], target_mat[0:3, 0:3]))

    iteration = 2
    for ite in range(0, iteration):
        T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
        my_mat = quaternion_matrix(my_r)
        R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
        my_mat[0:3, 3] = my_t

        new_cloud = torch.bmm((cloud - T), R).contiguous()
        pred_r, pred_t = refiner(new_cloud, emb, idx)
        pred_r = pred_r.view(1, 1, -1)
        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
        my_r_2 = pred_r.view(-1).cpu().data.numpy()
        my_t_2 = pred_t.view(-1).cpu().data.numpy()
        my_mat_2 = quaternion_matrix(my_r_2)

        my_mat_2[0:3, 3] = my_t_2

        my_mat_final = np.dot(my_mat, my_mat_2)
        my_r_final = copy.deepcopy(my_mat_final)
        my_r_final[0:3, 3] = 0
        my_r_final = quaternion_from_matrix(my_r_final, True)
        my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

        my_pred = np.append(my_r_final, my_t_final)
        my_r = my_r_final
        my_t = my_t_final

    refined_mat = quaternion_matrix(my_r)
    refined_mat[0:3, 3] = my_t
    refined_t = my_t

    #print('cloud shape', cloud.shape)


    def transform_points(points, tranformation_4x4):
        p = np.column_stack((points, np.ones(points.shape[0])))
        # print(tranformation_4x4)
        tp = (tranformation_4x4 @ p.T).T
        d = np.atleast_2d(tp[:, 3]).T
        n = tp  # (tp / d)
        return n[:, :3]


    def show_points(ptss, title, markers=['o', 'v', 's', '8']):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d', title=title)

        if type(ptss) is list:
            for i, pts in enumerate(ptss):
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], marker=markers[i])
        else:
            pts = ptss
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], marker='o')

    est_r_err = rotation_error(est_mat[0:3, 0:3], target_mat[0:3, 0:3])
    ref_r_err = rotation_error(refined_mat[0:3, 0:3], target_mat[0:3, 0:3])
    est_t_err = np.linalg.norm(est_t - target_t)
    ref_t_err = np.linalg.norm(refined_t - target_t)

    estimated_r_err.append(est_r_err)
    estimated_t_err.append(est_t_err)
    refined_r_err.append(ref_r_err)
    refined_t_err.append(ref_t_err)

    print('estimated rot err', est_r_err)
    print('refined rot err', ref_r_err)
    print('estimated trans err', est_t_err)
    print('refined trans err', ref_t_err)

    rr = {
        'i': i,
        'exr_file': exr_file,
        'class': idx.cpu()[0][0].item(),
        'estimated': {
            'rotation_err': est_r_err,
            'translation_err': est_t_err,
        },
        'refined': {
            'rotation_err': ref_r_err,
            'translation_err': ref_t_err,
        }
    }
    print(rr)
    results.append(rr)


    #exit()

    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d', title='viy')
    #pts = cloud.cpu()[0]
    #ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], marker='o')
    #pts = transform_points(model_points.cpu()[0].numpy(), est_mat)
    #pts -= my_t
    #ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], marker='v')

    #show_points([cloud.cpu()[0], transform_points(model_points.cpu()[0].numpy(), est_mat)], title='estimated')
    #show_points([cloud.cpu()[0], transform_points(model_points.cpu()[0].numpy(), refined_mat)], title='refined')

    #plt.show()
    #exit()

    #my_result_wo_refine.append(my_pred.tolist())

print('Mean estimated rotation error', np.array(estimated_r_err).mean())
print('Mean refined rotation error', np.array(refined_r_err).mean())
print('Mean estimated translation error', np.array(estimated_t_err).mean())
print('Mean refined translation error', np.array(refined_t_err).mean())

save_json(results, opt.output_json)

with open('table.tex', 'w') as f:
    f.write('''
\\begin{table}
\\begin{center}
\\begin{tabular}{ |c||c|c|c|c| }
\\hline
 & \\multicolumn{2}{|c|}{Základný odhad} & \\multicolumn{2}{|c|}{Vylepšený odhad} \\\\
 \\hline
 Vzorka & Chyba rotácie & Chyba posunu & Chyba rotácie & Chyba posunu \\\\
 \\hline
 \\hline\n''')
    for i in range(len(estimated_r_err)):
        pp = exr_files[i].split('/')[-2:]
        pp1 = pp[1].split('_')[:2]
        name = f'{'\_'.join(pp[0].split('_'))}/{'\_'.join(pp1)}'
        f.write(f'    {name} & {estimated_r_err[i]:.4f} & {estimated_t_err[i]:.4f} & {refined_r_err[i]:.4f} & {refined_t_err[i]:.4f} \\\\  \\hline\n')
    f.write('''\\end{tabular}
\\end{center}
\\caption{Odhad pózy na testovacej množine}
\\label{tbl_best_conf}
\\end{table}''')

