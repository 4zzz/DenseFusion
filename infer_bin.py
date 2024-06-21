import argparse
import torch
from torch.autograd import Variable
from lib.network import PoseNet, PoseRefineNet
from datasets.bin_dataset.dataset import BinDataset
from lib.transformations import quaternion_matrix, quaternion_from_matrix
import numpy as np
from matplotlib import pyplot as plt
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--estimator_weights', type=str, help='estimator model weights')
parser.add_argument('--refiner_weights', type=str, help='estimator model weights')
parser.add_argument('--positions', type=str, help='exr file with structured pointcloud')
opt = parser.parse_args()

num_obj = 8
num_points = 1000
test_dataset = BinDataset(dataset_root='datasets/bin_dataset/Gajdosech_etal_2021_dataset/', mode='test', num_points=1000, width=256, height=256, preload=False)

dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

estimator = PoseNet(num_points = num_points, num_obj = num_obj)

#torch.onnx.export(estimator, X_test, 'iris.onnx', input_names=["features"], output_names=["logits"])

estimator.cuda()
estimator.load_state_dict(torch.load(opt.estimator_weights))
estimator.eval()

refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
refiner.cuda()
refiner.load_state_dict(torch.load(opt.refiner_weights))
refiner.eval()

def to_batch(t):
    #points, choose, img, target, model_points, idx = t

    #points = points.unsqueeze(0)
    #print('points shape', points.shape)
    #choose = torch.tensor([choose])
    #img = torch.tensor([img])
    #target = torch.tensor([target])
    #model_points = torch.tensor([model_points])
    #idx = torch.tensor([idx])

    return tuple([i.unsqueeze(0) for i in t])

#for i in dataloader:
#    points, choose, img, target, model_points, idx = i
#    print('points shape', points.shape)
#    exit()

sample_i = np.random.randint(0, len(test_dataset))

cloud, choose, img, target, model_points, idx = to_batch(test_dataset[sample_i])
cloud, choose, img, target, model_points, idx = Variable(cloud).cuda(), \
                                                Variable(choose).cuda(), \
                                                Variable(img).cuda(), \
                                                Variable(target).cuda(), \
                                                Variable(model_points).cuda(), \
                                                Variable(idx).cuda()


#print('cloud.shape', cloud.shape)
#print('choose.shape', choose.shape)
#print('img.shape', img.shape)
#print('target.shape', target.shape)
#print('model_points.shape', model_points.shape)
#print('idx', idx)

#torch.onnx.export(estimator, (img, cloud, choose, idx), 'densefusion.onnx', input_names=["features"], output_names=["logits"])

pred_r, pred_t, pred_c, emb = estimator(img, cloud, choose, idx)


print('infer result:')
print('\tpred_r shape:', pred_r.shape)
print('\tpred_t shape:', pred_t.shape)
print('\tpred_c shape:', pred_c.shape)

bs = 1
pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

pred_c = pred_c.view(bs, num_points)
how_max, which_max = torch.max(pred_c, 1)
pred_t = pred_t.view(bs * num_points, 1, 3)
points = cloud.view(bs * num_points, 1, 3)

my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
my_pred = np.append(my_r, my_t)

print('my_r', my_r)
print('my_t', my_t)

est_mat = quaternion_matrix(my_r)
est_mat[0:3, 3] = my_t

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

print('cloud shape', cloud.shape)


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

exit()

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d', title='viy')
#pts = cloud.cpu()[0]
#ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], marker='o')
pts = transform_points(model_points.cpu()[0].numpy(), est_mat)
#pts -= my_t
#ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], marker='v')

show_points([cloud.cpu()[0], transform_points(model_points.cpu()[0].numpy(), est_mat)], title='estimated')
show_points([cloud.cpu()[0], transform_points(model_points.cpu()[0].numpy(), refined_mat)], title='refined')

plt.show()

#my_result_wo_refine.append(my_pred.tolist())
