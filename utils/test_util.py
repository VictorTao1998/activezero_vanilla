"""
Author: Isabella Liu 8/15/21
Feature: Util functions when testing
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import open3d as o3d

from .config import cfg

from utils.np_utils import depth2pts_np
from matplotlib import cm

MAX_DIST = 10


def load_from_dataparallel_model(model_pth, sub_model_name):
    # original saved file with DataParallel
    state_dict = torch.load(model_pth)[sub_model_name]
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def calc_left_ir_depth_from_rgb(k_main, k_l, rt_main, rt_l, rgb_depth):
    rt_lmain = rt_l @ np.linalg.inv(rt_main)
    h, w = rgb_depth.shape
    irl_depth = cv2.rgbd.registerDepth(
        k_main, k_l, None, rt_lmain, rgb_depth, (w, h), depthDilation=True
    )
    irl_depth[np.isnan(irl_depth)] = 0
    irl_depth[np.isinf(irl_depth)] = 0
    irl_depth[irl_depth < 0] = 0
    return irl_depth


def save_img(
    log_dir,
    prefix,
    pred_disp_np,
    gt_disp_np,
    pred_disp_err_np,
    pred_depth_np,
    gt_depth_np,
    realsense_depth_np,
    pred_depth_err_np,
    pred_conf,
    cam_intrinsic,
    cam_extrinsic=np.eye(4),
):
    disp_path = os.path.join("pred_disp", prefix) + ".png"
    disp_gt_path = os.path.join("gt_disp", prefix) + ".png"
    disp_abs_err_cm_path = os.path.join("pred_disp_abs_err_cmap", prefix) + ".png"
    depth_path = os.path.join("pred_depth", prefix) + ".png"
    depth_gt_path = os.path.join("gt_depth", prefix) + ".png"
    depth_abs_err_cm_path = os.path.join("pred_depth_abs_err_cmap", prefix) + ".png"
    pred_conf_path = os.path.join("pred_conf", prefix) + ".png"
    pred_pcd_path = os.path.join("pred_pcd", prefix) + ".ply"
    pred_pcd_err_path = os.path.join("pred_pcd", f'{prefix}_err') + ".ply"
    realsense_pcd_path = os.path.join("realsense_pcd", prefix) + ".ply"
    realsense_pcd_err_path = os.path.join("realsense_pcd", f'{prefix}_err') + ".ply"
    gt_pcd_path = os.path.join("gt_pcd", prefix) + ".ply"

    # Save predicted images
    masked_pred_disp_np = np.ma.masked_where(
        pred_disp_np == -1, pred_disp_np
    )  # mark background as red
    custom_cmap = plt.get_cmap("viridis").copy()
    custom_cmap.set_bad(color="red")
    plt.imsave(
        os.path.join(log_dir, disp_path),
        masked_pred_disp_np,
        cmap=custom_cmap,
        vmin=0,
        vmax=cfg.ARGS.MAX_DISP,
    )

    plt.imsave(
        os.path.join(log_dir, pred_conf_path),
        pred_conf,
        cmap=custom_cmap,
        vmin=0.0,
        vmax=1.0,
    )

    # Save points clouds
    # gt
    gt_pts = depth2pts_np(gt_depth_np * 1000, cam_intrinsic, cam_extrinsic)
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_pts)
    o3d.io.write_point_cloud(os.path.join(log_dir, gt_pcd_path), gt_pcd)

    # pred
    pred_conf_color = custom_cmap(pred_conf)[..., :3]
    pred_pts = depth2pts_np(pred_depth_np * 1000, cam_intrinsic, cam_extrinsic)
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_pts)
    pred_pcd.colors = o3d.utility.Vector3dVector(pred_conf_color.reshape(-1, 3))
    o3d.io.write_point_cloud(os.path.join(log_dir, pred_pcd_path), pred_pcd)

    dists = pred_pcd.compute_point_cloud_distance(gt_pcd)
    dists = np.asarray(dists)
    dists = np.clip(dists, 0, MAX_DIST) / (MAX_DIST - 0)
    dists_color = cm.jet(dists)[..., :3]
    pred_pcd.colors = o3d.utility.Vector3dVector(dists_color)
    o3d.io.write_point_cloud(os.path.join(log_dir, pred_pcd_err_path), pred_pcd)

    # realsense
    realsense_pts = depth2pts_np(realsense_depth_np * 1000, cam_intrinsic, cam_extrinsic)
    realsense_pcd = o3d.geometry.PointCloud()
    realsense_pcd.points = o3d.utility.Vector3dVector(realsense_pts)
    o3d.io.write_point_cloud(os.path.join(log_dir, realsense_pcd_path), realsense_pcd)

    dists = realsense_pcd.compute_point_cloud_distance(gt_pcd)
    dists = np.asarray(dists)
    dists = np.clip(dists, 0, MAX_DIST) / (MAX_DIST - 0)
    dists_color = cm.jet(dists)[..., :3]
    realsense_pcd.colors = o3d.utility.Vector3dVector(dists_color)
    o3d.io.write_point_cloud(os.path.join(log_dir, realsense_pcd_err_path), realsense_pcd)


    masked_pred_depth_np = np.ma.masked_where(
        pred_depth_np == -1, pred_depth_np
    )  # mark background as red
    plt.imsave(
        os.path.join(log_dir, depth_path),
        masked_pred_depth_np,
        cmap=custom_cmap,
        vmin=0,
        vmax=1.25,
    )
    np.save(os.path.join(log_dir, f"{depth_path[:-4]}.npy"), pred_depth_np)
    np.save(os.path.join(log_dir, f"{depth_gt_path[:-4]}.npy"), gt_depth_np)

    # Save ground truth images
    masked_gt_disp_np = np.ma.masked_where(
        gt_disp_np == -1, gt_disp_np
    )  # mark background as red
    plt.imsave(
        os.path.join(log_dir, disp_gt_path),
        masked_gt_disp_np,
        cmap=custom_cmap,
        vmin=0,
        vmax=cfg.ARGS.MAX_DISP,
    )
    masked_gt_depth_np = np.ma.masked_where(
        gt_depth_np == -1, gt_depth_np
    )  # mark background as red
    plt.imsave(
        os.path.join(log_dir, depth_gt_path),
        masked_gt_depth_np,
        cmap=custom_cmap,
        vmin=0,
        vmax=1.25,
    )

    # Save error images
    plt.imsave(os.path.join(log_dir, disp_abs_err_cm_path), pred_disp_err_np)
    plt.imsave(os.path.join(log_dir, depth_abs_err_cm_path), pred_depth_err_np)
    plt.close("all")


def save_gan_img(img_outputs, path, nrow=2, ncol=2):
    # Create plt figure
    fig = plt.figure(figsize=(24, 12))
    count = 1
    for tag, dict_value in img_outputs.items():
        for subtag, img_value in dict_value.items():
            img = img_value[0].permute(1, 2, 0).cpu().detach().numpy()  # [H, W, 3]
            img = (img + 1) / 2  # normalize to [0,1]
            img_title = f"{tag}-{subtag}"
            # Add image
            fig.add_subplot(nrow, ncol, count)
            plt.imshow(img)
            plt.axis("off")
            plt.title(img_title, fontsize=20)
            count += 1
    plt.tight_layout()
    plt.savefig(path, pad_inches=0)
    plt.close("all")


def save_obj_err_file(
    total_obj_disp_err, total_obj_depth_err, total_obj_depth_4_err, log_dir
):
    result = np.hstack(
        (
            np.arange(cfg.SPLIT.OBJ_NUM)[:, None].astype(int),
            total_obj_disp_err[:, None],
            total_obj_depth_err[:, None],
            total_obj_depth_4_err[:, None],
        )
    )
    result = result.astype("str").tolist()
    head = [["     ", "disp_err", "depth_err", "depth_err_4"]]
    result = head + result

    err_file = open(os.path.join(log_dir, "obj_err.txt"), "w")
    for line in result:
        content = " ".join(line)
        err_file.write(content + "\n")
    err_file.close()


def save_prob_volume(
    prob_volume,
    log_dir,
    prefix,
    threshold=0.01,
):
    h, w, d = prob_volume.shape
    custom_cmap = plt.get_cmap("jet")
    mask = (prob_volume > threshold).reshape(-1)

    color = custom_cmap(prob_volume)[..., :3].reshape(-1, 3)

    coor = np.zeros((h, w, d, 3))
    for i in range(h):
        coor[i, :, :, 0] = i
    for i in range(w):
        coor[:, i, :, 1] = i
    for i in range(d):
        coor[:, :, i, 2] = i

    coor = coor.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coor[mask])
    pcd.colors = o3d.utility.Vector3dVector(color[mask])

    o3d.io.write_point_cloud(os.path.join(log_dir, f"prob_volume/{prefix}.ply"), pcd)