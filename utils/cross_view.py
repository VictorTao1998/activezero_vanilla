"""
Author: Isabella Liu 2/26/22
Feature: Cross-view reprojection loss
"""

import torch


def depth2pts_tensor(depth: torch.tensor, cam_intrinsic: torch.tensor, cam_extrinsic: torch.tensor):
    """
    depth_map: [bs, 1, H, W]
    cam_intrinsic: [bs, 3, 3]
    cam_extrinsic: [bs, 4, 4]
    return:
        world_points: [bs, N, 3]
    """
    bs, _, h, w = depth.shape

    # Get pixel grid in tensor
    x_linespace = torch.linspace(0.5, w - 0.5, w)
    y_linespace = torch.linspace(0.5, h - 0.5, h)
    x_coordinates, y_coordinates = torch.meshgrid(x_linespace, y_linespace, indexing='xy')
    x_coordinates = torch.reshape(x_coordinates, (1, -1))
    y_coordinates = torch.reshape(y_coordinates, (1, -1))
    ones = torch.ones_like(x_coordinates).type(torch.float64)
    grid = torch.cat((x_coordinates, y_coordinates, ones), dim=0).unsqueeze(0).repeat(bs, 1, 1).to(depth.device)  # [bs, 3, N]

    uv = torch.matmul(torch.linalg.inv(cam_intrinsic), grid)  # [bs, 3, N]
    cam_points = uv * torch.reshape(depth, (bs, 1, -1))  # [bs, 3, N]

    R = cam_extrinsic[:, :3, :3]  # [bs, 3, 3]
    t = cam_extrinsic[:, :3, 3:4]  # [bs, 3, 1]

    world_points = torch.matmul(torch.linalg.inv(R), cam_points - t).permute(0, 2, 1).contiguous()  # [bs, N, 3]

    return world_points


def pts2depth_tensor(img_h, img_w, pts, cam_intrinsic, cam_extrinsic):
    """
    img_h, img_w: height and width of depth map
    pts: [bs, N, 3]
    cam_intrinsic: [bs, 3, 3]
    cam_extrinsic: [bs, 4, 4]
    return:
        depth_img: [bs, 1, H, W]
    """
    # TODO currently only support bs=1, since pts # will vary between different batches after the filter
    bs, N, _ = pts.shape
    R = cam_extrinsic[:, :3, :3]  # [bs, 3, 3]
    t = cam_extrinsic[:, :3, 3:4]  # [bs, 3, 1]

    pts = pts.permute(0, 2, 1).contiguous()  # [bs, 3, N]
    cam_points = torch.matmul(R, pts) + t

    #     # Sort the points according to the dist to cam origin, important when two points map to the same pixel
    #     cam_points = cam_points[:, :,cam_points[:, -1].argsort()]

    depth = cam_points[:, -1, :]  # [bs, N]

    uv = cam_points / depth  # [bs, 3, N]
    uv = torch.matmul(cam_intrinsic, uv)  # [bs, 3, N]
    uv = uv[:, :2, :]  # [bs, 2, N]

    uv_round = torch.round(uv - 0.5).type(torch.long)  # [bs, 2, N]
    uv_round = uv_round.permute(0, 2, 1).contiguous()  # [bs, N, 2]

    mask_u = (uv_round[:, :, 0] >= 0) * (uv_round[:, :, 0] < img_w)  # [bs, N]
    uv_round = uv_round[mask_u].unsqueeze(0)  # [bs, N, 2]
    depth = depth[mask_u].unsqueeze(0)  # [bs, N]

    mask_v = (uv_round[:, :, 1] >= 0) * (uv_round[:, :, 1] < img_h)  # [bs, N]
    uv_round = uv_round[mask_v]  # [N, 2]
    depth = depth[mask_v]  # [N,]
    depth = depth.type(torch.float32)

    depth_img = torch.zeros((bs, 1, img_h, img_w)).to(pts.device)
    depth_img[:, :, uv_round[:, 1], uv_round[:, 0]] = depth

    return depth_img


