import json
import os

import gdown
import imageio
import numpy as np
import torch


trans_t = lambda t: torch.tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
).float()

rot_phi = lambda phi: torch.tensor(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
).float()

rot_theta = lambda th: torch.tensor(
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ]
).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).float()
        @ c2w
    )
    return c2w

def load_neurofluid_data(
    datadir: str,
    scene_name: str,
    train_skip: int,
    val_skip: int,
    test_skip: int,
    cam_scale_factor: float,
    white_bkgd: bool,
):
    basedir = os.path.join(datadir, "data_release", scene_name)
    cam_trans = np.diag(np.array([1, -1, -1, 1], dtype=np.float32))

    images = []
    extrinsics = []
    counts = [0]
    splits = ["train", "val", "test"]
    metas = {}

    file_order = []
    for s in splits:
        for folder in os.listdir(basedir):
            if folder[:4] != 'view':
                continue
            sub_dir = os.path.join(basedir, folder)
            if len(file_order) < 5:
                file_order.append(folder)

            with open(os.path.join(sub_dir, "transforms_{}.json".format(s)), "r") as fp:
                if s in metas.keys():
                    metas[s]["frames"].extend(json.load(fp)["frames"])
                else:
                    metas[s] = json.load(fp)
                    view_datanum = len(metas[s]["frames"])

    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []

        if s == "train":
            skip = train_skip
        elif s == "val":
            skip = val_skip
        elif s == "test":
            skip = test_skip

        i = 0
        for folder in file_order:
            sub_dir = os.path.join(basedir, folder)
            start = i * view_datanum
            end = (i + 1) * view_datanum
            for frame in meta["frames"][start:end:skip]:
                fname = os.path.join(sub_dir, frame["file_path"] + ".png")
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame["transform_matrix"]))
            i += 1
        imgs = (np.array(imgs) / 255.0).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        images.append(imgs)
        extrinsics.append(poses)
        counts.append(counts[-1] + imgs.shape[0])

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    images = np.concatenate(images, 0)

    extrinsics = np.concatenate(extrinsics, 0)

    extrinsics[:, :3, 3] *= cam_scale_factor
    extrinsics = extrinsics @ cam_trans

    h, w = imgs[0].shape[:2]
    num_frame = len(extrinsics)
    i_split += [np.arange(num_frame)]

    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
    intrinsics = np.array(
        [
            [[focal, 0.0, 0.5 * w], [0.0, focal, 0.5 * h], [0.0, 0.0, 1.0]]
            for _ in range(num_frame)
        ]
    )
    image_sizes = np.array([[h, w] for _ in range(num_frame)])

    render_poses = torch.stack(
        [
            pose_spherical(angle, -30.0, 4.0) @ cam_trans
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )
    render_poses[:, :3, 3] *= cam_scale_factor
    near = 2.0
    far = 6.0

    if white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    else:
        images = images[..., :3]

    return (
        images,
        intrinsics,
        extrinsics,
        image_sizes,
        near,
        far,
        (-1, -1),
        i_split,
        render_poses,
        view_datanum,
    )
