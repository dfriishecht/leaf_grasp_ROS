"""
Helper module to convert a depth and semantic segmentation data into a
reconstructed leaf point cloud.
"""

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter


def apply_depth_mask(pointcloud_path, mask_path, image_path, plot=False):
    """
    Function for applying leaf mask over depth image. Expects 1080x1440 resolution.

    Args:
        pointcloud_path (string): Filepath to pointcloud with .pcd format.

        mask_path (string): Filepath to 1080x1440 image mask.

        image_path (string): Filepath to the original captured 1080x1440
                rectified camera image.

        plot (bool): Optional input to plot this function's result.

    Returns:
        masked_points (numpy array): A 1080x1440x4 numpy array containing data about the
                segmented plants. The first three 3rd dimension indices provide the
                real-world x, y, and z position of each point. The final index provides
                the leaf id that each point belongs to. If the id is 0, then the point is
                not part of a leaf.
    """

    # Load pointcloud from file and save its point position and original color as two numpy arrays
    pcd_path = pointcloud_path
    pcd_load = o3d.io.read_point_cloud(pcd_path, format="pcd")
    pcd_colors = np.asarray(pcd_load.colors)
    fill = np.zeros((1555200 - len(pcd_colors), 3))
    pcd_colors = np.concatenate((pcd_colors, fill), axis=0)

    # Load mask to be used for leaf segmentation
    mask = mask_path
    mask_one_d = mask.copy()
    mask_one_d = np.reshape(mask_one_d, (1555200, 1))


    # Use leaf mask to remove all non-leaf portions of the pointcloud
    mask_erode = np.reshape(mask, (1555200, 1))
    pcd_colors = pcd_colors * mask_erode

    # Pull depth data from point cloud
    depth_points = np.concatenate((np.asarray(pcd_load.points), fill), axis=0)
    depth = depth_points[:, 2]
    depth = np.reshape(depth, (1080, 1440, 1))

    # Set all non-leaf point position data to 0
    index = np.argwhere(mask_erode != 0)
    mask_erode[index] = 1
    mask_erode = np.reshape(mask_erode, (1080, 1440, 1))
    depth_masked = depth * mask_erode

    xy_pos = np.asarray(depth_points)[:, 0:2]
    xy_pos = np.reshape(xy_pos, (1080, 1440, 2))
    xy_pos_masked = mask_erode * xy_pos

    # Reapply unique mask ids as integer values instead of rgb color values
    mask_colors = np.unique(mask_one_d, axis=0)
    print(f"unique mask colors: {len(mask_colors)}")
    color_index = np.zeros(shape=(1555200, 1))
    i = 1
    for color in mask_colors:
        index = np.all(mask_one_d == color, axis=-1)
        color_index[index] = i
        i += 1
    color_index = np.reshape(color_index, (1080, 1440, 1)).astype("uint8")
    masked_points = np.concatenate((xy_pos_masked, depth_masked, color_index), axis=2)

    # Plot all the data stored in 'masked_points'
    if plot is True:

        _, ax = plt.subplot_mosaic(
            [["x", "y", "z", "mask"]], figsize=(15, 10)
        )

        ax["x"].imshow(masked_points[:, :, 0])
        ax["x"].set_title("X Coords (m)")
        ax["y"].imshow(masked_points[:, :, 1])
        ax["y"].set_title("Y Coords (m)")
        ax["z"].imshow(masked_points[:, :, 2])
        ax["z"].set_title("Depth (m)")
        ax["mask"].imshow(masked_points[:, :, 3])
        ax["mask"].set_title("Leaf ID")
        plt.show()

    return masked_points, depth_points


def extract_pcd(masked_points):
    """
    Pulls out point position data from masked_points, and coverts the arrays into
    an open3d point cloud object.

    Args:
        masked_points (3d numpy array): A 1080x1440x4 numpy array containing data about the
                segmented plants.

    Returns:
        pcd (o3d point cloud): Point cloud containing position data from masked_points.
    """
    pcd = o3d.geometry.PointCloud()
    points = np.reshape(masked_points[:, :, 0:3], (1555200, 3))
    points = np.delete(points, np.argwhere(points == [0, 0, 0]), axis=0)
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def compute_normals(point_cloud):
    """
    Estimates the normal vector of every point in a point cloud.
    The resulting normal vectors will all have a magnitude of one, and their
    coordinates saved relative to the point they correspond to. Each point
    will be its respective normal vector's "(0,0,0)".

    Agrs:
        point_cloud (open3d point cloud object): The desired point cloud to
                compute normals for.

    Returns:
        point_cloud (open3d point cloud object): The input point cloud with
        normal vector data now included. Can be called with point_cloud.normals.
    """
    print("Computing Normals!")
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    return point_cloud
