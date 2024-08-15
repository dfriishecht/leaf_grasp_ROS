#!/usr/bin/env python

"""
Node for sending optimal grasp location information

SUBSCRIBES TO:
    /depth_image (depth): float32[] imageData
    /leaves_masks (masks): float32[] imageData

PUBLISHES TO:
    /grasp_loc (grasp): geometry_msgs/Vector3 grasp      -> Grasp location vector
                        geometry_msgs/Vector3 norm       -> Normal vector at grasp point
                        geometry_msgs/Vector3 approach   -> Robot approach vector
    
    contains leaf grasp and approach vector dimensional data, relative to the left-camera frame

"""
import sys
import os
import rospy
import argparse
import glob
import numpy as np

HOME_DIR = os.path.expanduser('~')
print(os.getcwd())
print(HOME_DIR)

from tqdm import tqdm
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Point
from message_filters import ApproximateTimeSynchronizer, Subscriber
import open3d as o3d
import time
import skfmm
import cv2 as cv
from raftstereo.msg import depth
from yoloV8_seg.msg import masks
from leaf_grasp_ROS.msg import grasp
from paretoset import paretoset

from submodules.plant_pcd_helpers import apply_depth_mask, compute_normals
from submodules.mask_helpers import clean_mask, mean_mask_depth, find_tall_leaves, get_centroids, compute_minmax_dist
from submodules.conv_helpers import get_kernels, compute_graspable_areas

class LeafGrasp:
    def __init__(self):
        """
        Setup paramters upon node startup.
        """

        self.depth_sub = Subscriber('/depth_image', depth)
        self.mask_sub = Subscriber('/leaves_masks', masks)
        #rospy.Subscriber('/theia/left/image_rect_color', Image, self.recieve_image, queue_size=2) Uncomment if you want to also receive rgb image data.

        self.pub = rospy.Publisher('/grasp_loc', grasp, queue_size=2)

        self.combine = ApproximateTimeSynchronizer([self.mask_sub, self.depth_sub], queue_size=1, slop=0.05)
        self.combine.registerCallback(self.image_callback)
        self.init_()

    def recieve_image(self, image): #Only used if the Image subscriber above is uncommented.
        """
        Recieve rectified rgb left-camera image.
        """

        print("Left camera image recieved!")
        image = np.ndarray(shape=(self.img_height, self.img_width, 3), dtype=np.uint8)
        self.image = image    

    def init_(self):
        """
        Re-initialize node once it has run.
        """

        rospy.set_param('/leaf_done', False)


        self.img_width = 1440
        self.img_height = 1080
        self.image = np.zeros((self.img_width, self.img_height))
        self.depth = np.zeros((self.img_width, self.img_height))
        self.mask = np.zeros((self.img_width, self.img_height))
        self.graspable_mask = np.zeros((self.img_width, self.img_height))
        self.leaves = np.zeros((self.img_width, self.img_height, 4))
        self.grasp_point = np.zeros(3)
        self.approach_point = np.zeros(3)
        self.leaf_normal = np.zeros(3)

        print('Node Initialized...')


    def image_callback(self, mask, depth):
        """
        Runs once depth and mask data is recieved.
        Finds the leaf grasp location and publishes it.
        """

        rospy.set_param('/leaf_done', False )
        print("Mask and Depth data recieved!")
        depth = np.asarray(depth.imageData).astype('float32')
        mask = np.asarray(mask.imageData).astype('uint8')

        depth = np.reshape(depth, newshape=(self.img_height, self.img_width))
        mask = np.reshape(mask, newshape=(self.img_height, self.img_width))

        self.depth = depth
        self.mask = mask
        self.find_leaf() # Calculate leaf grasp location

        msg = grasp() # Setup publisher data
        msg.grasp.x = self.grasp_point[0]
        msg.grasp.y = self.grasp_point[1]
        msg.grasp.z = self.grasp_point[2]
        msg.norm.x = self.leaf_normal[0]
        msg.norm.y = self.leaf_normal[1]
        msg.norm.z = self.leaf_normal[2]
        msg.approach.x = self.approach_point[0]
        msg.approach.y = self.approach_point[1]
        msg.approach.z = self.approach_point[2]
        self.pub.publish(msg)

        rospy.set_param('/leaf_done', True)
    
        time.sleep(1)
        raft_status = rospy.get_param('raft_done') # Wait for next iteration of RAFT to finish prior to new cycle.
        while raft_status is False:
            print(f"\rWating for next raft to finish...", end="")
            raft_status = rospy.get_param('raft_done') 
        self.init_()

    def find_leaf(self):
        """
        Main function for calculating leaf grasping point.
        """
        tot_t = time.time()
        # Combine mask and depth data together to segment out leaves
        pcd_path = f"{HOME_DIR}/SDF_OUT/temp/temp.pcd"
        mask_path = self.mask
        image_path = self.image
        leafs, real_depth = apply_depth_mask(pcd_path, mask_path, image_path, plot=False)
        mask = clean_mask(leafs)
        leafs[:, :, 3] = mask
        ############################################################
        
        # Convolve each leaf with microneedle array-scaled kernels to get graspable area
        print("Computing Graspable Area")
        t = time.time()
        depth_image = leafs[:, :, 2].astype("float32")
        mask_image = leafs[:, :, 3].astype("uint8")
        kernels = get_kernels(depth_image, mask_image)
        graspable_mask = compute_graspable_areas(kernels, mask_image)
        print(f"Computation took {time.time()-t:.3f} s")
        ############################################################

        # Compute the normal vectors of every leaf
        binary_graspable_mask = graspable_mask >= 1
        leafs[:, :, 3] = leafs[:, :, 3] * binary_graspable_mask
        leafs_ = np.reshape(leafs[:, :, 0:3], (1555200, 3))
        index = np.argwhere(leafs_ == [0, 0, 0])
        inverse_index = np.nonzero(leafs_[:, 2])
        leafs_ = np.delete(leafs_, index, 0)
        processed_pcd = o3d.geometry.PointCloud()
        processed_pcd.points = o3d.utility.Vector3dVector(leafs_)
        cl, id = processed_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        compute_normals(processed_pcd)
        sqrt_dist = np.sum((processed_pcd.normals[999]) ** 2, axis=0)
        dist = np.sqrt(sqrt_dist)
        normal_orientation = abs(np.asarray(processed_pcd.normals)[:, 2])

        normal_corrected = np.asarray(processed_pcd.normals)
        for normal in normal_corrected:
            if normal[2] < 0:
                normal *= -1
        print(normal_corrected.shape)
        orientation_color = np.zeros((len(normal_orientation), 3))
        orientation_color[:, 0] = normal_orientation
        orientation_color[:, 1:] = 0
        processed_pcd.colors = o3d.utility.Vector3dVector(orientation_color)
        #############################################################


        # Estimate leaf flatness based on normal vectors
        leaf_flatness = np.zeros((1555200, 1))
        leaf_normals = np.zeros((1555200, 3))
        j = 0
        for i, _ in enumerate(inverse_index[0]):
            current_index = inverse_index[0][i]
            if j < normal_orientation.size:
                leaf_flatness[current_index, 0] = normal_orientation[j]
                leaf_normals[current_index, :] = normal_corrected[j, :]
            j += 1

        leaf_flatness = np.reshape(leaf_flatness, (1080, 1440, 1))
        leaf_normals = np.reshape(leaf_normals, (1080, 1440, 3))
        leafs = np.concatenate((leafs, leaf_flatness), axis=2)
        #############################################################


        # Combine graspable area with flat area to determine optimal leaf grapsing locations
        ALPHA = 0.4  # Adjustable parameter to change blend between grasping area and flat area
        smooth_section = leafs[:, :, 4]
        leaf_selection_a = ALPHA * smooth_section + (1 - ALPHA) * binary_graspable_mask

        leaf_selection_ab = np.where(
            leaf_selection_a, leaf_selection_a >= np.amax(leaf_selection_a) * 0.95, 0
        )
        leafs[:, :, 3] *= leaf_selection_ab
        viable_leaf_regions = clean_mask(leafs)
        ###############################################################

        # Calculate the mean depth of each leaf and identify tall leaves
        depth_list = mean_mask_depth(leafs)
        depth_list_norm = mean_mask_depth(leafs, normalized=True)
        tall_leaves = find_tall_leaves(depth_list, leafs)
        tall_presence = False
        if sum(sum(tall_leaves)) > 0:
            tall_presence = True
        ###############################################################


        # Find the SDF of the leaves to calculate global clutter minima and maxima
        leafs[:, :, 3] = viable_leaf_regions
        viable_leaf_regions = clean_mask(leafs)
        cleaned_masks = viable_leaf_regions >= 1
        cleaned_masks = np.where(cleaned_masks, cleaned_masks == 0, 1)
        SDF = skfmm.distance(cleaned_masks, dx=1)
        if tall_presence:
            cleaned_masks_tall = tall_leaves >= 1
            cleaned_masks_tall = np.where(cleaned_masks_tall, cleaned_masks_tall == 0, 1)
            SDF_X = skfmm.distance(cleaned_masks_tall, dx=1)
            min_tall = np.unravel_index(SDF_X.argmin(), SDF_X.shape)
            max_tall = np.unravel_index(SDF_X.argmax(), SDF_X.shape)

        min_global = np.unravel_index(SDF.argmin(), SDF.shape)
        max_global = np.unravel_index(SDF.argmax(), SDF.shape)
        #################################################################


        # Find the centroid of each leaf
        centroids, mask, areas = get_centroids(viable_leaf_regions.astype("uint8"))
        leafs[:, :, 3] = mask

        if tall_presence:
            tall_leaves = tall_leaves * viable_leaf_regions
            centroids_tall, mask_tall, areas_tall = get_centroids(
                tall_leaves.astype("uint8")
            )
        #################################################################


        # Find the distance of each centroid from the image's SDFminima and maxima
        data = compute_minmax_dist(centroids, min_global, max_global)
        if tall_presence:
            data_tall = compute_minmax_dist(centroids_tall, min_tall, max_tall)
        #################################################################


        # Use the distances to determine the optimal leaves to choose based on their pareto set
        mask = paretoset(data)
        paretoset_sols = data[mask]
        res = mask
        opt_leaves = np.where(res == True)[0]

        if tall_presence:
            mask_tall = paretoset(data_tall)
            paretoset_sols_tall = data_tall[mask_tall]
            res_tall = mask_tall
            opt_leaves_tall = np.where(res_tall == True)[0]
        #################################################################


        # Select the final grapsing point based on distance to scene SDF Maxima
        max_leaf = None
        min_distance = float('inf')


        for idx, sol in enumerate(paretoset_sols):
            if sol[1] < min_distance:
                min_distance = sol[1]
                max_leaf = idx

        opt_point = centroids[opt_leaves[max_leaf]]
        ################################################################

        # Convert pixel coords to real-world coords
        real_grasp_coord = leafs[opt_point[1], opt_point[0], 0:3]
        real_grasp_coord = np.round(real_grasp_coord, 4)
        grasp_normal = leaf_normals[opt_point[1], opt_point[0], :]

        SDF_max = np.unravel_index(SDF.argmax(), SDF.shape)

        x_dist = SDF_max[1] - opt_point[0]
        y_dist = SDF_max[0] - opt_point[1]

        tot_dist = np.sqrt((x_dist**2)+(y_dist**2))

        x_dist /= tot_dist * .005 #This fractional distance is currently arbitrarily set - in future alter to adapt based on leaves in trajectory
        y_dist /= tot_dist * .005
        target_vec = (int(x_dist+opt_point[0]), int(y_dist+opt_point[1]))

        real_depth = np.reshape(real_depth, (1080, 1440, 3))
        real_target_vec = real_depth[target_vec[1], target_vec[0], :]
        real_target_vec[2] = real_grasp_coord[2]
        real_target_vec = np.round(real_target_vec, 4)
        #################################################################

        print(f"Grasping point: {real_grasp_coord} \n Approach Vector: {real_target_vec} \n Normal Vector: {grasp_normal}")
        print(f"Total runtime: {time.time()-tot_t:.3f} s")
        
        if real_grasp_coord[2] == 0:
            print("Error in point selection process. Calcualted depth was 0.")
            return
        else:
            self.grasp_point = real_grasp_coord
            self.approach_point = real_target_vec
            self.leaf_normal = grasp_normal
            fig, ax = plt.subplot_mosaic([
            ['points']
                ], figsize=(15,10))
            ax['points'].imshow(viable_leaf_regions)
            x1, y1 = [opt_point[0], opt_point[1]]
            x2, y2 = [SDF_max[1], SDF_max[0]]
            ax["points"].axline((x1, y1), (x2, y2), marker='o')
            ax["points"].plot(target_vec[0], target_vec[1], marker='*', markersize=12)
            for i in opt_leaves:    
                ax['points'].plot(centroids[i][0], centroids[i][1], "r*")
            if tall_presence:
                for i in opt_leaves_tall:
                    ax["points"].plot(centroids_tall[i][0], centroids_tall[i][1], "b*")
            ax["points"].plot(opt_point[0], opt_point[1], "bo", markersize=8)
            ax["points"].plot(opt_point[0],opt_point[1], "r+", markersize=8)
            ax["points"].set_title("Selected Points (Blue = Tall Outliers)")
            fig.savefig(f"{HOME_DIR}/Grasp_OUT/viz.png")

def init():
    """
    Calls functions to create the Node.
    """
    print("in the init func")
    b = LeafGrasp()
    rospy.init_node('leaf_grasp', anonymous=False)
    rospy.spin()

if __name__ == '__main__':
    init()

