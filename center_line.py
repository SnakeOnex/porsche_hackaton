import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from heapq import nlargest


def average_y_for_unique_x(points):
    x_values = points[:, 0]
    y_values = points[:, 1]
    unique_x = np.unique(x_values)
    averaged_points = np.array(
        [[x, np.mean(y_values[x_values == x])] for x in unique_x])
    return averaged_points


def mean_x_coord(arr): return np.mean(arr[:, 1])


class CenterLinePredictor:
    def __init__(self,
                 # downsample input image to this size (width, height)
                 imsize=(640, 360),
                 polydeg=2,
                 # percentage of image of 4 corners of roi as a fraction of image size (height, width)
                 # roi_points=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                 roi_points=[[0.41, 0.42], [0.58, 0.42],
                             [0.58, 0.78], [0.41, 0.78]],
                 edge_mask_lower_thresh=[0, 0, 230],
                 edge_mask_upper_thresh=[180, 255, 255],
                 edge_detect_lower_thresh=200,
                 edge_detect_upper_thresh=400,
                 ):
        self.imsize = imsize
        self.width, self.height = imsize
        self.polydeg = polydeg
        self.roi_points = np.array(roi_points)
        self.edge_mask_lower_thresh = edge_mask_lower_thresh
        self.edge_mask_upper_thresh = edge_mask_upper_thresh
        self.edge_detect_lower_thresh = edge_detect_lower_thresh
        self.edge_detect_upper_thresh = edge_detect_upper_thresh
        self.roi_points *= np.array(imsize)
        self.roi_points = self.roi_points.astype(np.int32)

    def detect_edges(self, image, return_mask=True):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # show_image("hsv", hsv)
        lower_blue = np.array(self.edge_mask_lower_thresh)
        upper_blue = np.array(self.edge_mask_upper_thresh)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # detect edges
        edges = cv2.Canny(mask, self.edge_detect_lower_thresh,
                          self.edge_detect_upper_thresh)
        if return_mask:
            return edges, mask
        else:
            return edges

    def crop_to_roi(self, image, invert=False):
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [self.roi_points], 255)
        if invert:
            image[mask > 0] = 0
            return image
        else:
            return cv2.bitwise_and(image, mask)

    def find_center_line_points(self, left_poly_points, right_poly_points, n_points=50):
        x_left = left_poly_points[:, 0]
        x_right = right_poly_points[:, 0]
        largest_min_x = np.max([np.min(x_left), np.min(x_right)])
        smallest_max_x = np.min([np.max(x_left), np.max(x_right)])
        x_center = np.linspace(largest_min_x, smallest_max_x, n_points)
        return x_center

    def select_left_right_points(self, points):
        points_left = points[points[:, 1] < self.width//2]
        points_right = points[points[:, 1] > self.width//2]
        return points_left, points_right

    def select_left_right_points_dbscan(self, points, third_cluster_merge_threshold=0.15):
        # Use n_jobs=-1 to use all available cores
        db = DBSCAN(eps=5, min_samples=10, n_jobs=5).fit(points)
        labels = db.labels_

        # Extract the unique clusters and their counts, excluding noise
        unique_labels, counts = np.unique(
            labels[labels != -1], return_counts=True)

        # Find the top three largest clusters without sorting all
        largest_indices = nlargest(
            3, range(len(counts)), key=counts.__getitem__)

        # Extract the points corresponding to the three largest clusters
        clusters = [(count, points[labels == label]) for count, label in zip(
            counts[largest_indices], unique_labels[largest_indices])]

        # If the third largest cluster is significant, consider merging
        if len(clusters) > 2 and clusters[2][0] >= third_cluster_merge_threshold * min(clusters[0][0], clusters[1][0]):
            mean_x_coords = [mean_x_coord(cluster[1]) for cluster in clusters]
            dist_third_to_first = abs(mean_x_coords[0] - mean_x_coords[2])
            dist_third_to_second = abs(mean_x_coords[1] - mean_x_coords[2])

            if dist_third_to_first < dist_third_to_second:
                clusters[0] = (clusters[0][0] + clusters[2][0],
                               np.vstack((clusters[0][1], clusters[2][1])))
            else:
                clusters[1] = (clusters[1][0] + clusters[2][0],
                               np.vstack((clusters[1][1], clusters[2][1])))

        # Determine left and right clusters based on the mean X coordinates
        left_cluster, right_cluster = sorted(
            (clusters[0][1], clusters[1][1]), key=mean_x_coord)

        return left_cluster, right_cluster

    def predict_to_poly(self, image, visualize=False, invert_roi=True, return_points=False):
        resized_image = cv2.resize(
            image, self.imsize, interpolation=cv2.INTER_AREA)
        edges, mask = self.detect_edges(resized_image, return_mask=True)
        roi_mask = self.crop_to_roi(mask, invert=invert_roi)
        points = np.argwhere(roi_mask > 0)
        points_left, points_right = self.select_left_right_points_dbscan(
            points)
        points_left_for_poly = average_y_for_unique_x(points_left)
        points_right_for_poly = average_y_for_unique_x(points_right)
        left_poly = np.polyfit(
            points_left_for_poly[:, 0], points_left_for_poly[:, 1], self.polydeg)
        right_poly = np.polyfit(
            points_right_for_poly[:, 0], points_right_for_poly[:, 1], self.polydeg)
        center_poly = (left_poly + right_poly) / 2
        if visualize:
            self.visualize(resized_image, edges, mask, roi_mask, points_left_for_poly, points_right_for_poly,
                           left_poly, right_poly, center_poly)
        if return_points:
            return center_poly, points_left_for_poly, points_right_for_poly
        else:
            return center_poly

    def predict_to_points(self, image, visualize=False, invert_roi=True, n_points=50):
        """Return the Y predictions of polynomial fitted to the center line and the valid X points"""
        poly, left_pts, right_pts = self.predict_to_poly(
            image, return_points=True, visualize=visualize, invert_roi=invert_roi)
        x_center_points = self.find_center_line_points(
            left_pts, right_pts, n_points=n_points)
        poly_pred_center = np.polyval(poly, x_center_points)
        return poly_pred_center, x_center_points

    def predict_to_image(self, image, visualize=False, invert_roi=True):
        """Return the Y predictions of polynomial fitted to the center line and the valid X points"""
        poly, left_pts, right_pts = self.predict_to_poly(
            image, return_points=True, visualize=visualize, invert_roi=invert_roi)
        x_center_points = self.find_center_line_points(
            left_pts, right_pts, n_points=self.height).astype(np.int32)
        poly_pred_center = np.polyval(poly, x_center_points).astype(np.int32)
        mask = np.zeros(shape=self.imsize[::-1])
        for x, y in zip(x_center_points, poly_pred_center):
            cv2.circle(mask, (y, x), 2, (255, 255, 255), 5)
        return mask, (poly_pred_center, x_center_points)

    def visualize(self, image, edges, mask, roi_mask, points_left_poly, points_right_poly, left_poly, right_poly, center_poly):
        fig, ax = plt.subplots(2, 4, figsize=(25, 5), dpi=300)
        # adjust figures
        for i in range(2):
            for j in range(4):
                if i == 0:
                    ax[i][j].axis('off')
                ax[i][j].set_aspect(self.height/self.width)
                ax[i][j].set_xlim(0, self.width)
                ax[i][j].set_ylim(self.height, 0)

        #  plot images in the top row
        ax[0][0].imshow(image)
        ax[0][0].set_title('image')
        ax[0][1].imshow(edges)
        ax[0][1].set_title('edges')
        ax[0][2].imshow(mask)
        ax[0][2].set_title('mask')
        ax[0][3].imshow(roi_mask)
        ax[0][3].set_title('roi_mask')

        #  plot the points and polynomial predictions
        ax[1][0].plot(points_left_poly[:, 1], points_left_poly[:, 0],
                      'o', color="blue", label="left points for interpolation")
        ax[1][0].plot(points_right_poly[:, 1], points_right_poly[:, 0],
                      'o', color="red", label="right points for interpolation")
        ax[1][0].legend()
        x_left = points_left_poly[:, 0]
        x_right = points_right_poly[:, 0]
        poly_pred_left = np.polyval(left_poly, x_left)
        poly_pred_right = np.polyval(right_poly, x_right)
        ax[1][1].plot(poly_pred_left, x_left, color="blue",
                      label="left polynomial")
        ax[1][1].plot(poly_pred_right, x_right,
                      color="red", label="right polynomial")
        x_center = self.find_center_line_points(
            points_left_poly, points_right_poly)
        poly_pred_center = np.polyval(center_poly, x_center)
        ax[1][1].plot(poly_pred_center, x_center,
                      color="green", label="center polynomial")
        ax[1][1].legend()

        # show the predicted centerline in the mask
        ax[1][2].imshow(mask)
        ax[1][2].plot(poly_pred_center, x_center,
                      color="green", label="center polynomial")
        ax[1][2].set_title('mask with centerline')

        fig.tight_layout()
        plt.show()

    # def predict_to_poly(self, image, visualize=False, invert_roi=True, return_points=False):
    #     resized_image = cv2.resize(
    #         image, self.imsize, interpolation=cv2.INTER_AREA)
    #     edges, mask = self.detect_edges(resized_image, return_mask=True)
    #     roi_mask = self.crop_to_roi(mask, invert=invert_roi)
    #     points = np.argwhere(roi_mask > 0)
    #     points_left = points[points[:, 1] < roi_mask.shape[1]//2]
    #     points_right = points[points[:, 1] > roi_mask.shape[1]//2]
    #     points_left_for_poly = average_y_for_unique_x(points_left)
    #     points_right_for_poly = average_y_for_unique_x(points_right)
    #     left_poly = np.polyfit(
    #         points_left_for_poly[:, 0], points_left_for_poly[:, 1], self.polydeg)
    #     right_poly = np.polyfit(
    #         points_right_for_poly[:, 0], points_right_for_poly[:, 1], self.polydeg)
    #     center_poly = (left_poly + right_poly) / 2
    #     if visualize:
    #         self.visualize(resized_image, edges, mask, roi_mask, points_left_for_poly, points_right_for_poly,
    #                        left_poly, right_poly, center_poly)
    #     if return_points:
    #         return center_poly, points_left_for_poly, points_right_for_poly
    #     else:
    #         return center_poly

    # def predict_to_points(self, image, visualize=False, invert_roi=True, n_points=50):
    #     """Return the Y predictions of polynomial fitted to the center line and the valid X points"""
    #     poly, left_pts, right_pts = self.predict_to_poly(
    #         image, return_points=True, visualize=visualize, invert_roi=invert_roi)
    #     x_center_points = self.find_center_line_points(
    #         left_pts, right_pts, n_points=n_points)
    #     poly_pred_center = np.polyval(poly, x_center_points)
    #     return (poly_pred_center, x_center_points), poly

    # def predict_to_image(self, image, visualize=False, invert_roi=True):
    #     """Return the Y predictions of polynomial fitted to the center line and the valid X points"""
    #     poly, left_pts, right_pts = self.predict_to_poly(
    #         image, return_points=True, visualize=visualize, invert_roi=invert_roi)
    #     x_center_points = self.find_center_line_points(
    #         left_pts, right_pts, n_points=self.height).astype(np.int32)
    #     poly_pred_center = np.polyval(poly, x_center_points).astype(np.int32)
    #     mask = np.zeros(shape=self.imsize[::-1])
    #     for x, y in zip(x_center_points, poly_pred_center):
    #         cv2.circle(mask, (y, x), 2, (255, 255, 255), 5)
    #     return mask
