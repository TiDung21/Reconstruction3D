import cv2
import numpy as np
import os
import time
from scipy.optimize import least_squares
from tomlkit import boolean
from tqdm import tqdm
import matplotlib.pyplot as plt


class Image_loader():
    def __init__(self, img_dir: str, downscale_factor: float):

        with open(img_dir + '\\CameraMatrix.txt') as f:
            self.CameraMatrix = np.array(
                list((map(lambda x: list(map(lambda x: float(x), x.strip().split(' '))), f.read().split('\n')))))
            self.image_list = []
        # Loading the set of images
        for image in sorted(os.listdir(img_dir)):
            if image[-4:].lower() == '.jpg' or image[-5:].lower() == '.png':
                self.image_list.append(img_dir + '\\' + image)

        self.path = os.getcwd()
        self.factor = downscale_factor
        self.downscale()

    def downscale(self) -> None:

        self.CameraMatrix[0, 0] /= self.factor
        self.CameraMatrix[1, 1] /= self.factor
        self.CameraMatrix[0, 2] /= self.factor
        self.CameraMatrix[1, 2] /= self.factor

    def downscale_image(self, image):

        image = cv2.pyrDown(image)
        return image


class Sfm():
    def __init__(self, img_dir: str, downscale_factor: float = 2.0) -> None:
        self.img_obj = Image_loader(img_dir, downscale_factor)

    def find_features(self, image_0, image_1) -> tuple:

        '''
        Phát hiện đặc trưng sử dụng thuật toán SIFT ,Brute-Force Matcher và KNN
        Trả về các đặc trưng của ảnh 1 và ảnh 2
        '''

        sift = cv2.xfeatures2d.SIFT_create()
        key_points_0, desc_0 = sift.detectAndCompute(cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY), None)
        key_points_1, desc_1 = sift.detectAndCompute(cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY), None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_0, desc_1, k=2)
        good_feature = []
        feature = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_feature.append(m)
                feature.append([m])

        return np.float32([key_points_0[m.queryIdx].pt for m in good_feature]), np.float32([key_points_1[m.trainIdx].pt for m in good_feature]), key_points_0, key_points_1, feature

    def common_points(self, image_points_1, image_points_2, image_points_3) -> tuple:

        '''
        Tìm các điểm đặc trưng tương đồng giữa các ảnh liên tiếp
        Trả về điểm đặc trưng chung giữa ảnh 1-2, ảnh 2-3 và mask của điểm đặc trưng chung giữa ảnh 1-2, ảnh 2-3
        '''

        cm_points_1 = []
        cm_points_2 = []
        for i in range(image_points_1.shape[0]):
            a = np.where(image_points_2 == image_points_1[i, :])
            if a[0].size != 0:
                cm_points_1.append(i)
                cm_points_2.append(a[0][0])

        mask_array_1 = np.ma.array(image_points_2, mask=False)
        mask_array_1.mask[cm_points_2] = True
        mask_array_1 = mask_array_1.compressed()
        mask_array_1 = mask_array_1.reshape(int(mask_array_1.shape[0] / 2), 2)

        mask_array_2 = np.ma.array(image_points_3, mask=False)
        mask_array_2.mask[cm_points_2] = True
        mask_array_2 = mask_array_2.compressed()
        mask_array_2 = mask_array_2.reshape(int(mask_array_2.shape[0] / 2), 2)
        print(" Shape New Array", mask_array_1.shape, mask_array_2.shape)
        return np.array(cm_points_1), np.array(cm_points_2), mask_array_1, mask_array_2

    def PnP(self, obj_point, image_point, CameraMatrix, dist_coeff, rot_vector, initial) -> tuple:

        '''
        Tìm tư thế của đối tượng từ các điểm tương ứng 3D-2D sử dụng thuật toán RANSAC
        Trả về ma trận quay, ma trận tịnh tiến, điểm ảnh, điểm đối tượng và vectơ quay
        '''

        if initial == 1:
            obj_point = obj_point[:, 0, :]
            image_point = image_point.T
            rot_vector = rot_vector.T
        _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, CameraMatrix, dist_coeff, iterationsCount=100, flags=cv2.SOLVEPNP_ITERATIVE)
        # Converts a rotation matrix to a rotation vector or vice versa
        rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)

        if inlier is not None:
            image_point = image_point[inlier[:, 0]]
            obj_point = obj_point[inlier[:, 0]]
            rot_vector = rot_vector[inlier[:, 0]]
        return rot_matrix, tran_vector, image_point, obj_point, rot_vector

    def triangulation(self, point_2d_1, point_2d_2, projection_matrix_1, projection_matrix_2) -> tuple:

        '''
        Tam giác hóa điểm 3D từ vectơ 2D và ma trận chiếu
        Trả về ma trận chiếu của 2 camera và đám mây điểm
        '''

        pt_cloud = cv2.triangulatePoints(point_2d_1, point_2d_2, projection_matrix_1.T, projection_matrix_2.T, )
        return projection_matrix_1.T, projection_matrix_2.T, (pt_cloud / pt_cloud[3])

    def reprojection_error(self, obj_points, image_points, transform_matrix, CameraMatrix, homogenity) -> tuple:
        '''
        Tính toán lỗi tái chiếu, tức là khoảng cách giữa các điểm đã chiếu và các điểm thực tế, sau đó trả về tổng lỗi và điểm đối tượng.
        '''
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        if homogenity == 1:
            obj_points = cv2.convertPointsFromHomogeneous(obj_points.T)
        image_points_calc, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, CameraMatrix, None)
        image_points_calc = np.float32(image_points_calc[:, 0, :])
        total_error = cv2.norm(image_points_calc, np.float32(image_points.T) if homogenity == 1 else np.float32(image_points), cv2.NORM_L2)
        return total_error / len(image_points_calc), obj_points

    def optimal_reprojection_error(self, obj_points) -> np.array:
        '''
        Tính toán sai số tái chiếu trong quá trình điều chỉnh bó và trả về các giá trị sai số
        '''
        transform_matrix = obj_points[0:12].reshape((3, 4))
        CameraMatrix = obj_points[12:21].reshape((3, 3))
        rest = int(len(obj_points[21:]) * 0.4)
        p = obj_points[21:21 + rest].reshape((2, int(rest / 2))).T
        obj_points = obj_points[21 + rest:].reshape((int(len(obj_points[21 + rest:]) / 3), 3))
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        image_points, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, CameraMatrix, None)
        image_points = image_points[:, 0, :]
        error = [(p[idx] - image_points[idx]) ** 2 for idx in range(len(p))]
        return np.array(error).ravel() / len(p)

    def bundle_adjustment(self, _3d_point, opt, transform_matrix_new, CameraMatrix, r_error) -> tuple:
        '''
        Điều chỉnh bó cho các điểm ảnh và điểm đối tượng và trả về các điểm đối tượng, điểm ảnh, ma trận biến đổi
        '''
        opt_variables = np.hstack((transform_matrix_new.ravel(), CameraMatrix.ravel()))
        opt_variables = np.hstack((opt_variables, opt.ravel()))
        opt_variables = np.hstack((opt_variables, _3d_point.ravel()))

        values_corrected = least_squares(self.optimal_reprojection_error, opt_variables, gtol=r_error).x
        CameraMatrix = values_corrected[12:21].reshape((3, 3))
        rest = int(len(values_corrected[21:]) * 0.4)
        return values_corrected[21 + rest:].reshape((int(len(values_corrected[21 + rest:]) / 3), 3)), values_corrected[ 21:21 + rest].reshape((2, int(rest / 2))).T, values_corrected[0:12].reshape((3, 4))

    def to_ply(self, path, point_cloud, colors) -> None:
        '''
        Tạo ra file .ply chứa các đám mây điểm
        '''
        out_points = point_cloud.reshape(-1, 3) * 200
        out_colors = colors.reshape(-1, 3)
        print(out_colors.shape, out_points.shape)
        verts = np.hstack([out_points, out_colors])

        mean = np.mean(verts[:, :3], axis=0)
        scaled_verts = verts[:, :3] - mean
        dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)
        indx = np.where(dist < np.mean(dist) + 300)
        verts = verts[indx]
        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar blue
            property uchar green
            property uchar red
            end_header
            '''
        with open(path + '\\res\\' + self.img_obj.image_list[0].split('\\')[-2] + '.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')



    def __call__(self, enable_bundle_adjustment: boolean = True):
        # hình ảnh
        # cv2.namedWindow('Image 1', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('Image 2', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('Match Image', cv2.WINDOW_NORMAL)
        pose_array = self.img_obj.CameraMatrix.ravel()
        transform_matrix_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        transform_matrix_1 = np.empty((3, 4))

        pose_0 = np.matmul(self.img_obj.CameraMatrix, transform_matrix_0)
        pose_1 = np.empty((3, 4))
        total_points = np.zeros((1, 3))
        total_colors = np.zeros((1, 3))
        threshold = 0.5

        # khởi tạo
        image_0 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[0]))
        image_1 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[1]))
        # trích xuất đặc trưng
        feature_0, feature_1, key_points_0, key_points_1, feature = self.find_features(image_0, image_1)

        # Essential matrix: lọc các ngoại lệ và tạo ra ma trận cơ bản
        essential_matrix, em_mask = cv2.findEssentialMat(feature_0, feature_1, self.img_obj.CameraMatrix, method=cv2.RANSAC, prob=0.999, threshold=threshold, mask=None)
        feature_0 = feature_0[em_mask.ravel() == 1]
        feature_1 = feature_1[em_mask.ravel() == 1]

        _, rot_matrix, tran_matrix, em_mask = cv2.recoverPose(essential_matrix, feature_0, feature_1, self.img_obj.CameraMatrix)
        feature_0 = feature_0[em_mask.ravel() > 0]
        feature_1 = feature_1[em_mask.ravel() > 0]
        transform_matrix_1[:3, :3] = np.matmul(rot_matrix, transform_matrix_0[:3, :3])
        transform_matrix_1[:3, 3] = transform_matrix_0[:3, 3] + np.matmul(transform_matrix_0[:3, :3], tran_matrix.ravel())

        pose_1 = np.matmul(self.img_obj.CameraMatrix, transform_matrix_1)

        feature_0, feature_1, points_3d = self.triangulation(pose_0, pose_1, feature_0, feature_1)
        error, points_3d = self.reprojection_error(points_3d, feature_1, transform_matrix_1, self.img_obj.CameraMatrix, homogenity=1)
        # ideally error < 1
        print("REPROJECTION ERROR: ", error)
        _, _, feature_1, points_3d, _ = self.PnP(points_3d, feature_1, self.img_obj.CameraMatrix,
                                                 np.zeros((5, 1), dtype=np.float32), feature_0, initial=1)

        total_images = len(self.img_obj.image_list)
        pose_array = np.hstack((np.hstack((pose_array, pose_0.ravel())), pose_1.ravel()))

        for i in tqdm(range(total_images - 2)):
            if (i % 1) == 0:
                image_2 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[i + 2]))
                features_cur, features_2, key_points_0, key_points_1, feature = self.find_features(image_1, image_2)
                # keypointimage_1 = cv2.drawKeypoints(image_1, key_points_0, None, color=(0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                # keypointimage_2 = cv2.drawKeypoints(image_2, key_points_1, None, color=(0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                # cv2.imshow('Image 1', keypointimage_1)
                # cv2.imshow('Image 2', keypointimage_2)
                # cv2.waitKey()
                # matchesimage = cv2.drawMatchesKnn(image_1, key_points_0, image_2, key_points_1, feature, None,flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                # cv2.imshow('Match Image', matchesimage)

                if i != 0:
                    feature_0, feature_1, points_3d = self.triangulation(pose_0, pose_1, feature_0, feature_1)
                    feature_1 = feature_1.T
                    points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
                    points_3d = points_3d[:, 0, :]

                cm_points_0, cm_points_1, cm_mask_0, cm_mask_1 = self.common_points(feature_1, features_cur, features_2)
                cm_points_2 = features_2[cm_points_1]
                cm_points_cur = features_cur[cm_points_1]

                rot_matrix, tran_matrix, cm_points_2, points_3d, cm_points_cur = self.PnP(points_3d[cm_points_0], cm_points_2, self.img_obj.CameraMatrix, np.zeros((5, 1), dtype=np.float32), cm_points_cur, initial=0)
                transform_matrix_1 = np.hstack((rot_matrix, tran_matrix))
                pose_2 = np.matmul(self.img_obj.CameraMatrix, transform_matrix_1)

                error, points_3d = self.reprojection_error(points_3d, cm_points_2, transform_matrix_1, self.img_obj.CameraMatrix, homogenity=0)

                cm_mask_0, cm_mask_1, points_3d = self.triangulation(pose_1, pose_2, cm_mask_0, cm_mask_1)
                error, points_3d = self.reprojection_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.CameraMatrix, homogenity=1)
                print("Reprojection Error: ", error)
                pose_array = np.hstack((pose_array, pose_2.ravel()))

                if enable_bundle_adjustment:
                    points_3d, cm_mask_1, transform_matrix_1 = self.bundle_adjustment(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.CameraMatrix, threshold)
                    pose_2 = np.matmul(self.img_obj.CameraMatrix, transform_matrix_1)
                    error, points_3d = self.reprojection_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.CameraMatrix, homogenity=0)
                    print("Bundle Adjusted error: ", error)
                    total_points = np.vstack((total_points, points_3d))
                    points_left = np.array(cm_mask_1, dtype=np.int32)
                    color_vector = np.array([image_2[l[1], l[0]] for l in points_left])
                    total_colors = np.vstack((total_colors, color_vector))
                else:
                    total_points = np.vstack((total_points, points_3d[:, 0, :]))
                    points_left = np.array(cm_mask_1, dtype=np.int32)
                    color_vector = np.array([image_2[l[1], l[0]] for l in points_left.T])
                    total_colors = np.vstack((total_colors, color_vector))

                transform_matrix_0 = np.copy(transform_matrix_1)
                pose_0 = np.copy(pose_1)
                plt.scatter(i, error)
                plt.pause(0.05)

                image_0 = np.copy(image_1)
                image_1 = np.copy(image_2)
                feature_0 = np.copy(features_cur)
                feature_1 = np.copy(features_2)
                pose_1 = np.copy(pose_2)
                # cv2.imshow(self.img_obj.image_list[0].split('\\')[-2], image_2)

                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
        cv2.destroyAllWindows()

        print("Printing to .ply file")
        print(total_points.shape, total_colors.shape)
        self.to_ply(self.img_obj.path, total_points, total_colors)
        print("Completed Exiting ...")


start_time = time.time()

if __name__ == '__main__':
    sfm = Sfm("Datasets\\QuaBong")
    sfm()

end_time = time.time()

total_time = end_time - start_time

print("Total time: ", total_time)
