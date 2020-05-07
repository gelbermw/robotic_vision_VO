import cv2
import odataset as data
import functions as calc
import numpy as np
from pyransac.ransac import find_inliers
from pyransac.ransac import RansacParams
from pyransac.ThreeD import ThreeD
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import gtsam
from gtsam.utils import plot
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

NUM_FEATURES = 6
sigma_val = 1.0
FIRST = True
FIRST_IMAGE = True
ransac_thresh = 0.01
x_val = []
y_val = []
z_val = []
inlier_count = []

filter_max = None
filter_min = None


def filter_image(img):
    global filter_max, filter_min
    if filter_max is None:
        filter_max = np.max(img)
        filter_min = np.min(img)
    img = (img - filter_min) / (filter_max - filter_min)
    img[img > 1] = 1
    img[img < 0] = 0
    return img


def pose_change_a_to_b(frame_a, frame_b):

    try:
        first_img = frame_a.get_amplitude_image()  # adjust image sizes
        first_img = np.float32(np.gradient(data.normalize_full(first_img), 2, axis=0, edge_order=2))
        first_img = np.uint8(filter_image(first_img) * 255)
        second_img = frame_b.get_amplitude_image()
        second_img = np.float32(np.gradient(data.normalize_full(second_img), 2, axis=0, edge_order=2))
        second_img = np.uint8(filter_image(second_img) * 255)

        sift_img = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.0015, edgeThreshold=10, nOctaveLayers=3, sigma=1.2)
        mask = np.zeros_like(first_img)
        mask[frame_a.confidence >= 6] = 1
        kp1, desc1 = sift_img.detectAndCompute(first_img, mask=mask)  # sift test first image
        mask[frame_b.confidence >= 6] = 1
        kp2, desc2 = sift_img.detectAndCompute(second_img, mask=mask)  # sift second image

        img1_points = []
        img2_points = []
        detected_feature_loc1 = []
        detected_feature_loc2 = []

        if desc1 is None or desc2 is None or not kp1 or not kp2:  # remove frames that don't have anything
            pass
            # cv2.imshow(frame_text + " 1", first_img)
            # cv2.imshow(frame_text + " 2", second_img)
        else:
            compare_desc = cv2.BFMatcher()  # setup brute force matcher
            match = compare_desc.knnMatch(desc1, desc2, k=2)
            j = 0
            for points, points2 in match:  # {
                if points.distance > 0.55 * points2.distance:
                    continue
                # get the x and y pixel locations the features from the matcher
                # get points from the key points array that were matched
                (x1, y1) = kp1[points.queryIdx].pt
                (x2, y2) = kp2[points.trainIdx].pt

                # print(np.float32([kp1[points.queryIdx].pt]))
                # point pairs for each image, indices match pairs
                # array to hold all (x,y) pixel loc of features
                img1_points = np.append(img1_points, [x1, y1])
                img2_points = np.append(img2_points, [x2, y2])

                # round values and cast to integer
                img1_points = np.int_(np.round(img1_points, decimals=0, out=None))
                img2_points = np.int_(np.round(img2_points, decimals=0, out=None))

                # img1_points = img1_points[status == 1]
                # img2_points = img2_points[status == 1]

                # get the xvect, yvect, and z distance of each point of interest based on the pixel xy location
                # and round to 4 decimal places.  final full array of all matching points
                feature_loc1 = np.round(frame_a.get_position(img1_points[j * 2], img1_points[j * 2 + 1]), 4)
                feature_loc2 = np.round(frame_b.get_position(img2_points[j * 2], img2_points[j * 2 + 1]), 4)
                detected_feature_loc1 = np.append(detected_feature_loc1,
                                                  (feature_loc1[1], feature_loc1[2], feature_loc1[0]))
                detected_feature_loc2 = np.append(detected_feature_loc2,
                                                  (feature_loc2[1], feature_loc2[2], feature_loc2[0]))
                j += 1

            img1_points = np.reshape(img1_points, (-1, 1, 2))
            img2_points = np.reshape(img2_points, (-1, 1, 2))
            # print("img1 points: ", img1_points)

            H, status = cv2.findHomography(img1_points, img2_points, cv2.RANSAC,
                                           ransacReprojThreshold=0.999)

            detected_feature_loc1 = np.reshape(detected_feature_loc1, (-1, 1, 3))
            detected_feature_loc2 = np.reshape(detected_feature_loc2, (-1, 1, 3))
            # print("img1 points: ", detected_feature_loc1)
            detected_feature_loc1 = detected_feature_loc1[status == 1]
            detected_feature_loc2 = detected_feature_loc2[status == 1]

            img1_points = img1_points[status == 1]
            img2_points = img2_points[status == 1]

            # show two original frames and matched features
            # display = np.hstack((first_img, second_img))
            display = second_img
            img1_points = img1_points.reshape((-1, 2))
            img2_points = img2_points.reshape((-1, 2))
            display = calc.render_keypoints(display, img2_points, img1_points, scale=3)  # , offset=(first_img.shape[1], 0))

            cv2.imshow(frame_text, display)
            # cv2.waitKey(10)

        # after each image loop before getting next image
        # reshape vector into 3D array then take transpose
        detected_feature_loc1 = np.reshape(detected_feature_loc1, (-1, 3))  # format [[x1y1z1][x2y2z2]]
        detected_feature_loc2 = np.reshape(detected_feature_loc2, (-1, 3))  # format [[x1y1z1][x2y2z2]]

        detected_feature_loc1_tuple = tuple(detected_feature_loc1)
        detected_feature_loc2_tuple = tuple(detected_feature_loc2)

        tupled_features = [[tuple(x), tuple(y)] for x, y in zip(detected_feature_loc1, detected_feature_loc2)]

        # send to ransac calculation
        param = RansacParams(samples=8, iterations=15, confidence=0.9, threshold=ransac_thresh)
        my_model = ThreeD()
        found_inliers = find_inliers(tupled_features, my_model, param)
        # print(found_inliers)
        if len(found_inliers) < 4:
            return None
        print("\tnumber of inliers: ", len(found_inliers))
        # inlier_count = np.append(inlier_count, len(found_inliers))

    except:
        traceback.print_exc()
        return None
        # return np.mat(np.eye(3)), np.mat(np.zeros(3).T)

    return my_model.rotate, my_model.translate


prior_model = gtsam.noiseModel_Diagonal.Sigmas(np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1]))
odom_noise = gtsam.noiseModel_Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
estimates = gtsam.Values()
# estimates.insert(gtsam)
graph = gtsam.NonlinearFactorGraph()

while True:
    data_file = input("\n\nSelect data:\n\tV for Video\noption: ")
    which_data = data_file.lower()
    which_data += "_data"
    loc_1 = 0

    # print("Previous threshold value: ", ransac_thresh)
    # ransac_thresh = float(input("Enter threshold value: "))

    all_pitch = []
    all_yaw = []
    all_roll = []
    all_x = []
    all_y = []
    all_z = []

    data_dirs = {'v_data': [('./data/RV_Data2/', 'vid')]}

    if data_file == 'V' or data_file == 'v':
        frame_text = "Video Run"

    data_set_1 = data.Dataset(data_dirs[which_data][loc_1])
    for frame_1 in data_set_1.data:
        frame_1.x = -1 * frame_1.x
        temp_y1 = frame_1.y
        frame_1.y = frame_1.z
        frame_1.z = temp_y1

        frame_1.amplitude[frame_1.amplitude > 250] = 0  # adjust maximum
        frame_1.amplitude[frame_1.amplitude == 0] = np.max(frame_1.amplitude)

        frame_1.amplitude = gaussian_filter(frame_1.amplitude, sigma=sigma_val)  # start gaussian filter
        frame_1.x = gaussian_filter(frame_1.x, sigma=sigma_val)
        frame_1.y = gaussian_filter(frame_1.y, sigma=sigma_val)
        frame_1.z = gaussian_filter(frame_1.z, sigma=sigma_val)

    # print("Dataset Length: ", len(data_set_1.data))
    frame_count = 0
    for i in range(len(data_set_1.data)):                          # runs through all images in set
        print(i)
        try:
            if FIRST:
                frame_1 = data_set_1.next_entry()
                FIRST = False
            else:
                frame_1 = prev_frame

            frame_2 = data_set_1.next_entry()
            prev_frame = frame_2

            new_rotate, new_translation = pose_change_a_to_b(frame_1, frame_2)

            # print("rotation: ", new_rotate)
            # print(i, "translation: ", new_translation.T)
            # graph the translation for dead reckoning
            if FIRST_IMAGE:
                prev_rot = np.mat(np.eye(3))
                prev_trans = np.mat(np.zeros(3))
                x_val.append(0)
                y_val.append(0)
                z_val.append(0)
                FIRST_IMAGE = False

            dt = np.transpose(new_rotate * new_translation) * prev_rot
            prev_rot = np.dot(prev_rot, new_rotate)
            prev_trans = prev_trans - dt
            # print(i, "translation: ", new_translation.T, " | ", prev_trans)
            x_val.append(prev_trans[0, 0])
            y_val.append(prev_trans[0, 1])
            z_val.append(prev_trans[0, 2])

            estimates.insert(frame_count, gtsam.Pose3(gtsam.Rot3(prev_rot), gtsam.Point3(np.array(prev_trans).reshape(3))))
            if frame_count > 0:
                factor = gtsam.BetweenFactorPose3(frame_count-1, frame_count, gtsam.Pose3(gtsam.Rot3(new_rotate), gtsam.Point3(np.array(new_translation).reshape(3))), odom_noise)
                graph.add(factor)

            frame_count = frame_count + 1

            pry = calc.rotationMatrixToEulerAngles(new_rotate)
            all_pitch = np.append(all_pitch, pry[0])
            all_roll = np.append(all_roll, pry[1])
            all_yaw = np.append(all_yaw, pry[2])

            all_x = np.append(all_x, new_translation[0])
            all_y = np.append(all_y, new_translation[1])
            all_z = np.append(all_z, new_translation[2])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except:
            import traceback
            traceback.print_exc()
            pass

    # after all images have been processed
    plt.plot(x_val, y_val, label="Dead Reckoning x,y")
    plt.plot(x_val, z_val, label="Dead Reckoning x,z")
    plt.plot(y_val, z_val, label="Dead Reckoning y,z")
    plt.title('Dead Reckoning')
    plt.legend()
    plt.show()

    gtparams = gtsam.GaussNewtonParams()
    optimizer = gtsam.GaussNewtonOptimizer(graph, estimates, gtparams)
    result = optimizer.optimize()
    result_poses = gtsam.allPose3s(result)

    from mpl_toolkits.mplot3d import Axes3D
    for k in range(result_poses.size):
        plot.plot_pose3(1, result_poses.atPose3(k))

    plt.show()


    # print("(x, y, z) average (mm): (", np.round(np.mean(all_x) * 1000, 3), ", ", np.round(np.mean(all_y) * 1000, 3),
    #       ", ", np.round(np.mean(all_z) * 1000, 3), ")")
    # print("(x, y, z) std dev (mm): (", np.round(np.std(all_x * 1000), 3), ", ", np.round(np.std(all_y * 1000), 3),
    #       ", ", np.round(np.std(all_z * 1000), 3), ")\n")
    #
    # print("(roll, pitch, yaw) average (degrees): (", np.round(np.rad2deg(np.mean(all_roll)), 3), ", ",
    #       np.round(np.rad2deg(np.mean(all_pitch)), 3), ", ", np.round(np.rad2deg(np.mean(all_yaw)), 3), ")")
    # print("(roll, pitch, yaw) std dev (degrees): (", np.round(np.rad2deg(np.std(all_roll)), 3), ", ",
    #       np.round(np.rad2deg(np.std(all_pitch)), 3), ", ", np.round(np.rad2deg(np.std(all_yaw)), 3), ")")
    #
    # print("\nAverage num inliers/frame: ", np.mean(inlier_count))

