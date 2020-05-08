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

sigma_val = 1.0         # setting defaults
ransac_thresh = 0.01    # setting defaults
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
        # first_img = np.float32(np.gradient(data.normalize_full(first_img), 2, axis=0, edge_order=2))
        first_img = np.uint8(filter_image(first_img) * 255)
        second_img = frame_b.get_amplitude_image()
        # second_img = np.float32(np.gradient(data.normalize_full(second_img), 2, axis=0, edge_order=2))
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

                # point pairs for each image, indices match pairs
                # array to hold all (x,y) pixel loc of features
                img1_points = np.append(img1_points, [x1, y1])
                img2_points = np.append(img2_points, [x2, y2])

                # round values and cast to integer
                img1_points = np.int_(np.round(img1_points, decimals=0, out=None))
                img2_points = np.int_(np.round(img2_points, decimals=0, out=None))

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

            H, status = cv2.findHomography(img1_points, img2_points, cv2.RANSAC,
                                           ransacReprojThreshold=0.999)

            detected_feature_loc1 = np.reshape(detected_feature_loc1, (-1, 1, 3))
            detected_feature_loc2 = np.reshape(detected_feature_loc2, (-1, 1, 3))
            detected_feature_loc1 = detected_feature_loc1[status == 1]
            detected_feature_loc2 = detected_feature_loc2[status == 1]

            img1_points = img1_points[status == 1]
            img2_points = img2_points[status == 1]

            # show two original frames and matched features
            display = np.hstack((first_img, second_img))
            # display = second_img
            img1_points = img1_points.reshape((-1, 2))
            img2_points = img2_points.reshape((-1, 2))
            display = calc.render_keypoints(display, img1_points, img2_points,
                                            scale=3, offset=(first_img.shape[1], 0))

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
        param = RansacParams(samples=4, iterations=15, confidence=0.8, threshold=ransac_thresh)
        my_model = ThreeD()
        found_inliers = find_inliers(tupled_features, my_model, param)
        # print(found_inliers)
        if len(found_inliers) < 4:
            return None
        # print("\tnumber of inliers: ", len(found_inliers))
        # inlier_count = np.append(inlier_count, len(found_inliers))

    except:
        # traceback.print_exc()
        return None
        # return np.mat(np.eye(3)), np.mat(np.zeros(3).T)

    return my_model.rotate, my_model.translate


prior_model = gtsam.noiseModel_Diagonal.Sigmas(np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1]))
odom_noise = gtsam.noiseModel_Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
estimates = gtsam.Values()
graph = gtsam.NonlinearFactorGraph()

while True:
    img_type = 2
    data_file = input("\n\nSelect data:\n\tT for Translation\n\tP for Pitch\n\tY for Yaw\n\tA for Test1\n\tB for Test2\n\tV for video\noption: ")
    data_file = data_file.lower()
    which_data = data_file
    which_data += "_data"
    frame_1 = 0
    frame_1 = int(input("Select for set of data (0 for default): "))
    frame_2 = 0
    if data_file != 'v':
        frame_2 = int(input("Select a number (1, 2, or 3) for what data to be compared: "))
    # print("Previous threshold value: ", ransac_thresh)
    # ransac_thresh = float(input("Enter threshold value: "))
    # thresh = 0.01

    all_pitch = []
    all_yaw = []
    all_roll = []
    all_x = []
    all_y = []
    all_z = []
    inlier_count = []

    data_dirs = {'t_data': [('./data/Translation/Y1/', 'trans_1'),
                           ('./data/Translation/Y2/', 'trans_2'),
                           ('./data/Translation/Y3/', 'trans_3'),
                           ('./data/Translation/Y4/', 'trans_4')],
                 'p_data': [('./data/Pitch/d1_-31/', 'pitch_1'),
                           ('./data/Pitch/d2_-34/', 'pitch_2'),
                           ('./data/Pitch/d3_-37/', 'pitch_3'),
                           ('./data/Pitch/d4_-40/', 'pitch_4')],
                 'y_data': [('./data/Yaw/d1_44/', 'yaw_1'),
                         ('./data/Yaw/d2_41/', 'yaw_2'),
                         ('./data/Yaw/d3_38/', 'yaw_3'),
                         ('./data/Yaw/d4_35/', 'yaw_4')],
                 'a_data': [('./data/Test_Data1/pose1/', 'pose_1'),
                            ('./data/Test_Data1/pose2/', 'pose_2')],
                 'b_data': [('./data/Test_Data2/p1/', 'p_1'),
                            ('./data/Test_Data2/p2/', 'p_2'),
                            ('./data/Test_Data2/p3/', 'p_3')],
                 'v_data': [('./data/RV_Data2/', 'vid')]}

    if data_file == 't':
        frame_text = "Translation"
        ransac_thresh = 0.01
        sigma_val = 1.3
    elif data_file == 'p':
        frame_text = "Pitch"
        ransac_thresh = 0.001
        sigma_val = 1.0
    elif data_file == 'y':
        frame_text = "Yaw"
        ransac_thresh = 0.001
        sigma_val = 1.0
    elif data_file == 'a':
        frame_text = "Test 1"
        ransac_thresh = 0.01
        sigma_val = 1.3
    elif data_file == 'b':
        frame_text = "Test 2"
        ransac_thresh = 0.5
        sigma_val = 1.3
    elif data_file == 'v':
        frame_text = "Video Run"
        ransac_thresh = 0.0001
        sigma_val = 1.0

    data_set_1 = data.Dataset(data_dirs[which_data][frame_1])
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

    if data_file != 'v':
        data_set_2 = data.Dataset(data_dirs[which_data][frame_2])
        for frame_2 in data_set_2.data:
            frame_2.x = -1 * frame_2.x
            temp_y1 = frame_2.y
            frame_2.y = frame_2.z
            frame_2.z = temp_y1

            frame_2.amplitude[frame_2.amplitude > 250] = 0  # adjust maximum
            frame_2.amplitude[frame_2.amplitude == 0] = np.max(frame_2.amplitude)

            frame_2.amplitude = gaussian_filter(frame_2.amplitude, sigma=sigma_val)  # start gaussian filter
            frame_2.x = gaussian_filter(frame_2.x, sigma=sigma_val)
            frame_2.y = gaussian_filter(frame_2.y, sigma=sigma_val)
            frame_2.z = gaussian_filter(frame_2.z, sigma=sigma_val)

    frame_count = 0
    for i in range(len(data_set_1.data)):                                           # runs through all images in set
        try:
            if data_file == 'v':                # for grabbing two frame out of the same dataset for video and
                if i == 0:                      # collecting data on pose change of video
                    frame_1 = data_set_1.next_entry()
                else:
                    frame_1 = prev_frame

                frame_2 = data_set_1.next_entry()
                prev_frame = frame_2

                new_rotate, new_translation = pose_change_a_to_b(frame_1, frame_2)

                if i == 0:
                    prev_rot = np.mat(np.eye(3))
                    prev_trans = np.mat(np.zeros(3))
                    x_val.append(0)
                    y_val.append(0)
                    z_val.append(0)

                # calculate pose change incorporating rotation into translation for dead reckoning graph
                dt = np.transpose(new_rotate * new_translation) * prev_rot
                prev_rot = np.dot(prev_rot, new_rotate)
                prev_trans = prev_trans - dt

                x_val.append(prev_trans[0, 0])      # store translations into list for graphing
                y_val.append(prev_trans[0, 1])
                z_val.append(prev_trans[0, 2])

                estimates.insert(frame_count,
                                 gtsam.Pose3(gtsam.Rot3(prev_rot), gtsam.Point3(np.array(prev_trans).reshape(3))))
                if frame_count > 0:
                    factor = gtsam.BetweenFactorPose3(frame_count - 1, frame_count, gtsam.Pose3(gtsam.Rot3(new_rotate),
                                                      gtsam.Point3(np.array(new_translation).reshape(3))), odom_noise)
                else:
                    factor = gtsam.PriorFactorPose3(0, gtsam.Pose3(gtsam.Rot3(prev_rot),
                                                    gtsam.Point3(np.array(prev_trans).reshape(3))), prior_model)

                graph.add(factor)
                frame_count = frame_count + 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:         # for grabbing the Nth pair out of two datasets and collecting data on pose change
                frame_1 = data_set_1.next_entry()                                           # get next entries
                frame_2 = data_set_2.next_entry()

                new_rotate, new_translation = pose_change_a_to_b(frame_1, frame_2)

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
            # traceback.print_exc()
            pass

    # output after all images have been processed
    if data_file == 'v':
        plt.plot(x_val, y_val, label="Dead Reckoning x,y")
        plt.plot(x_val, z_val, label="Dead Reckoning x,z")
        plt.plot(z_val, y_val, label="Dead Reckoning z,y")
        plt.title('Dead Reckoning')
        plt.legend()
        plt.show()          # plot dead reckoning

        gtparams = gtsam.GaussNewtonParams()
        optimizer = gtsam.GaussNewtonOptimizer(graph, estimates, gtparams)
        result = optimizer.optimize()
        result_poses = gtsam.allPose3s(result)

        gtsam_x_val = []
        gtsam_y_val = []
        gtsam_z_val = []
        r_off = np.mat([[1., 0., 0.],
                        [0., 0.91363181, - 0.40654264],
                        [0., 0.40654264, 0.91363181]])

        from mpl_toolkits.mplot3d import Axes3D
        for k in range(result_poses.size()):
            plot.plot_pose3(1, result_poses.atPose3(k))
            p = result_poses.atPose3(k)
            t = p.translation()
            # r = p.roation().matrix()
            # print(t)
            t = np.array(np.mat([t.x(), t.y(), t.z()]) * r_off).reshape(3)
            gtsam_x_val.append(t[0])
            gtsam_y_val.append(t[1])
            gtsam_z_val.append(t[2])

        plt.show()          # plot 3D graph slam results
        plt.plot(gtsam_x_val, gtsam_y_val, label="Graph SLAM x,y")
        plt.plot(gtsam_x_val, gtsam_z_val, label="Graph SLAM x,z")
        plt.plot(gtsam_z_val, gtsam_y_val, label="Graph SLAM z,y")
        plt.legend()
        plt.show()          # plot 2D graph slam results

    else:
        print("(x, y, z) average (mm): (", np.round(np.mean(all_x) * 1000, 3), ", ", np.round(np.mean(all_y) * 1000, 3),
              ", ", np.round(np.mean(all_z) * 1000, 3), ")")
        print("(x, y, z) std dev (mm): (", np.round(np.std(all_x * 1000), 3), ", ", np.round(np.std(all_y * 1000), 3),
              ", ", np.round(np.std(all_z * 1000), 3), ")\n")

        print("(roll, pitch, yaw) average (degrees): (", np.round(np.rad2deg(np.mean(all_roll)), 3), ", ",
              np.round(np.rad2deg(np.mean(all_pitch)), 3), ", ", np.round(np.rad2deg(np.mean(all_yaw)), 3), ")")
        print("(roll, pitch, yaw) std dev (degrees): (", np.round(np.rad2deg(np.std(all_roll)), 3), ", ",
              np.round(np.rad2deg(np.std(all_pitch)), 3), ", ", np.round(np.rad2deg(np.std(all_yaw)), 3), ")")
