import cv2
import odataset as data
import functions as calc
import numpy as np
import transform as tr
from pyransac.ransac import find_inliers
from pyransac.ransac import RansacParams
from pyransac.ThreeD import ThreeD
from scipy.ndimage.filters import gaussian_filter
from tabulate import tabulate
from scipy.spatial.transform import Rotation
import string
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# NUM_FEATURES = 4       # works for pitch and yaw translation 100 and 300
NUM_FEATURES = 6
# sigma_val = 1.0     # works for pitch and yaw
sigma_val = 1.3         # 0.6 works for translation 100 and 300
FIRST = True
ransac_thresh = 0.0
while True:
    img_type = 2
    data_file = input("\n\nSelect data:\n\tT for Translation\n\tP for Pitch\n\tY for Yaw\n\tA for Test1\n\tB for Test2\noption: ")
    which_data = data_file.lower()
    which_data += "_data"
    frame_1 = 0
    frame_1 = int(input("Select for set of data (0 for default): "))
    frame_2 = 0
    if img_type == 2:
        frame_2 = int(input("Select a number (2, 3, or 4) for what data to be compared: "))
        frame_2 -= 1
        frame_num = frame_2
    avg = 0
    print("Previous threshold value: ", ransac_thresh)
    ransac_thresh = float(input("Enter threshold value: "))
    # thresh = float(input("Enter threshold value for RANSAC: "))
    # thresh = 0.001 # works for pitch and yaw
    thresh = 0.01
    # initialize list of matching point pairs
    img1_points = []
    img2_points = []
    alpha_arr = []
    beta_arr = []
    gamma_arr = []
    detected_feature_loc1 = []
    detected_feature_loc2 = []
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

    if data_file == 'T' or data_file == 't':
        frame_text = "Translation Image"
    elif data_file == 'P' or data_file == 'p':
        frame_text = "Pitch Image"
    elif data_file == 'Y' or data_file == 'y':
        frame_text = "Yaw Image"
    elif data_file == 'A' or data_file == 'a':
        frame_text = "Test 1 Image"
    elif data_file == 'B' or data_file == 'b':
        frame_text = "Test 2 Image"

    data_set_1 = data.Dataset(data_dirs[which_data][frame_1])
    data_set_2 = data.Dataset(data_dirs[which_data][frame_2])

    for i in range(len(data_set_1.data)):                                           # runs through all images in set
        try:
            frame_1 = data_set_1.next_entry()                                           # get next entries
            frame_2 = data_set_2.next_entry()

            frame_1.x = -1 * frame_1.x
            frame_2.x = -1 * frame_2.x
            temp_y1 = frame_1.y
            temp_y2 = frame_2.y
            frame_1.y = frame_1.z
            frame_2.y = frame_2.z
            frame_1.z = temp_y1
            frame_2.z = temp_y2

            frame_1.amplitude[frame_1.amplitude > 250] = 0                           # adjust maximum
            frame_2.amplitude[frame_2.amplitude > 250] = 0
            frame_1.amplitude[frame_1.amplitude == 0] = np.max(frame_1.amplitude)
            frame_2.amplitude[frame_2.amplitude == 0] = np.max(frame_2.amplitude)

            frame_1.amplitude = gaussian_filter(frame_1.amplitude, sigma=sigma_val)     # start gaussian filter
            frame_1.x = gaussian_filter(frame_1.x, sigma=sigma_val)
            frame_1.y = gaussian_filter(frame_1.y, sigma=sigma_val)
            frame_1.z = gaussian_filter(frame_1.z, sigma=sigma_val)
            frame_2.amplitude = gaussian_filter(frame_2.amplitude, sigma=sigma_val)
            frame_2.x = gaussian_filter(frame_2.x, sigma=sigma_val)
            frame_2.y = gaussian_filter(frame_2.y, sigma=sigma_val)
            frame_2.z = gaussian_filter(frame_2.z, sigma=sigma_val)                     # end gaussian filter

            first_img = frame_1.get_amplitude_image()                                   #adjust image sizes
            first_img = np.uint8(first_img * 255)
            second_img = frame_2.get_amplitude_image()
            second_img = np.uint8(second_img * 255)

            sift_img = cv2.xfeatures2d.SIFT_create()
            kp1, desc1 = sift_img.detectAndCompute(first_img, None)                     # sift test first image
            # first_img = cv2.drawKeypoints(first_img, kp1, None)
            first_loc_xy = [x.pt for x in kp1]

            kp2, desc2 = sift_img.detectAndCompute(second_img, None)                    # sift second image
            # second_img = cv2.drawKeypoints(second_img, kp2, None)
            second_loc_xy = [x.pt for x in kp2]

            img1_points = []
            img2_points = []
            detected_feature_loc1 = []
            detected_feature_loc2 = []

            if desc1 is None or desc2 is None or not kp1 or not kp2:                # remove frames that don't have anything
                cv2.imshow(frame_text + " 1", first_img)
                cv2.imshow(frame_text + " 2", second_img)
            else:
                # compare_desc = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)              # setup brute force matcher
                compare_desc = cv2.BFMatcher()              # setup brute force matcher
                match = compare_desc.knnMatch(desc1, desc2, k=2)
                # match = compare_desc.match(desc1, desc2)
                # print(match)
                # match = sorted(match, key=lambda x:x.distance)
                i = 0
                # print(match)
                for points, points2 in match:  # {
                # for points in match[:NUM_FEATURES]:  # {
                # for points in match:  # {

                    if points.distance > 0.55 * points2.distance:
                        continue
                    # get the x and y pixel locations the features from the matcher
                    # point_image1 = points.queryIdx
                    # point_image2 = points.trainIdx

                    # get points from the key points array that were matched
                    (x1, y1) = kp1[points.queryIdx].pt
                    (x2, y2) = kp2[points.trainIdx].pt

                    # print(np.float32([kp1[points.queryIdx].pt]))
                    # point pairs for each image, indices match pairs
                    # array to hold all (x,y) pixel loc of features
                    img1_points = np.append(img1_points, (x1, y1))
                    img2_points = np.append(img2_points, (x2, y2))

                    # round values and cast to integer
                    img1_points = np.int_(np.round(img1_points, decimals=0, out=None))
                    img2_points = np.int_(np.round(img2_points, decimals=0, out=None))

                    # get the xvect, yvect, and z distance of each point of interest based on the pixel xy location
                    # and round to 4 decimal places.  final full array of all matching points
                    feature_loc1 = np.round(frame_1.get_position(img1_points[i*2], img1_points[i*2+1]), 4)
                    feature_loc2 = np.round(frame_2.get_position(img2_points[i*2], img2_points[i*2+1]), 4)
                    detected_feature_loc1 = np.append(detected_feature_loc1, (feature_loc1[1], feature_loc1[2], feature_loc1[0]))
                    detected_feature_loc2 = np.append(detected_feature_loc2, (feature_loc2[1], feature_loc2[2], feature_loc2[0]))

                    i += 1
                # }

                # show two original frames and matched features
                # disp_match = cv2.drawMatches(first_img, kp1, second_img, kp2, match[:NUM_FEATURES], None, flags=2)
                # disp_match = cv2.drawMatchesKnn(first_img, kp1, second_img, kp2, match, None, flags=2)
                # disp_match = calc.draw_delta(first_img, img1_points, img2_points, scale=1)
                display = np.hstack((first_img, second_img))
                img1_points = img1_points.reshape((-1, 2))
                img2_points = img2_points.reshape((-1, 2))
                display = calc.render_keypoints(display, img1_points, img2_points, scale=3, offset=(first_img.shape[1],0))
                # display = np.vstack((display, disp_match))
                cv2.imshow(frame_text, display)
                cv2.waitKey(10)

            # testing##################
            # print(detected_feature_loc1)
            # print(detected_feature_loc2)

            # after each image loop before getting next image
            # reshape vector into 3D array then take transpose
            detected_feature_loc1 = np.reshape(detected_feature_loc1, (-1, 3))      # format [[x1y1z1][x2y2z2]]
            detected_feature_loc2 = np.reshape(detected_feature_loc2, (-1, 3))      # format [[x1y1z1][x2y2z2]]

            # testing##################
            # print("\n\n")
            # print(detected_feature_loc1)
            # print(detected_feature_loc2)

            detected_feature_loc1_tuple = tuple(detected_feature_loc1)
            detected_feature_loc2_tuple = tuple(detected_feature_loc2)

            # if FIRST:
                # print(detected_feature_loc1_tuple)
                # print("\n")
                # print(detected_feature_loc2_tuple)
            tupled_features = [[tuple(x), tuple(y)] for x, y in zip(detected_feature_loc1, detected_feature_loc2)]
                # print(tupled_features)
                # FIRST = False

            # send to ransac calculation
            param = RansacParams(samples=4, iterations=15, confidence=0.8, threshold=ransac_thresh)
            my_model = ThreeD()
            found_inliers = find_inliers(tupled_features, my_model, param)
            # print(found_inliers)
            if len(found_inliers) == 0:
                continue
            print("\tnumber of inliers: ", len(found_inliers))
            inlier_count = np.append(inlier_count, len(found_inliers))

            new_rotate = my_model.rotate
            new_translation = my_model.translate

            detected_feature_loc1_new = []
            detected_feature_loc2_new = []
            # euler_angles = Rotation.as_euler(new_rotate, 'zxy', degrees=False)
            # print(euler_angles)
            # pyr = []
            pry = calc.rotationMatrixToEulerAngles(new_rotate)
            # print(np.rad2deg(pry))
            #
            all_pitch = np.append(all_pitch, pry[0])
            all_roll = np.append(all_roll, pry[1])
            all_yaw = np.append(all_yaw, pry[2])
            #
            all_x = np.append(all_x, new_translation[0])
            all_y = np.append(all_y, new_translation[1])
            all_z = np.append(all_z, new_translation[2])
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

        except:
            import traceback
            traceback.print_exc()
            pass

    # after all images have been processed
    #if data_file == 'T' or data_file == 't' or data_file == 'B' or data_file == 'b':
    if 1:
        # print("x average (mm): ", np.round(np.mean(all_x) * 1000, 3))   # abs removed -> np.round(np.abs(np.mean(all_x) * 1000), 3)
        # print("y average (mm): ", np.round(np.mean(all_y) * 1000, 3))
        # print("z average (mm): ", np.round(np.mean(all_z) * 1000, 3))
        print("(x, y, z) average (mm): (", np.round(np.mean(all_x) * 1000, 3), ", ", np.round(np.mean(all_y) * 1000, 3),
              ", ", np.round(np.mean(all_z) * 1000, 3), ")")
        print("(x, y, z) std dev (mm): (", np.round(np.std(all_x * 1000), 3), ", ", np.round(np.std(all_y * 1000), 3),
              ", ", np.round(np.std(all_z * 1000), 3), ")\n")
        #print("\nx standard dev: ", np.round(np.std(all_x * 1000), 3))
        #print("y standard dev: ", np.round(np.std(all_y * 1000), 3))
        #print("z standard dev: ", np.round(np.std(all_z * 1000), 3))
    #else:
        # print("roll mean (degrees): ", np.round(np.rad2deg(np.mean(all_roll)), 3))
        # print("pitch mean (degrees): ", np.round(np.rad2deg(np.mean(all_pitch)), 3))
        # print("yaw mean (degrees): ", np.round(np.rad2deg(np.mean(all_yaw)), 3))
        print("(roll, pitch, yaw) average (degrees): (", np.round(np.rad2deg(np.mean(all_roll)), 3), ", ",
              np.round(np.rad2deg(np.mean(all_pitch)), 3), ", ", np.round(np.rad2deg(np.mean(all_yaw)), 3), ")")
        print("(roll, pitch, yaw) std dev (degrees): (", np.round(np.rad2deg(np.std(all_roll)), 3), ", ",
              np.round(np.rad2deg(np.std(all_pitch)), 3), ", ", np.round(np.rad2deg(np.std(all_yaw)), 3), ")")
        #print("\nroll standard dev: ", np.round(np.rad2deg(np.std(all_roll)), 3))
        #print("pitch standard dev: ", np.round(np.rad2deg(np.std(all_pitch)), 3))
        #print("yaw standard dev: ", np.round(np.rad2deg(np.std(all_yaw)), 3))

        # trans = [np.round(np.mean(all_x) * 1000, 3), np.round(np.mean(all_y) * 1000, 3), np.round(np.mean(all_z) * 1000, 3)]
        # rot = [np.round(np.rad2deg(np.mean(all_roll)), 3), np.round(np.rad2deg(np.mean(all_pitch)), 3), np.round(np.rad2deg(np.mean(all_yaw)), 3)]
        # combined = np.reshape([trans, rot], 6)
        # print(tabulate(combined, headers=['x', 'y', 'z', 'roll', 'pitch', 'yaw'], tablefmt='orgtbl'))

        print("\nAverage num inliers/frame: ", np.mean(inlier_count))
