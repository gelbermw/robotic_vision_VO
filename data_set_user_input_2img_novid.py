import cv2
import odataset as data
import functions as calc
import numpy as np
import transform as tr
from scipy.ndimage.filters import gaussian_filter
import string
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# NUM_FEATURES = 4       # works for pitch and yaw translation 100 and 300
NUM_FEATURES = 6
# sigma_val = 1.0     # works for pitch and yaw
sigma_val = 1.0         # 0.6 works for translation 100 and 300
while True:
    img_type = 2
    data_file = input("Select which type of data to view:\n\tT for Translation\n\tP for Pitch\n\tY for Yaw\n")
    which_data = data_file.lower()
    which_data += "_data"
    frame_1 = 0
    frame_2 = 0
    if img_type == 2:
        frame_2 = int(input("Select a number (2, 3, or 4) for what data to be compared: "))
        frame_2 -= 1
        frame_num = frame_2
    avg = 0
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

    data_dirs = {'t_data': [('./data/Translation/Y1/', 'trans_1'),
                           ('./data/Translation/Y2/', 'trans_2'),
                           ('./data/Translation/Y3/', 'trans_3'),
                           ('./data/Translation/Y4/', 'trans_4')],
                 'p_data': [('./data/Pitch/d1_-40/', 'pitch_1'),
                           ('./data/Pitch/d2_-37/', 'pitch_2'),
                           ('./data/Pitch/d3_-34/', 'pitch_3'),
                           ('./data/Pitch/d4_-31/', 'pitch_4')],
                 'y_data': [('./data/Yaw/d1_44/', 'yaw_1'),
                         ('./data/Yaw/d2_41/', 'yaw_2'),
                         ('./data/Yaw/d3_38/', 'yaw_3'),
                         ('./data/Yaw/d4_35/', 'yaw_4')],
                 'v_data': [('./data/RV_Data2/', 'vid')]}

    subP = 0
    subY = 0
    subT = 0
    if data_file == 'T' or data_file == 't':
        frame_text = "Translation Image"
        # subT = frame_num * 0.1
        # subP = 0
        # subY = 0
    elif data_file == 'P' or data_file == 'p':
        frame_text = "Pitch Image"
        # subP = frame_num * 3
        # subY = 0
        # subT = 0
    elif data_file == 'Y' or data_file == 'y':
        frame_text = "Yaw Image"
        # subP = 0
        # subY = frame_num * 3
        # subT = 0

    data_set_1 = data.Dataset(data_dirs[which_data][frame_1])
    data_set_2 = data.Dataset(data_dirs[which_data][frame_2])

    for i in range(len(data_set_1.data)):                                           # runs through all images in set
        try:
            frame_1 = data_set_1.next_entry()                                           # get next entries
            frame_2 = data_set_2.next_entry()

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
            first_img = cv2.drawKeypoints(first_img, kp1, None)
            first_loc_xy = [x.pt for x in kp1]

            kp2, desc2 = sift_img.detectAndCompute(second_img, None)                    # sift second image
            second_img = cv2.drawKeypoints(second_img, kp2, None)
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
                # print(match)
                # match = sorted(match, key=lambda x:x.distance)
                i = 0
                # print(match)
                for points, points2 in match:  # {
                # for points in match[:NUM_FEATURES]:  # {
                # for points in match:  # {
                    if points.distance > 0.6 * points2.distance:
                        continue
                    # get the x and y pixel locations the features from the matcher
                    # point_image1 = points.queryIdx
                    # point_image2 = points.trainIdx

                    # get points from the key points array that were matched
                    (x1, y1) = kp1[points.queryIdx].pt
                    (x2, y2) = kp2[points.trainIdx].pt

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
                disp_match = cv2.drawMatchesKnn(first_img, kp1, second_img, kp2, match, None, flags=2)
                display = np.hstack((first_img, second_img))
                display = np.vstack((display, disp_match))
                cv2.imshow(frame_text, display)
                # cv2.imshow(frame_text + " 1", first_img)
                # cv2.imshow(frame_text + " 2", second_img)
                # cv2.imshow("Confirmed Matching Features", disp_match)

            # after each image loop before getting next image
            # reshape vector into 2D array then take transpose
            detected_feature_loc1 = np.reshape(detected_feature_loc1, (-1, 3))
            detected_feature_loc2 = np.reshape(detected_feature_loc2, (-1, 3))

            H, status = cv2.findHomography(detected_feature_loc1, detected_feature_loc2, cv2.RANSAC,
                                           ransacReprojThreshold=thresh)
            # sort through detected features and remove ones with 0 status from
            detected_feature_loc1_new = []
            detected_feature_loc2_new = []
            for j in range(len(detected_feature_loc1)):
                if status[j] == 1:
                    detected_feature_loc1_new.append(detected_feature_loc1[j])
                    detected_feature_loc2_new.append(detected_feature_loc2[j])

            detected_feature_loc1_new = np.round(np.transpose(detected_feature_loc1_new), 4)
            # detected_feature_loc1_new = np.round(calc.transpose_array(detected_feature_loc1_new), 4)
            detected_feature_loc2_new = np.round(np.transpose(detected_feature_loc2_new), 4)

            rotate, translate = tr.rigid_transform_3D(np.asmatrix(detected_feature_loc1_new),
                                                      np.asmatrix(detected_feature_loc2_new))

            detected_feature_loc1_new = []
            detected_feature_loc2_new = []

            # pyr = []
            pyr = calc.rotationMatrixToEulerAngles(rotate)

            all_pitch = np.append(all_pitch, pyr[0])
            all_roll = np.append(all_roll, pyr[2])
            all_yaw = np.append(all_yaw, pyr[1])

            all_x = np.append(all_x, translate[0] * -1)
            all_y = np.append(all_y, translate[2] * -1)
            all_z = np.append(all_z, translate[1] * -1)

            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

        except:
            pass

    # after all images have been processed
    if data_file == 'T' or data_file == 't':
        print("x average (meters): ", np.abs(np.mean(all_x) * 1000))
        print("y average (meters): ", np.abs(np.mean(all_y) * 1000))
        print("z average (meters): ", np.abs(np.mean(all_z) * 1000))
        print("x standard dev: ", np.std(all_x * 1000))
        print("y standard dev: ", np.std(all_y * 1000))
        print("z standard dev: ", np.std(all_z * 1000))
    else:
        print("pitch mean error (degrees): ", np.abs(subP - np.rad2deg(np.mean(all_pitch))))
        print("roll mean error (degrees): ", np.abs(np.rad2deg(np.mean(all_roll))))
        print("yaw mean error (degrees): ", np.abs(subY - np.rad2deg(np.mean(all_yaw))))
        print("pitch standard dev: ", np.rad2deg(np.std(all_pitch)))
        print("roll standard dev: ", np.rad2deg(np.std(all_roll)))
        print("yaw standard dev: ", np.rad2deg(np.std(all_yaw)))
