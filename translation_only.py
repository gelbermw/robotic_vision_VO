import cv2
import odataset as dat
import functions as calc
import numpy as np
import transform as tr
from scipy.ndimage.filters import gaussian_filter
from skimage import data, transform
# from skimage import
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

NUM_FEATURES = 4
sigma_val = 1.0

img_type = 2
data_file = 'T'
which_data = data_file.lower()
which_data += "_data"
# print("Starting frame assumed as first set of data")
frame_1 = 0
frame_2 = 0
if img_type == 2:
    frame_2 = int(input("Select a number (2, 3, or 4) for what data to be compared: "))
    frame_2 -= 1
    frame_num = frame_2

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

if frame_1 == 0 and (data_file != 'V' and data_file != 'v'):
    data_set_1 = dat.Dataset(data_dirs[which_data][frame_1])
    frame_text = "Translation Image"

    if frame_2 != 0:
        data_set_2 = dat.Dataset(data_dirs[which_data][frame_2])
        for i in range(len(data_set_1.data)):
        #while True:
            frame_1 = data_set_1.next_entry()
            frame_2 = data_set_2.next_entry()
            # if data_file == 'P' or data_file == 'p'
            frame_1.amplitude[frame_1.amplitude == 65533] = 0
            frame_2.amplitude[frame_2.amplitude == 65533] = 0
            frame_1.amplitude = gaussian_filter(frame_1.amplitude, sigma=sigma_val)
            frame_1.x = gaussian_filter(frame_1.x, sigma=sigma_val)
            frame_1.y = gaussian_filter(frame_1.y, sigma=sigma_val)
            frame_1.z = gaussian_filter(frame_1.z, sigma=sigma_val)
            frame_2.amplitude = gaussian_filter(frame_2.amplitude, sigma=sigma_val)
            frame_2.x = gaussian_filter(frame_2.x, sigma=sigma_val)
            frame_2.y = gaussian_filter(frame_2.y, sigma=sigma_val)
            frame_2.z = gaussian_filter(frame_2.z, sigma=sigma_val)
            first_img = frame_1.get_amplitude_image()
            first_img = np.uint8(first_img * 255)

            second_img = frame_2.get_amplitude_image()
            second_img = np.uint8(second_img * 255)

            # sift test first image
            sift_img = cv2.xfeatures2d.SIFT_create()
            kp1, desc1 = sift_img.detectAndCompute(first_img, None)
            first_img = cv2.drawKeypoints(first_img, kp1, None)
            first_loc_xy = [x.pt for x in kp1]

            # sift second image
            kp2, desc2 = sift_img.detectAndCompute(second_img, None)
            second_img = cv2.drawKeypoints(second_img, kp2, None)
            second_loc_xy = [x.pt for x in kp2]

            img1_points = []
            img2_points = []
            detected_feature_loc1 = []
            detected_feature_loc2 = []

            # compare_desc = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            if desc1 == "None" or desc2 == "None" or not kp1 or not kp2:
                cv2.imshow(frame_text + " 1", first_img)
                cv2.imshow(frame_text + " 2", second_img)
            else:
                compare_desc = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
                match = compare_desc.match(desc1, desc2)
                match = sorted(match, key=lambda x:x.distance)
                i = 0
                for points in match[:NUM_FEATURES]:  # {
                # for points in match:  # {
                    # get the x and y pixel locations the features from the matcher
                    point_image1 = points.queryIdx
                    point_image2 = points.trainIdx

                    # get points from the keypoints array that were matched
                    (x1,y1) = kp1[point_image1].pt
                    (x2,y2) = kp2[point_image2].pt

                    # point pairs for each image, indices match pairs
                    img1_points = np.append(img1_points, (x1,y1))
                    img2_points = np.append(img2_points, (x2,y2))

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

                disp_match = cv2.drawMatches(first_img, kp1, second_img, kp2, match[:5], None, flags=2)
                cv2.imshow(frame_text + " 1", first_img)
                cv2.imshow(frame_text + " 2", second_img)
                cv2.imshow("Confirmed Matching Features", disp_match)
                # if len(loc_xy) > 10:
                #    print(single_avg)

            # after each image loop before getting next image
            # reshape vector into 2D array then take transpose
            detected_feature_loc1 = np.reshape(detected_feature_loc1, (-1, 3))
            detected_feature_loc2 = np.reshape(detected_feature_loc2, (-1, 3))

            H, status = cv2.findHomography(detected_feature_loc1, detected_feature_loc2, cv2.RANSAC,
                                           ransacReprojThreshold=0.0001)
            # print(H)
            detected_feature_loc1_new = []
            detected_feature_loc2_new = []
            for j in range(len(detected_feature_loc1)):
                if status[j] == 1:
                    detected_feature_loc1_new.append(detected_feature_loc1[j])
                    detected_feature_loc2_new.append(detected_feature_loc2[j])
            detected_feature_loc1_new = np.round(calc.transpose_array(detected_feature_loc1_new), 4)
            detected_feature_loc2_new = np.round(calc.transpose_array(detected_feature_loc2_new), 4)

            rotate, translate = tr.rigid_transform_3D(np.asmatrix(detected_feature_loc1_new),
                                                      np.asmatrix(detected_feature_loc2_new))

            detected_feature_loc1_new = []
            detected_feature_loc2_new = []

            # ####trying new thing for translation
            # new_H = np.transpose(translate) * rotate * translate
            # # S = []
            # x_scale = 1
            # y_scale = 1
            # x_skew = 0.00001
            # y_skew = 0.00001
            # S = [[x_scale, 0, 0], [0, y_scale, 0], [x_skew, y_skew, 1]]
            # S = np.asarray(S)
            # r, c = first_img.shape
            # T = [[1, 0, -c/2], [0, 1, -r/2], [0, 0, 1]]
            # T = np.array(T)
            # img1_new = cv2.warpPerspective(first_img, H, (144, 132))
            # img1_rot = cv2.warpPerspective(img1_new, S.dot(np.linalg.inv(T).dot(new_H).dot(translate)), (144, 132))
            # cv2.imshow("skewed", img1_rot)
            # img1_rotate = transform.homography(first_img, new_H)

            # k = first_img
            # k_inv = np.linalg.inv(k)
            # Hi = k.dot(rotate)
            # Hii = k_inv.dot(H)
            # H1 = Hi.dot(Hii)
            # # print ("H1 = ", H1)
            # im_out = cv2.warpPerspective(first_img, H1, (first_img.shape[1], first_img.shape[0]))
            # cv2.imshow("skewed", im_out)
            # test = [[0, 0, 0],[0, 0, 0],[0, 0, 0]]
            # cam_rotate, cam_translate = tr.rigid_transform_3D(rotate, np.asmatrix(test))
            # new_T = cam_translate * np.transpose(translate)
            # print(new_T)

            # row, col, ch = first_img.shape
            # m = cv2.getAffineTransform(img1_points, img2_points)
            # dst = cv2.warpAffine(first_img, m, (col, row))
            # cv2.imshow("skewed", dst)
            new_rot = rotate[0:5]
            (h, w) = first_img.shape
            center = (w/2, h/2)
            skew_img = cv2.warpAffine(first_img, new_rot, (w, h))
            cv2.imshow("skew", skew_img)

            # print(S[0][0])
            # #####end of new thing
            pyr = calc.rotationMatrixToEulerAngles(rotate)

            all_pitch = np.append(all_pitch, pyr[0])
            all_roll = np.append(all_roll, pyr[2])
            all_yaw = np.append(all_yaw, pyr[1])

            all_x = np.append(all_x, translate[0])
            all_y = np.append(all_y, translate[1])
            all_z = np.append(all_z, translate[2])

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        # print("pitch average (degrees): ", np.rad2deg(np.mean(all_pitch)))
        # print("roll average (degrees): ", np.rad2deg(np.mean(all_roll)))
        # print("yaw average (degrees): ", np.rad2deg(np.mean(all_yaw)))
        print("x mean (millimeters): ", np.mean(all_x*1000))
        print("y mean (millimeters): ", np.mean(all_y*1000))
        print("z mean (millimeters): ", np.mean(all_z*1000))
