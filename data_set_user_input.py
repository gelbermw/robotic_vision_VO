import cv2
import odataset as data
import functions as calc
import numpy as np
import string
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

NUM_FEATURES = 5

img_type = int(input("Enter 1 for single image or 2 to compare images: "))
data_file = input("Select which type of data to view:\n\tT for Translation\n\tP for Pitch\n\tY for Yaw\n\tV for Video\n")
which_data = data_file.lower()
which_data += "_data"
print(which_data)
print("Starting frame assumed as first set of data")
frame_1 = 0
frame_2 = 0
if img_type == 2:
    frame_2 = int(input("Select a number (2, 3, or 4) for what data to be compared: "))
    frame_2 -= 1
    frame_num = frame_2
avg = 0

# initialize list of matching point pairs
img1_points = []
img2_points = []
alpha_arr = []
beta_arr = []
gamma_arr = []


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
    data_set_1 = data.Dataset(data_dirs[which_data][frame_1])
    if data_file == 'T' or data_file == 't':
        frame_text = "Translation Image"
    elif data_file == 'P' or data_file == 'p':
        frame_text = "Pitch Image"
    elif data_file == 'Y' or data_file == 'y':
        frame_text = "Yaw Image"

    if frame_2 != 0:
        data_set_2 = data.Dataset(data_dirs[which_data][frame_2])
        for i in range(len(data_set_1.data)):
        #while True:
            frame_1 = data_set_1.next_entry()
            frame_2 = data_set_2.next_entry()

            # amp = frame.get_combined_image()
            first_img = frame_1.get_amplitude_image()
            # single_avg = np.average(img_1)
            first_img = np.uint8(first_img * 255)

            second_img = frame_2.get_amplitude_image()
            # single_avg = np.average(second_img)
            second_img = np.uint8(second_img * 255)

            # sift test first image
            sift_img = cv2.xfeatures2d.SIFT_create()
            kp1, desc1 = sift_img.detectAndCompute(first_img, None)
            first_img = cv2.drawKeypoints(first_img, kp1, None)
            # cv2.imshow(frame_text1, first_img)
            first_loc_xy = [x.pt for x in kp1]

            # sift second image
            kp2, desc2 = sift_img.detectAndCompute(second_img, None)
            second_img = cv2.drawKeypoints(second_img, kp2, None)
            # cv2.imshow(frame_text2, second_img)
            second_loc_xy = [x.pt for x in kp2]

            # compare_desc = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            if desc1 == "None" or desc2 == "None" or not kp1 or not kp2:
                cv2.imshow(frame_text + " 1", first_img)
                cv2.imshow(frame_text + " 2", second_img)
            else:
                compare_desc = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
                match = compare_desc.match(desc1,desc2)
                match = sorted(match, key=lambda x:x.distance)
                i = 0
                for points in match[:NUM_FEATURES]: # {
                    # get the x and y pixel locations the features from the matcher
                    point_image1 = points.queryIdx
                    point_image2 = points.trainIdx

                    # get points from the keypoints array that were matched
                    (x1,y1) = kp1[point_image1].pt
                    (x2,y2) = kp2[point_image2].pt

                    # point pairs for each image, indices match pairs
                    img1_points = np.append(img1_points, (x1,y1))
                    img2_points = np.append(img2_points, (x2,y2))
                    # img1_points = np.append([(x1,y1)])
                    # img2_points = np.append([(x2,y2)])

                    img1_points = np.int_(np.round(img1_points, decimals=0, out=None))
                    img2_points = np.int_(np.round(img2_points, decimals=0, out=None))

                    # print("point 1: ", img1_points, " | point 2:", img2_points)    # debugging lines
                    print("run # ", i)

                    # num1 = len(img1_points) / 2    #debugging lines
                    # print("number of image 1 points", len(img1_points) / 2)    # debugging lines
                    # print("number of image 2 points", len(img2_points) / 2)    # debugging lines

                    # get the xvect, yvect, and z distance of each point of interest based on the pixel xy location
                    # and round to 4 decimal places.
                    feature_loc1 = np.round(frame_1.get_position(img1_points[i*2], img1_points[i*2+1]),4)
                    feature_loc2 = np.round(frame_2.get_position(img2_points[i*2], img2_points[i*2+1]),4)
                    # print(feature_loc1, " | ", feature_loc2)    #debugging lines

                    # calculate distance between camera and each point of interest
                    dist_cam_to_1 = calc.distance_to_cam(feature_loc1[0], feature_loc1[1], feature_loc1[2])
                    dist_cam_to_2 = calc.distance_to_cam(feature_loc2[0], feature_loc2[1], feature_loc2[2])
                    dist_1_to_2 = calc.distance_between_pts(feature_loc2[0], feature_loc2[1], feature_loc2[2],
                                                            feature_loc1[0], feature_loc1[1], feature_loc1[2])
                    # print(dist_cam_to_1, " | ", dist_cam_to_2, " | ", dist_1_to_2)    # debugging lines

                    alpha = np.rad2deg(calc.camera_angle(dist_1_to_2, dist_cam_to_2, dist_cam_to_1))
                    beta = np.rad2deg(calc.point1_angle(dist_1_to_2, dist_cam_to_2, dist_cam_to_1))
                    gamma = np.rad2deg(calc.point2_angle(dist_1_to_2, dist_cam_to_2, dist_cam_to_1))
                    alpha_arr = np.append(alpha_arr, alpha)
                    beta_arr = np.append(beta_arr, beta)
                    gamma_arr = np.append(gamma_arr, gamma)
                    print(alpha, " | ", beta - 90, " | ", gamma - 90)    # debugging lines

                    i += 1
                # }

                # print(len(match))

                # for dot in match: # finding distance between matches, the lower the distance, the better it is
                #     print(dot.distance)

                disp_match = cv2.drawMatches(first_img, kp1, second_img, kp2, match[:5], None, flags=2)
                cv2.imshow(frame_text + " 1", first_img)
                cv2.imshow(frame_text + " 2", second_img)
                cv2.imshow("Confirmed Matching Features", disp_match)
                # if len(loc_xy) > 10:
                #    print(single_avg)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        alpha_avg = np.average(alpha_arr)
        alpha_err = abs(alpha_avg - (frame_num * 3))
        print(alpha_avg, " | ", alpha_err, " | ", np.std(alpha_arr))

    else: #only looking at one image for testing purposes

        # ####image display
        img = np.zeros_like(data_set_1.data[0].amplitude)
        img = np.float32(img)
        if avg == 1:
            frame_text += ": averaged"
            i = 0
            for frame in data_set_1.data:
                # ###normal method
                # img = img + frame.get_amplitude_image()

                # ###filtering method
                new = frame.get_amplitude_image()
                single_avg = np.average(new)
                # print("image#", i, "| ", single_avg)
                if single_avg > 0.107:
                    img = img + new
                    i += 1
                    print("image#", i, "| ", single_avg)
                else:
                    data_set_1.next_entry()
            # ####normal method
            # img = img / len(data_set.data)

            # ###filtered method
            img = img / i
            img = np.uint8(img * 255)
            # np.set_printoptions(threshold=sys.maxsize)
            # print(img)
            # print("max of average array: ", np.amax(img))

            # sift test
            sift_img = cv2.xfeatures2d.SIFT_create()
            kp = sift_img.detect(img, None)
            img = cv2.drawKeypoints(img, kp, None)
            loc_xy = [x.pt for x in kp]
            loc_xy = np.round(loc_xy,decimals=0,out=None)
            print(loc_xy)
            height, width = img.shape[:2]
            cv2.namedWindow(frame_text, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(frame_text, width, height)
            cv2.imshow(frame_text, img)


            # #confidence adjustment
            # frame_text += " + conf"
            # for frame in data_set.data:
                # con = frame.get_confidence_image()

            cv2.waitKey(0)

        else:
            while True:
                frame = data_set_1.next_entry()
                # amp = frame.get_combined_image()
                img = frame.get_amplitude_image()
                single_avg = np.average(img)
                img = np.uint8(img * 255)

                # sift test
                sift_img = cv2.xfeatures2d.SIFT_create()
                kp = sift_img.detect(img, None)
                img = cv2.drawKeypoints(img, kp, None)
                cv2.imshow(frame_text, img)
                loc_xy = [x.pt for x in kp]
                if len(loc_xy) > 10:
                    print(single_avg)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

elif frame_1 == 0 and (data_file == 'V' or data_file == 'v'):
    data_set = data.Dataset(data_dirs[which_data][frame_1])
    frame_text = "Video Data"
    while True:
        frame = data_set.next_entry()
        # amp = frame.get_combined_image()
        amp = frame.get_amplitude_image()
        amp = np.uint8(amp*255)

        # sift test
        sift_img = cv2.xfeatures2d.SIFT_create()
        kp = sift_img.detect(amp, None)
        amp = cv2.drawKeypoints(amp, kp, None)

        cv2.imshow(frame_text, amp)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
