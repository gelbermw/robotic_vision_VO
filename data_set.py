import cv2
import odataset as data
import numpy as np
import sys


# ###### select first data to use
T = -1
P = -1
Y = 0
V = -1
avg = 1


def normalize(data_1, max_value=1, min_value=0):
    data_1 = np.abs(data_1)
    mx = np.max(data_1)
    mn = np.min(data_1)
    mx = max(mx, abs(mn))
    mn = 0
    return (data_1 - mn) / (mx - mn) * (max_value - min_value) + min_value


data_dirs = {'trans': [('./data/Translation/Y1/', 'trans_1'),
                       ('./data/Translation/Y2/', 'trans_2'),
                       ('./data/Translation/Y3/', 'trans_3'),
                       ('./data/Translation/Y4/', 'trans_4')],
             'pitch': [('./data/Pitch/d1_-40/', 'pitch_1'),
                       ('./data/Pitch/d2_-37/', 'pitch_2'),
                       ('./data/Pitch/d3_-34/', 'pitch_3'),
                       ('./data/Pitch/d4_-31/', 'pitch_4')],
             'yaw': [('./data/Yaw/d1_44/', 'yaw_1'),
                     ('./data/Yaw/d2_41/', 'yaw_2'),
                     ('./data/Yaw/d3_38/', 'yaw_3'),
                     ('./data/Yaw/d4_35/', 'yaw_4')],
             'video': [('./data/RV_Data2/', 'vid')]}

if T >= 0 or P >= 0 or Y >= 0:
    if T >= 0:
        data_set = data.Dataset(data_dirs['trans'][T])
        frame_text = "Translation Image"
    elif P >= 0:
        data_set = data.Dataset(data_dirs['pitch'][P])
        frame_text = "Pitch Image"
    elif Y >= 0:
        data_set = data.Dataset(data_dirs['yaw'][Y])
        frame_text = "Yaw Image"

    # ####image display
    img = np.zeros_like(data_set.data[0].amplitude)
    img = np.float32(img)
    if avg == 1:
        frame_text += ": averaged"
        i = 0
        for frame in data_set.data:
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
                data_set.next_entry()
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
            frame = data_set.next_entry()
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

elif V >= 0:
    data_set = data.Dataset(data_dirs['video'][V])
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



