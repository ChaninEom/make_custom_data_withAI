from ntpath import join
import cv2
import os
import mediapipe as mp
import numpy as np
import sys
import math
import img_processing as ip

#yolov5의 label format은 x_center, y_center, width, height

label = "zero"
dir_path = "dataset/"
max_num_hands = 1
train_mode = False

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


def get_label_xy(joint):
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    weight = 0.03

    x_max = np.max(np.transpose(joint)[0])
    x_min = np.min(np.transpose(joint)[0])
    y_max = np.max(np.transpose(joint)[1])
    y_min = np.min(np.transpose(joint)[1])

    raw_r1 = (x_min, y_max)
    raw_r2 = (x_max, y_min)
    r1 = (int((x_min-weight)*w), math.ceil((y_max+weight)*h))        # x좌표값 찾는 식 --> max값은 올림, min값음 내림
    r2 = (math.ceil((x_max+weight)*w), int((y_min-weight)*h))        # y좌표값 찾는 식 --> max값은 올림, min값음 내림
    
    return raw_r1, raw_r2, r1, r2

def make_img_set(img, joint, r1, r2):
    #cv2.waitKey(3000)
    hand_mask = get_hand_mask(img, joint, r1, r2)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path+"/images")
        os.makedirs(dir_path+"/labels")
    ip.save_basic_labelimg(img,dir_path, r1, r2)
    ip.save_geo_labelimg(img,dir_path, r1, r2)
    ip.save_segmented_labelimg(img, hand_mask, dir_path, r1, r2)

def get_result(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return result

def get_hand_mask(img, joint, r1, r2):

    x_min, y_max = r1
    x_max, y_min = r2
    r1 = (int((x_min-0.03)*640), math.ceil((y_max+0.03)*480))
    r2 = (math.ceil((x_max+0.03)*640), int((y_min-0.03)*480))
    mask_img = []
    backporj_format = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (1, 2), (2, 3), (3, 4), 
                        (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), 
                        (15, 16), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
    channels = [1, 2]
    cr_bins = 128
    cb_bins = 128
    histSize = [cr_bins, cb_bins]
    cr_range = [0, 256]
    cb_range = [0, 256]
    ranges = cr_range + cb_range

    for f, s in backporj_format:
        
        first = (round(joint[f][0]*640), round(joint[f][1]*480))
        second = (round(joint[s][0]*640), round(joint[s][1]*480))
        midle = (round((first[0]+second[0])/2), round((first[1]+second[1])/2))
        x, y = midle
        w, h = round(abs((first[0]-x)*0.5)), round(abs((first[1]-y)*0.5))
        img_ycrcb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2YCrCb)
        crop = img_ycrcb[y:y+h, x:x+w]

        hist = cv2.calcHist([crop], channels, None, histSize, ranges)


        backporj = cv2.calcBackProject([img_ycrcb], channels, hist, ranges, 1)
        dst = cv2.copyTo(img, backporj)
        mask_img.append(dst)

    mask = mask_img[0]
    for i in range(len(mask_img)-1):
        mask = cv2.add(mask, mask_img[i+1])

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_area = mask[r2[1]:r1[1], r1[0]:r2[0]].copy()
    mask[:, :] = 0  
    mask_area[mask_area>0] = 255
    mask[r2[1]:r1[1], r1[0]:r2[0]] = mask_area
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, -1)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ###############################ok train 시킬땐 아래 for문 내 튜플 마지막 -1 -> 2로 변경할것
    for contour in contours:
        mask = cv2.drawContours(mask, [contour], -1, 255, 2)
    
    return mask




cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Camera oepn failed!')
    sys.exit()

while True:
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    result = get_result(img)
    key = cv2.waitKey(1)
    if key == ord('r') or key == ord('R'):
        cap = cv2.VideoCapture('rtsp://172.30.1.33:8080/h264_ulaw.sdp')
        continue
    elif key == ord('m') or key == ord('M'):
        cap = cv2.VideoCapture(0)
        continue
    if key == 27:
        break
    if key == ord('t') or key == ord('T'):
        train_mode = not train_mode        

    if train_mode == True:
        if result.multi_hand_landmarks:
            rps_result = []
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]
                
                raw_r1, raw_r2, r1, r2 = get_label_xy(joint)
                
                # print(raw_r1, raw_r2)
                # print(r1, r2)
                if key == ord('a'):
                    mask = get_hand_mask(img, joint, raw_r1, raw_r2)
                    cv2.imshow('mask', mask)

                if key == ord(' '):
                    make_img_set(img, joint, raw_r1, raw_r2)
                    print('sucess!')
                    # sys.exit()

                cv2.rectangle(img, r1, r2, (200, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, "DETECTED", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2, cv2.LINE_AA )                
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                cv2.circle(img, r1, 5, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(img, r2, 5, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (   round((raw_r2[0]+raw_r1[0])*640/2), round((raw_r1[1]+raw_r2[1])*480/2)   ), 5, (255,0,0), -1, cv2.LINE_AA)
        cv2.putText(img, "TRAIN MODE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 0, 0), 2, cv2.LINE_AA )
        cv2.putText(img, "label : {}".format(label), (25,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2, cv2.LINE_AA)


    cv2.imshow('img', img)


cv2.destroyAllWindows()