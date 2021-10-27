import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os

from torchvision.models import segmentation
import segmentation as seg

background_route = 'editing_img/'

# r1 = ((0.4693355858325958, 0.5777826309204102))
# r2 = (0.5774698853492737, 0.3952954411506653)
# dir_path = "dataset"
# img = cv2.imread('test_img.jpg', cv2.IMREAD_COLOR)

# mid = ((r2[0]+r1[0])/2, (r1[1]+r2[1])/2)
# width = r2[0]-r1[0]+0.06
# height = r1[1]- r2[1]+0.06
# print(mid, width, height)

# if img is None:
#     print('Image load faield!')
#     sys.exit()

def convert_r1_r2(r1, r2):
    t_weight = 0.03
    r1 = (int((r1[0]-t_weight)*640), math.ceil((r1[1]+t_weight)*480))        # x좌표값 찾는 식 --> max값은 올림, min값음 내림
    r2 = (math.ceil((r2[0]+t_weight)*640), int((r2[1]-t_weight)*480))
    return r1, r2
def get_flag_layerdimg(img, r1, r2):
    basic = basic_process()
    num = 7
    flag = []
    for i in range(num):
        TF = random.randint(0,3)
        if TF == 0:
            FLAG = True
        else:
            FLAG = False
        flag.append(FLAG)
    if flag[0] == True:
        flag[6] == False
    img = basic.brightness(img, flag[0], r1, r2)
    img = basic.contrast(img, flag[1], r1, r2)
    img = basic.g_blur(img, flag[2], r1, r2)
    img = basic.sharpning(img, flag[3], r1, r2)
    img = basic.bilateral(img, flag[4], r1, r2)
    img = basic.blocked(img, flag[5], r1, r2)
    img = basic.shadowed(img, flag[6], r1, r2)
    # print(flag)
    return img
def save_mecro(src, r1, r2, start_num, dir_path):
    cv2.imwrite(dir_path + "/images/{}.jpg".format(start_num), src)
    f = open(dir_path + "/labels/{}.txt".format(start_num), 'w')
    data = "0 {} {} {} {}".format((r2[0]+r1[0])/2, (r1[1]+r2[1])/2, (r2[0]-r1[0])+0.06, (r1[1]- r2[1])+0.06)
    f.write(data)
    f.close()

def save_basic_labelimg(img, dir_path, r1, r2):
    flag = True
    bp = basic_process()

    dir_length = len(os.listdir(dir_path+'/images'))
    if dir_length == 0:
        start_num = 0
    else:
        start_num = dir_length

    img = [bp.brightness(img, flag, r1, r2), bp.contrast(img, flag, r1, r2), bp.g_blur(img, flag, r1, r2),
            bp.shadowed(img, flag, r1, r2), bp.sharpning(img, flag, r1, r2), bp. sharpning_contrast(img, flag, r1, r2),
            bp.bilateral(img, flag, r1, r2)]

    for i in range(7):
        save_mecro(img[i], r1, r2, start_num, dir_path)
        # cv2.imwrite(dir_path + "/images/{}.jpg".format(start_num), img[i])
        # f = open(dir_path + "/labels/{}.txt".format(start_num), 'w')
        # data = "0 {} {} {} {}".format((r2[0]+r1[0])/2, (r1[1]+r2[1])/2, r2[0]-r1[0], r1[1]- r2[1])
        # f.write(data)
        # f.close()
        start_num += 1
        
def save_geo_labelimg(img, dir_path, r1, r2):     
    flag = True
    dir_length = len(os.listdir(dir_path+'/images'))
    gp = geometry_process()
    if dir_length == 0:
        start_num = 0
    else:
        start_num = dir_length

    shift_img, shift_r1, shift_r2 = gp.shift(get_flag_layerdimg(img, r1, r2), flag, False, r1, r2)
    sym_img, sym_r1, sym_r2 = gp.symmetry(get_flag_layerdimg(img, r1, r2), flag, False, r1, r2)
    rot_img, rot_r1, rot_r2 = gp.rotation(get_flag_layerdimg(img, r1, r2), flag, False, r1, r2)

    save_mecro(shift_img, shift_r1, shift_r2, start_num, dir_path)
    start_num += 1
    save_mecro(sym_img, sym_r1, sym_r2, start_num, dir_path)
    start_num += 1
    save_mecro(rot_img, rot_r1, rot_r2, start_num, dir_path)
    start_num += 1

    for i in range(2):
        shift_img, shift_r1, shift_r2 = gp.shift(img, flag, False, r1, r2)
        sym_img, sym_r1, sym_r2 = gp.symmetry(shift_img, flag, False, shift_r1, shift_r2)
        rot_img, rot_r1, rot_r2 = gp.rotation(img, flag, False, r1, r2)

        shift_img = shift_img.copy()
        sym_img = sym_img.copy()
        rot_img = rot_img.copy()
        
        save_mecro(shift_img, shift_r1, shift_r2, start_num, dir_path)
        start_num += 1
        save_mecro(sym_img, sym_r1, sym_r2, start_num, dir_path)
        start_num += 1
        save_mecro(rot_img, rot_r1, rot_r2, start_num, dir_path)
        start_num += 1

def save_segmented_labelimg(img, hand_mask, dir_path, r1, r2):
    gp = geometry_process()
    background_flag = True
    background_length = len(os.listdir(background_route))
    background = cv2.imread(background_route + "{}.jpg".format(random.randint(0,background_length-1)))
    background = seg.get_resized_background(img, background)
    people_mask = cv2.cvtColor(seg.people_seg_mask(img).astype(np.uint8)*255, cv2.COLOR_BGR2GRAY)
    people_mask = cv2.bitwise_or(people_mask, hand_mask)
    dir_length = len(os.listdir(dir_path+'/images'))
    if dir_length == 0:
        start_num = 0
    else:
        start_num = dir_length

    for i in range(6):

        src = img.copy()
        result, foreground, r_r1, r_r2 = src, people_mask, r1, r2
        background = cv2.imread(background_route + "{}.jpg".format(random.randint(0,background_length-1)))
        background = seg.get_resized_background(img, background)

        if random.randint(0, 2) != 0:
            src, r_r1, r_r2, = gp.symmetry(src, True, background_flag, r_r1, r_r2)
            foreground = cv2.flip(foreground, 1)
            
        if random.randint(0, 2) != 0:
            src = src.copy()
            src, r_r1, r_r2, x, y = gp.shift(src, True, background_flag, r_r1, r_r2)

            aff = np.array([[1, 0, x], [0, 1, y]], dtype=np.float32)
            foreground = cv2.warpAffine(foreground, aff, (0, 0))
        
        if random.randint(0, 2) != 0:
            src = src.copy()
            src, r_r1, r_r2, rot = gp.rotation(src, True, background_flag, r_r1, r_r2)
            foreground = cv2.warpAffine(foreground, rot, (0, 0))
            
        result = cv2.copyTo(src, foreground, background)
        save_mecro(result, r_r1, r_r2, start_num, dir_path)
        start_num += 1
        result = get_flag_layerdimg(result, r_r1, r_r2)
        save_mecro(result, r_r1, r_r2, start_num, dir_path)
        start_num += 1

        # print(h1, h2)
        # cv2.imshow('test', result)
        # cv2.waitKey()
        # cv2.imshow('res', result)
        # cv2.waitKey()

class basic_process:
    def brightness(self, src, flag, r1, r2):
        if flag == False:
            return src
        if random.randint(0, 1) == 0:
            rate = random.randint(-50, -20)
        else:
            rate = random.randint(20, 50)
        dst = cv2.add(src, (rate, rate, rate, 0))
        return dst
    
    def shadowed(self, src, flag, r1, r2):
        if flag == False:
            return src
        r1, r2 = convert_r1_r2(r1, r2)
        mid = (int((r1[0]+r2[0])/2), int((r1[1]+r2[1])/2))
        width = r2[0]-r1[0]
        height = r1[1]- r2[1]
        
        x_rate = random.randint(-width, width)//2
        y_rate = random.randint(-height, height)//2

        shadow_file_num = random.randint(1, 2)
        shadow_file_name = f'light_{shadow_file_num}.jpg'
        shadowed_img = cv2.imread(shadow_file_name)



        aff = np.array([[1, 0, mid[0]-320+x_rate],
                [0, 1, mid[1]-240+y_rate]], dtype=np.float32)
        shadowed_img = cv2.warpAffine(shadowed_img, aff, (0, 0), borderValue=(180,180,180))
        dst = cv2.subtract(src.copy(), shadowed_img)
        return dst

    def contrast(self, src, flag, r1, r2):
        if flag == False:
            return src
        img_yuv = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)

        #밝기 채널에 대해서 CLAHE 적용
        dst = img_yuv.copy()
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) #CLAHE 생성
        dst[:,:,0] = clahe.apply(dst[:,:,0])           #CLAHE 적용
        dst = cv2.cvtColor(dst, cv2.COLOR_YUV2BGR)
        return dst

    def g_blur(self, src, flag, r1, r2):
        if flag == False:
            return src
        r1, r2 = convert_r1_r2(r1, r2)
        x, y, w, h = r1[0], r1[1], r2[0]-r1[0], r1[1]-r2[1]
        alpha = round(random.uniform(2., 3.), 2)
        dst = src.copy()
        blur_area = cv2.GaussianBlur(src[y-h:y, x:x+w], (0, 0 ), alpha)
        dst[y-h:y, x:x+w] = blur_area
        return dst

    def g_blur_contrast(self, src, flag, r1, r2):
        if flag == False:
            return src
        dst = self.contrast(src, flag, r1, r2).copy()
        dst = self.g_blur(src, flag, r1, r2)
        return dst

    def sharpning(self, src, flag, r1, r2):
        if flag == False:
            return src
        src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
        src_f = src_ycrcb[:, :, 0].astype(np.float32) #ycrcb plane에서 0번째 plane만 사용함
        blr = cv2.GaussianBlur(src_f, (0, 0), 2.0) 
        src_ycrcb[:, :, 0] = np.clip(2.*src_f-blr, 0, 255).astype(np.uint8)
        dst = cv2.cvtColor(src_ycrcb, cv2.COLOR_YCrCb2BGR)
        return dst

    def sharpning_contrast(self, src, flag, r1, r2):
        if flag == False:
            return src
        dst = self.contrast(src, flag, r1, r2)
        dst = self.sharpning(src, flag, r1, r2)
        return dst

    def bilateral(self, src, flag, r1, r2):
        if flag == False:
            return src
        dst = self.contrast(src, flag, r1, r2).copy()
        dst = cv2.bilateralFilter(src, -1, 10, 5)
        return dst

    def blocked(self, src, flag, r1, r2):
        if flag == False:
            return src
        case = [(0, 0), (320, 0), (0, 240), (320, 240)]
        r1, r2 = convert_r1_r2(r1, r2)
        x, y, w, h = r1[0], r1[1], r2[0]-r1[0], r1[1]-r2[1]
        dst = src.copy()
        keep_img = src[y-h:y, x:x+w].copy()


        for i, start in enumerate(case):

            for i in range(1, random.randint(2, 3)):
                x_range = random.randint(70, 140)
                y_range = random.randint(50, 100)
                rect_x = random.randint(start[0]+round(x_range/2), start[0]+320-round(x_range/2))
                rect_y = random.randint(start[1]+y_range,start[1]+240-y_range)

                cv2.rectangle(dst, (rect_x, rect_y), (rect_x+x_range, rect_y+y_range), 
                            (random.randint(0, 25)*10, random.randint(0, 25)*10, random.randint(0, 25)*10),
                            -1, cv2.LINE_AA)
        dst[y-h:y, x:x+w] = keep_img
        return dst

class geometry_process:
    def shift(self, src, flag, background_flag , r1, r2):
        raw_r1, raw_r2 = r1, r2
        r1, r2 = convert_r1_r2(r1, r2)
        weight = round(random.uniform(0.15, 0.65), 3) #0.15 ~ 0.65 소수점 자리 3째자리 까지
        max_right, max_left = (640-r2[0])*weight, -r1[0]*weight
        max_up, max_down = -r2[1]*weight, (480-r1[1])*weight

        x = max_right if random.randint(0, 1) is 0 else max_left
        y = max_up if random.randint(0, 1) is 0 else max_down
        x_rate = x/640.
        y_rate = y/480.

        modify_r1 , modify_r2 = (raw_r1[0]+x_rate, raw_r1[1]+y_rate), (raw_r2[0]+x_rate, raw_r2[1]+y_rate)
        raw_modify_r1 , raw_modify_r2 = modify_r1 , modify_r2
        modify_r1 , modify_r2 = convert_r1_r2(modify_r1 , modify_r2)
        aff = np.array([[1, 0, x], [0, 1, y]], dtype=np.float32)
        dst = cv2.warpAffine(src, aff, (0, 0))

        if background_flag == True:
            return dst, raw_modify_r1, raw_modify_r2, x, y
            
        return dst, raw_modify_r1, raw_modify_r2

    def symmetry(self, src, flag, background_flag , r1, r2):
        raw_r1, raw_r2 = r1, r2

        # r1, r2 = convert_r1_r2(r1, r2)
        dot2mid = []
        for r in (raw_r1[0], raw_r2[0]):  # r1 r2와 해당 img 중앙과의 거리 계산
            dot2mid.append(0.5000-r)
        # print(dot2mid)
        raw_sym_r1 = (raw_r1[0]+dot2mid[0]*2, raw_r1[1]) # dot2mid값 바탕으로 단순 x축 대칭이동
        raw_sym_r2 = (raw_r2[0]+dot2mid[1]*2, raw_r2[1])

        # print("대칭 이동 r1 r2", raw_sym_r1, raw_sym_r2)

        final_raw_r1 = (raw_sym_r2[0], raw_sym_r1[1])  # label set 동일한 좌표 설정되도록 조정
        final_raw_r2 = (raw_sym_r1[0], raw_sym_r2[1])

        # sym_r1, sym_r2 = convert_r1_r2(final_raw_r1, final_raw_r2)

        dst = cv2.flip(src, 1)
        # cv2.circle(img, r1, 5, (0, 0, 255), -1, cv2.LINE_AA)
        # cv2.circle(img, r2, 5, (0, 0, 255), -1, cv2.LINE_AA)
        # cv2.circle(dst, sym_r1, 5, (0, 0, 0), -1, cv2.LINE_AA)
        # cv2.circle(dst, sym_r2, 5, (255, 0, 0), -1, cv2.LINE_AA)
        # cv2.imshow('dst', dst)

        return dst, final_raw_r1, final_raw_r2

    def rotation(self, src, flag, background_flag, r1, r2):

        r1, r2 = convert_r1_r2(r1, r2)

        block = np.zeros((480, 640), dtype = np.uint8)
        block[r2[1]:r1[1], r1[0]:r2[0]] = 255

        angle = random.randint(-30, 30)
        if angle>=0 & angle <10 :
            angle = random.randint(10, 30)
        elif angle > -10 & angle<0:
            angle = random.randint(-30, -10)

        scale = 0.9 - (abs(angle)/100)

        cp = (src.shape[1] / 2, src.shape[0] / 2)
        rot = cv2.getRotationMatrix2D(cp, angle, scale)

        dst = cv2.warpAffine(src, rot, (0, 0))
        rot_block = cv2.warpAffine(block, rot, (0, 0))

        corners = np.squeeze(cv2.goodFeaturesToTrack(rot_block, 4, 0.1, 10))

        x_min, x_max = int(np.min(corners.T[0]+0.045*640)), int(np.max(corners.T[0]-0.045*640))
        y_min, y_max = int(np.min(corners.T[1]+0.045*480)), int(np.max(corners.T[1]-0.045*480))

        r1 = (x_min, y_max)
        r2 = (x_max, y_min)

        rotated_raw_r1 = (r1[0]/640., r1[1]/480.)
        rotated_raw_r2 = (r2[0]/640., r2[1]/480.)

        m_r1, m_r2 = convert_r1_r2(rotated_raw_r1, rotated_raw_r2)

        # cv2.rectangle(dst, m_r1, m_r2, (255, 255, 0), 2)

        if background_flag == True:
            return dst, rotated_raw_r1, rotated_raw_r2, rot

        return dst, rotated_raw_r1, rotated_raw_r2

# bp = basic_process()
# result  = bp.shadowed(img, True, r1, r2)
# cv2.imshow('img', img)
# cv2.imshow('result', result)

# cv2.waitKey()
# cv2.destroyAllWindows()
# ge = geometry_process()
# dst = ge.rotation(img, True, r1, r2)
# cv2.waitKey()
# cv2.destroyAllWindows()
   
# save_geo_labelimg(img, dir_path, r1, r2)
