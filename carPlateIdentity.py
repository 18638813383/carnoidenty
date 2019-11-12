import cv2
import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from time import *
import subprocess
import threading
import pyttsx3
from queue import Queue
char_table = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '川', '鄂', '赣', '甘', '贵',
              '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '苏', '晋',
              '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']
#语音播报
def voiceplay(voice):
    engine = pyttsx3.init()
    engine.say(voice)
    engine.runAndWait()
def hist_image(img):
    assert img.ndim==2
    hist = [0 for i in range(256)]
    img_h,img_w = img.shape[0],img.shape[1]

    for row in range(img_h):
        for col in range(img_w):
            hist[img[row,col]] += 1
    p = [hist[n]/(img_w*img_h) for n in range(256)]
    p1 = np.cumsum(p)
    for row in range(img_h):
        for col in range(img_w):
            v = img[row,col]
            img[row,col] = p1[v]*255
    return img

def find_board_area(img):
    assert img.ndim==2
    img_h,img_w = img.shape[0],img.shape[1]
    top,bottom,left,right = 0,img_h,0,img_w
    flag = False
    h_proj = [0 for i in range(img_h)]
    v_proj = [0 for i in range(img_w)]

    for row in range(round(img_h*0.5),round(img_h*0.8),3):
        for col in range(img_w):
            if img[row,col]==255:
                h_proj[row] += 1
        if flag==False and h_proj[row]>12:
            flag = True
            top = row
        if flag==True and row>top+8 and h_proj[row]<12:
            bottom = row
            flag = False

    for col in range(round(img_w*0.3),img_w,1):
        for row in range(top,bottom,1):
            if img[row,col]==255:
                v_proj[col] += 1
        if flag==False and (v_proj[col]>10 or v_proj[col]-v_proj[col-1]>5):
            left = col
            break
    return left,top,120,bottom-top-10

def verify_scale(rotate_rect):
   error = 0.4
   aspect = 4#4.7272
   min_area = 10*(10*aspect)
   max_area = 150*(150*aspect)
   min_aspect = aspect*(1-error)
   max_aspect = aspect*(1+error)
   theta = 30

   # 宽或高为0，不满足矩形直接返回False
   if rotate_rect[1][0]==0 or rotate_rect[1][1]==0:
       return False

   r = rotate_rect[1][0]/rotate_rect[1][1]
   r = max(r,1/r)
   area = rotate_rect[1][0]*rotate_rect[1][1]
   if area>min_area and area<max_area and r>min_aspect and r<max_aspect:
       # 矩形的倾斜角度在不超过theta
       if ((rotate_rect[1][0] < rotate_rect[1][1] and rotate_rect[2] >= -90 and rotate_rect[2] < -(90 - theta)) or
               (rotate_rect[1][1] < rotate_rect[1][0] and rotate_rect[2] > -theta and rotate_rect[2] <= 0)):
           return True
   return False

def img_Transform(car_rect,image):
    img_h,img_w = image.shape[:2]
    rect_w,rect_h = car_rect[1][0],car_rect[1][1]
    angle = car_rect[2]

    return_flag = False
    if car_rect[2]==0:
        return_flag = True
    if car_rect[2]==-90 and rect_w<rect_h:
        rect_w, rect_h = rect_h, rect_w
        return_flag = True
    if return_flag:
        car_img = image[int(car_rect[0][1]-rect_h/2):int(car_rect[0][1]+rect_h/2),
                  int(car_rect[0][0]-rect_w/2):int(car_rect[0][0]+rect_w/2)]
        return car_img

    car_rect = (car_rect[0],(rect_w,rect_h),angle)
    box = cv2.boxPoints(car_rect)

    heigth_point = right_point = [0,0]
    left_point = low_point = [car_rect[0][0], car_rect[0][1]]
    for point in box:
        if left_point[0] > point[0]:
            left_point = point
        if low_point[1] > point[1]:
            low_point = point
        if heigth_point[1] < point[1]:
            heigth_point = point
        if right_point[0] < point[0]:
            right_point = point

    if left_point[1] <= right_point[1]:  # 正角度
        new_right_point = [right_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(image, M, (round(img_w*2), round(img_h*2)))
        car_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]

    elif left_point[1] > right_point[1]:  # 负角度
        new_left_point = [left_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(image, M, (round(img_w*2), round(img_h*2)))
        car_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]

    return car_img

def pre_process(orig_img):

    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray_img', gray_img)

    blur_img = cv2.blur(gray_img, (3, 3))
    #cv2.imshow('blur', blur_img)

    sobel_img = cv2.Sobel(blur_img, cv2.CV_16S, 1, 0, ksize=3)
    sobel_img = cv2.convertScaleAbs(sobel_img)
    #cv2.imshow('sobel', sobel_img)

    hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)

    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    # 黄色色调区间[26，34],蓝色色调区间:[100,124]
    blue_img = (((h > 26) & (h < 34)) | ((h > 100) & (h < 124))) & (s > 70) & (v > 70)
    blue_img = blue_img.astype('float32')

    mix_img = np.multiply(sobel_img, blue_img)
    #cv2.imshow('mix', mix_img)

    mix_img = mix_img.astype(np.uint8)

    ret, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imshow('binary',binary_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,5))
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('close', close_img)

    return close_img

# 给候选车牌区域做漫水填充算法，一方面补全上一步求轮廓可能存在轮廓歪曲的问题，
# 另一方面也可以将非车牌区排除掉
def verify_color(rotate_rect,src_image):
    img_h,img_w = src_image.shape[:2]
    mask = np.zeros(shape=[img_h+2,img_w+2],dtype=np.uint8)
    connectivity = 4 #种子点上下左右4邻域与种子颜色值在[loDiff,upDiff]的被涂成new_value，也可设置8邻域
    loDiff,upDiff = 30,30
    new_value = 255
    flags = connectivity
    flags |= cv2.FLOODFILL_FIXED_RANGE  #考虑当前像素与种子象素之间的差，不设置的话则和邻域像素比较
    flags |= new_value << 8
    flags |= cv2.FLOODFILL_MASK_ONLY #设置这个标识符则不会去填充改变原始图像，而是去填充掩模图像（mask）

    rand_seed_num = 5000 #生成多个随机种子
    valid_seed_num = 200 #从rand_seed_num中随机挑选valid_seed_num个有效种子
    adjust_param = 0.1
    box_points = cv2.boxPoints(rotate_rect)
    box_points_x = [n[0] for n in box_points]
    box_points_x.sort(reverse=False)
    adjust_x = int((box_points_x[2]-box_points_x[1])*adjust_param)
    col_range = [box_points_x[1]+adjust_x,box_points_x[2]-adjust_x]
    box_points_y = [n[1] for n in box_points]
    box_points_y.sort(reverse=False)
    adjust_y = int((box_points_y[2]-box_points_y[1])*adjust_param)
    row_range = [box_points_y[1]+adjust_y, box_points_y[2]-adjust_y]
    # 如果以上方法种子点在水平或垂直方向可移动的范围很小，则采用旋转矩阵对角线来设置随机种子点
    if (col_range[1]-col_range[0])/(box_points_x[3]-box_points_x[0])<0.4\
        or (row_range[1]-row_range[0])/(box_points_y[3]-box_points_y[0])<0.4:
        points_row = []
        points_col = []
        for i in range(2):
            pt1,pt2 = box_points[i],box_points[i+2]
            x_adjust,y_adjust = int(adjust_param*(abs(pt1[0]-pt2[0]))),int(adjust_param*(abs(pt1[1]-pt2[1])))
            if (pt1[0] <= pt2[0]):
                pt1[0], pt2[0] = pt1[0] + x_adjust, pt2[0] - x_adjust
            else:
                pt1[0], pt2[0] = pt1[0] - x_adjust, pt2[0] + x_adjust
            if (pt1[1] <= pt2[1]):
                pt1[1], pt2[1] = pt1[1] + adjust_y, pt2[1] - adjust_y
            else:
                pt1[1], pt2[1] = pt1[1] - y_adjust, pt2[1] + y_adjust
            temp_list_x = [int(x) for x in np.linspace(pt1[0],pt2[0],int(rand_seed_num /2))]
            temp_list_y = [int(y) for y in np.linspace(pt1[1],pt2[1],int(rand_seed_num /2))]
            points_col.extend(temp_list_x)
            points_row.extend(temp_list_y)
    else:
        points_row = np.random.randint(row_range[0],row_range[1],size=rand_seed_num)
        points_col = np.linspace(col_range[0],col_range[1],num=rand_seed_num).astype(np.int)

    points_row = np.array(points_row)
    points_col = np.array(points_col)
    hsv_img = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    h,s,v = hsv_img[:,:,0],hsv_img[:,:,1],hsv_img[:,:,2]
    # 将随机生成的多个种子依次做漫水填充,理想情况是整个车牌被填充
    flood_img = src_image.copy()
    seed_cnt = 0
    for i in range(rand_seed_num):
        rand_index = np.random.choice(rand_seed_num,1,replace=False)
        row,col = points_row[rand_index],points_col[rand_index]
        # 限制随机种子必须是车牌背景色
        try:
            if (((h[row,col]>26)&(h[row,col]<34))|((h[row,col]>100)&(h[row,col]<124)))&(s[row,col]>70)&(v[row,col]>70):
                cv2.floodFill(src_image, mask, (col,row), (255, 255, 255), (loDiff,) * 3, (upDiff,) * 3, flags)
                cv2.circle(flood_img,center=(col,row),radius=2,color=(0,0,255),thickness=2)
                seed_cnt += 1
                if seed_cnt >= valid_seed_num:
                    break
        except:
            print('Error 241行报错请检查')
    #======================调试用======================#
  #  show_seed = np.random.uniform(1,100,1).astype(np.uint16)
  #  cv2.imshow('floodfill'+str(show_seed),flood_img)
  # # cv2.imshow('flood_mask'+str(show_seed),mask)
  #  #======================调试用======================#
    # 获取掩模上被填充点的像素点，并求点集的最小外接矩形
    mask_points = []
    for row in range(1,img_h+1):
        for col in range(1,img_w+1):
            if mask[row,col] != 0:
                mask_points.append((col-1,row-1))
    mask_rotateRect = cv2.minAreaRect(np.array(mask_points))
    if verify_scale(mask_rotateRect):
        return True,mask_rotateRect
    else:
        return False,mask_rotateRect

# 车牌定位,返回定位到的车牌矩形照片
def locate_carPlate(orig_img,pred_image):
    carPlate_list = []
    temp1_orig_img = orig_img.copy() #调试用
    temp2_orig_img = orig_img.copy() #调试用
    #print(pred_image.shape)
    #cloneImg,contours,heriachy = cv2.findContours(pred_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours,heriachy = cv2.findContours(pred_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for i,contour in enumerate(contours):
        cv2.drawContours(temp1_orig_img, contours, i, (0, 255, 255), 2)
        # 获取轮廓最小外接矩形，返回值rotate_rect
        rotate_rect = cv2.minAreaRect(contour)
        # 根据矩形面积大小和长宽比判断是否是车牌
        if verify_scale(rotate_rect):
            ret,rotate_rect2 = verify_color(rotate_rect,temp2_orig_img)
            if ret == False:
                continue
            # 车牌位置矫正
            car_plate = img_Transform(rotate_rect2, temp2_orig_img)
            if car_plate.shape[0] != 0 and car_plate.shape[1] !=0 :
                car_plate = cv2.resize(car_plate,(car_plate_w,car_plate_h)) #调整尺寸为后面CNN车牌识别做准备
            #========================调试看效果========================#
            box = cv2.boxPoints(rotate_rect2)
            for k in range(4):
                n1,n2 = k%4,(k+1)%4
                cv2.line(temp1_orig_img,(box[n1][0],box[n1][1]),(box[n2][0],box[n2][1]),(0,0,255),2)
 #           cv2.imshow('opencv_' + str(i), car_plate)
            #========================调试看效果========================#
            carPlate_list.append(car_plate)
  #  cv2.imshow('aaa',temp1_orig_img)
 #   k = cv2.waitKey(0)
    global type
    if type == 0:
    #cv2.imshow('orignal', car_plate)
        cv2.imwrite('locate.png',car_plate)
        global locatfile
        locatfile = PhotoImage(file='locate.png')
        carnum.create_image(100,25,image=locatfile)
    return carPlate_list,temp1_orig_img

# 左右切割,获取切割后的字符图像列表
def horizontal_cut_chars(plate):
    char_addr_list = []
    area_left,area_right,char_left,char_right= 0,0,0,0
    img_w = plate.shape[1]

    # 获取车牌每列边缘像素点个数
    def getColSum(img,col):
        sum = 0
        for i in range(img.shape[0]):
            sum += round(img[i,col]/255)
        return sum;

    sum = 0
    for col in range(img_w):
        sum += getColSum(plate,col)
    # 每列边缘像素点必须超过均值的60%才能判断属于字符区域
    col_limit = 0#round(0.5*sum/img_w)
    # 每个字符宽度也进行限制
    charWid_limit = [round(img_w/12),round(img_w/5)]
    is_char_flag = False

    for i in range(img_w):
        colValue = getColSum(plate,i)
        if colValue > col_limit:
            if is_char_flag == False:
                area_right = round((i+char_right)/2)
                area_width = area_right-area_left
                char_width = char_right-char_left
                if (area_width>charWid_limit[0]) and (area_width<charWid_limit[1]):
                    char_addr_list.append((area_left,area_right,char_width))
                char_left = i
                area_left = round((char_left+char_right) / 2)
                is_char_flag = True
        else:
            if is_char_flag == True:
                char_right = i-1
                is_char_flag = False
    # 手动结束最后未完成的字符分割
    if area_right < char_left:
        area_right,char_right = img_w,img_w
        area_width = area_right - area_left
        char_width = char_right - char_left
        if (area_width > charWid_limit[0]) and (area_width < charWid_limit[1]):
            char_addr_list.append((area_left, area_right, char_width))
    return char_addr_list
def get_chars(car_plate):
    img_h,img_w = car_plate.shape[:2]
    h_proj_list = [] # 水平投影长度列表
    h_temp_len,v_temp_len = 0,0
    h_startIndex,h_end_index = 0,0 # 水平投影记索引
    h_proj_limit = [0.2,0.8] # 车牌在水平方向得轮廓长度少于20%或多余80%过滤掉
    char_imgs = []

    # 将二值化的车牌水平投影到Y轴，计算投影后的连续长度，连续投影长度可能不止一段
    h_count = [0 for i in range(img_h)]
    for row in range(img_h):
        temp_cnt = 0
        for col in range(img_w):
            if car_plate[row,col] == 255:
                temp_cnt += 1
        h_count[row] = temp_cnt
        if temp_cnt/img_w<h_proj_limit[0] or temp_cnt/img_w>h_proj_limit[1]:
            if h_temp_len != 0:
                h_end_index = row-1
                h_proj_list.append((h_startIndex,h_end_index))
                h_temp_len = 0
            continue
        if temp_cnt > 0:
            if h_temp_len == 0:
                h_startIndex = row
                h_temp_len = 1
            else:
                h_temp_len += 1
        else:
            if h_temp_len > 0:
                h_end_index = row-1
                h_proj_list.append((h_startIndex,h_end_index))
                h_temp_len = 0

    # 手动结束最后得水平投影长度累加
    if h_temp_len != 0:
        h_end_index = img_h-1
        h_proj_list.append((h_startIndex, h_end_index))
    # 选出最长的投影，该投影长度占整个截取车牌高度的比值必须大于0.5
    h_maxIndex,h_maxHeight = 0,0
    for i,(start,end) in enumerate(h_proj_list):
        if h_maxHeight < (end-start):
            h_maxHeight = (end-start)
            h_maxIndex = i
    if h_maxHeight/img_h < 0.5:
        return char_imgs
    chars_top,chars_bottom = h_proj_list[h_maxIndex][0],h_proj_list[h_maxIndex][1]

    plates = car_plate[chars_top:chars_bottom+1,:]
    cv2.imwrite('../images/opencv_output/test/car.jpg',car_plate)
    cv2.imwrite('../images/opencv_output/test/plate.jpg', plates)
    char_addr_list = horizontal_cut_chars(plates)
    for i,addr in enumerate(char_addr_list):
        char_img = car_plate[chars_top:chars_bottom+1,addr[0]:addr[1]]
        char_img = cv2.resize(char_img,(char_w,char_h))
        char_imgs.append(char_img)
    #cv2.imshow('char', char_imgs[1])
    try:
        cv2.imwrite('../images/opencv_output/test/char1.jpg', char_imgs[0])
        cv2.imwrite('../images/opencv_output/test/char2.jpg', char_imgs[1])
        cv2.imwrite('../images/opencv_output/test/char3.jpg', char_imgs[2])
        cv2.imwrite('../images/opencv_output/test/char4.jpg', char_imgs[3])
        cv2.imwrite('../images/opencv_output/test/char5.jpg', char_imgs[4])
        cv2.imwrite('../images/opencv_output/test/char6.jpg', char_imgs[5])
        cv2.imwrite('../images/opencv_output/test/char7.jpg', char_imgs[6])
    except:
        print('ERROR:保存字符图片失败')
    return char_imgs
#提取字符图像列表
def extract_char(car_plate):
    gray_plate = cv2.cvtColor(car_plate,cv2.COLOR_BGR2GRAY)
    ret,binary_plate = cv2.threshold(gray_plate,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    char_img_list = get_chars(binary_plate)
    #cv2.imshow('char',char_img_list[0])
    return char_img_list

def cnn_select_carPlate(plate_list,model_path):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if len(plate_list) == 0:
        print('-----------------未定位到疑似车牌，请检查相关参数设置-----------------')
        return False,plate_list
    g1 = tf.Graph()
    sess1 = tf.compat.v1.Session(graph=g1)
    with sess1.as_default():
        with sess1.graph.as_default():
            model_dir = os.path.dirname(model_path)
            saver = tf.compat.v1.train.import_meta_graph(model_path)
            saver.restore(sess1, tf.train.latest_checkpoint(model_dir))
            graph = tf.compat.v1.get_default_graph()
            net1_x_place = graph.get_tensor_by_name('x_place:0')
            net1_keep_place = graph.get_tensor_by_name('keep_place:0')
            net1_out = graph.get_tensor_by_name('out_put:0')

            input_x = np.array(plate_list)
            net_outs = tf.nn.softmax(net1_out)
            preds = tf.argmax(net_outs,1) #预测结果
            probs = tf.reduce_max(net_outs,reduction_indices=[1]) #结果概率值
            pred_list,prob_list = sess1.run([preds,probs],feed_dict={net1_x_place:input_x,net1_keep_place:1.0})
            # 选出概率最大的车牌
            result_index,result_prob = -1,0.
            for i,pred in enumerate(pred_list):
                if pred==1 and prob_list[i]>result_prob:
                    result_index,result_prob = i,prob_list[i]
            if result_index == -1:
                return False,plate_list[0]
            else:
                return True,plate_list[result_index]
#字符识别
def cnn_recongnize_char(img_list,model_path):
    g2 = tf.Graph()
    sess2 = tf.compat.v1.Session(graph=g2)
    text_list = []

    #print('img_list is:' ,img_list[0])#img_list为裁剪后的字符，img_list[0]代表第一个字符
    if len(img_list) == 0:
        return text_list
    with sess2.as_default():
        with sess2.graph.as_default():
            model_dir = os.path.dirname(model_path)
            saver = tf.compat.v1.train.import_meta_graph(model_path)
            saver.restore(sess2, tf.train.latest_checkpoint(model_dir))
            graph = tf.compat.v1.get_default_graph()
            net2_x_place = graph.get_tensor_by_name('x_place:0')
            net2_keep_place = graph.get_tensor_by_name('keep_place:0')
            net2_out = graph.get_tensor_by_name('out_put:0')

            data = np.array(img_list)
            # 数字、字母、汉字，从67维向量找到概率最大的作为预测结果
            net_out = tf.nn.softmax(net2_out) #样本属于各个类的概率
            preds = tf.argmax(net_out,1)#根据axis取值的不同返回每行或者每列最大值的索引
            my_preds= sess2.run(preds, feed_dict={net2_x_place: data, net2_keep_place: 1.0})
            #print('*****************',net_out)
            #print('*********preds**********',preds)
            #print('*****************my_preds****',my_preds)
            for i in my_preds:
                text_list.append(char_table[i])
            return text_list
#图像车牌号识别
def carnum_recognizefromimg(img):  #车牌号识别，根据给定的照片识别出车牌号
    img = cv2.imread(img)
    # 预处理
    print(strftime("%Y-%m-%d %H:%M:%S"),'******************开始图片预处理*******************')
    pred_img = pre_process(img)
    print(strftime("%Y-%m-%d %H:%M:%S"),'******************图片预处理结束，开始车牌定位*******************')
    # 车牌定位
    car_plate_list,temp1_orig_img = locate_carPlate(img,pred_img)
    # CNN车牌过滤，利用神经网络确认是否是车牌
    print(strftime("%Y-%m-%d %H:%M:%S"),'******************车牌定位结束，开始车牌过滤*******************')
    ret,car_plate = cnn_select_carPlate(car_plate_list,plate_model_path)
    if ret == False:
        print("未检测到车牌,请更换一张图片")
        #sys.exit(-1)
    else:
        print(strftime("%Y-%m-%d %H:%M:%S"),'******************车牌过滤结束，开始提取字符*******************')
        char_img_list = extract_char(car_plate)
        # CNN字符识别
        print(strftime("%Y-%m-%d %H:%M:%S"),'******************提取字符结束，字符识别*******************')
        text = cnn_recongnize_char(char_img_list,char_model_path)
        print(strftime("%Y-%m-%d %H:%M:%S"),'******************字符识别结束，输出字符结果*******************')
        # print(text)
        #print('%s %s' % (''.join(text[:2]), ''.join(text[2:])))
        resultnum = ''.join(text[:2]) +' ' + ''.join(text[2:])
        global result
        result.set(resultnum)
        print(resultnum)
#视频切割
def videocut(inputfilename):
    if os.path.exists('moviecutresult'):
        shutil.rmtree('moviecutresult')
        sleep(5)
        os.mkdir('.\\moviecutresult')
    else:
        os.mkdir('.\\moviecutresult')
    timespace = 1
    print('--------------开始切割视频，按照每',timespace ,'秒一个文件进行切割--------------')
    movie = cv2.VideoCapture(inputfilename)
    # 视频时间
    time = movie.get(7) / movie.get(5)
    starttime = 0
    endtime = time
    outnum = 1
    while (1):
        if starttime <= endtime:
            subprocess.call('ffmpeg -i ' + inputfilename + ' -ss ' + str(starttime)
                            + ' -t ' +  str(timespace) +' -codec copy .\\moviecutresult\\'
                            + str(outnum) + '_cut.mp4', shell=True)
            starttime += timespace
            outnum += 1
            print('开始时间:', starttime)
        else:
            #subprocess.call('ffmpeg -i '  + inputfilename + ' -ss ' + str(starttime)
             #               + ' -t ' + str(timespace) + ' -codec copy .\\moviecutresult\\'
             #               + str(outnum) + '_cut.mp4', shell=True)
            print('------------已切割至最后-----------')
            break
    print('-----------------视频切割结束!等待5s后开始分析文件-------------')
    sleep(5)
def recognizefromvideothread():
    tasks = []
    print('-----------多线程分析视频即将开始-----------------')
    for inputfilename in os.listdir('moviecutresult'):
        #outfilename = 'result_' + inputfilename + '.avi'
        t = threading.Thread(target=carnum_recognizefromvideo, args=['.\\moviecutresult\\' + inputfilename])
        t.setDaemon(True)
        tasks.append(t)
        t.start()
    for task in tasks: task.join()
    print('*********线程 %s 已结束.***********' % threading.current_thread().name)
#视频车牌识别
def carnum_recognizefromvideo(video):
    print('*************开始从视频中进行车牌号识别***********')
    cam = cv2.VideoCapture(video)
    print('*******开始分析的文件是：',video)
    filename = '.' + video.split('.')[1] + '.txt'
    print(filename)
    file = open(filename, mode='a', encoding='UTF-8')
    while(True):
        ret,img = cam.read()
        if ret :
            try:
                print(strftime("%Y-%m-%d %H:%M:%S"), '******************开始图片预处理*******************')
                pred_img = pre_process(img)
                print('*******************图片预处理结束************************************')
                car_plate_list,temp1_orig_img = locate_carPlate(img, pred_img)
                #cv2.imshow('video',temp1_orig_img)
                #k = cv2.waitKey(1)
                #if k == 27:
                #    break
            # CNN车牌过滤，利用神经网络确认是否是车牌
                print(strftime("%Y-%m-%d %H:%M:%S"), '******************车牌定位结束，开始车牌过滤*******************')
                plate, car_plate = cnn_select_carPlate(car_plate_list, plate_model_path)
                if plate == False:
                    print("未检测到车牌,请更换一张图片")
                    # sys.exit(-1)
                else:
                    print(strftime("%Y-%m-%d %H:%M:%S"), '******************车牌过滤结束，开始提取字符*******************')
                    char_img_list = extract_char(car_plate)
                # CNN字符识别
                    print(strftime("%Y-%m-%d %H:%M:%S"), '******************提取字符结束，字符识别*******************')
                    text = cnn_recongnize_char(char_img_list, char_model_path)
                    print(strftime("%Y-%m-%d %H:%M:%S"), '******************字符识别结束，输出字符结果*******************')
                # print(text)
                # print('%s %s' % (''.join(text[:2]), ''.join(text[2:])))
                    resultnum = ''.join(text[:2]) + ' ' + ''.join(text[2:])
                    print(strftime("%Y-%m-%d %H:%M:%S"), '最后的识别结果是：', resultnum)
                    file.write(strftime("%Y-%m-%d %H:%M:%S") + '\t' + resultnum + '\t' + video + '\n')
            except:
                print(strftime("%Y-%m-%d %H:%M:%S"),'ERROR: 识别失败请重试')
        else:
            print(strftime("%Y-%m-%d %H:%M:%S"),'视频打开失败！！！！！！！')
            break
    cam.release()
    cv2.destroyAllWindows()
    print(strftime("%Y-%m-%d %H:%M:%S"),'***********处理完成，请关闭视频**************')
    #file.close()
#启动车牌识别线程
recognize_queue = Queue(1) #最多只可启动一个线程
def startthread(img):
    print('-----------启动车牌号识别线程-----------------')
    if recognize_queue.full():
        print('--------------线程数已满退出线程-------------')
        return
    t = threading.Thread(target=recognizethread, args=[img])
    t.setDaemon(True)
    recognize_queue.put(t)
    try:
        t.start()
    except:
        print('Error 614行，线程启动失败')
        recognize_queue.get()
    print('thread %s ended. ' % t.name)
def recognizethread(img):
    print(strftime("%Y-%m-%d %H:%M:%S"),'--------车牌号识别线程已启动--------------')
    print(strftime("%Y-%m-%d %H:%M:%S"), '******************开始图片预处理*******************')
    pred_img = pre_process(img)
    # CNN车牌过滤，利用神经网络确认是否是车牌
    car_plate_list, temp1_orig_img = locate_carPlate(img, pred_img)
    print(strftime("%Y-%m-%d %H:%M:%S"), '******************车牌定位结束，开始车牌过滤*******************')
    try:
        plate, car_plate = cnn_select_carPlate(car_plate_list, plate_model_path)
    except:
        print('车牌定位异常，退出线程')
        recognize_queue.get()
        return
    if plate == False:
        print("未检测到车牌,请更换一张图片")
        recognize_queue.get()
        return
    else:
        file = open('capture_result.txt', mode='a', encoding='UTF-8')
        print(strftime("%Y-%m-%d %H:%M:%S"), '******************车牌过滤结束，开始提取字符*******************')
        char_img_list = extract_char(car_plate)
        # CNN字符识别
        print(strftime("%Y-%m-%d %H:%M:%S"), '******************提取字符结束，字符识别*******************')
        text = cnn_recongnize_char(char_img_list, char_model_path)
        print(strftime("%Y-%m-%d %H:%M:%S"), '******************字符识别结束，输出字符结果*******************')
        resultnum = ''.join(text[:2]) + ' ' + ''.join(text[2:])
        if len(resultnum) == 8:
            voiceplay('成功识别到：')
            voiceplay(resultnum)
            file.write(strftime("%Y-%m-%d %H:%M:%S") + '\t' + resultnum + '\n')
        file.close()
        recognize_queue.get()
        print(strftime("%Y-%m-%d %H:%M:%S"), '最后的识别结果是：', resultnum)



#摄像头识别
def carnum_recognizefromcapture(capturenum):
    print('*************开始从摄像头中进行车牌号识别***********')
    cam = cv2.VideoCapture(capturenum)
    while(True):
        ret,img = cam.read()
        if ret :
            cv2.imshow('capture',img)
            k = cv2.waitKey(25)
            if k == 27:
                break
            try:
                startthread(img)
            except:
                print('-------线程启动失败-------------')
        else:
            print(strftime("%Y-%m-%d %H:%M:%S"),'摄像头打开失败！！！！！！！')
            break
    cam.release()
    cv2.destroyAllWindows()
    print(strftime("%Y-%m-%d %H:%M:%S"),'***********处理完成，请关闭摄像头**************')
#合并txt文件
def combinetxt():
    print(strftime("%Y-%m-%d %H:%M:%S"), '******************开始txt文件合并*******************')
    filedir = './moviecutresult/'
    filelist = os.listdir(filedir)
    textfile = []
    for i in filelist:
        if i.split('.')[1] == 'txt':
            textfile.append(i)
    content = []
    for i in textfile:
        with open(filedir + i ,'rb') as f:
            content = content + f.readlines()
    with open('combile_video_result.txt','ab') as f:
        f.writelines(content)
    strftime("%Y-%m-%d %H:%M:%S"), '******************文件合并结束*******************'
#图片打开
def open_img():
    global imgname
    imgname = filedialog.askopenfilename(title='打开单个文件',
                                     filetypes=[('jpg文件','*.jpg' ),('png文件',"*.png"),  ('所有文件','*')],  # 只处理的文件类型
                                     initialdir='.') # 初始目录
    print('你选择的文件名称是：',imgname)
    tmp = cv2.imread(imgname)
    tmp = cv2.resize(tmp,(600,600))
    cv2.imwrite('tmp.png',tmp)
    global imagefile
    imagefile = PhotoImage(file='tmp.png')
    global type
    type  = 0
    canvas.create_image(300,300,image=imagefile)
#视频打开
def open_video():
    global videoname
    videoname = filedialog.askopenfilename(title='打开单个文件',
                                     filetypes=[('mp4文件','*.mp4' ),  ('所有文件','*')],  # 只处理的文件类型
                                     initialdir='.') # 初始目录
    print('你选择的文件名称是：',videoname)
    global type
    type  = 1
#摄像头打开
def open_capture():
    print('****************您已选择摄像头识别，请确认！！！****************')
    global type
    type =2
#车牌识别主函数
def carnum_recognize(type):
    print(strftime("%Y-%m-%d %H:%M:%S"),'-------------------车牌号识别开始---------------------')
    print(strftime("%Y-%m-%d %H:%M:%S"),'-------------------开始参数检测------------------识别类型是：',type)
    if type == 0:
        print(strftime("%Y-%m-%d %H:%M:%S"),'识别类型为图片，请确认！！！！')
        carnum_recognizefromimg(imgname)
    elif type == 1:
        print(strftime("%Y-%m-%d %H:%M:%S"),'识别类型为视频，请确认！！！！')
        videocut(videoname)
        recognizefromvideothread()
        combinetxt()
    elif type == 2:
        print(strftime("%Y-%m-%d %H:%M:%S"),'识别类型为摄像头，请确认！！！！')
        carnum_recognizefromcapture(capturenum)
    else:
        print(strftime("%Y-%m-%d %H:%M:%S"),'Error:您尚未选择识别类型，请选择！！！！！')

imgname = ''#打开图片的全局变量
videoname = ''#打开视频全局变量
type = ''
capturenum = 0
cur_dir = sys.path[0]
car_plate_w,car_plate_h = 136,36
char_w,char_h = 20,20
plate_model_path = os.path.join(cur_dir,'..\\model\\plate_recongnize\\trainmodel.ckpt-520.meta')
char_model_path = os.path.join(cur_dir,'..\\model\\char_recongnize\\trainmodel.ckpt-850.meta')
imgpath = os.path.join(cur_dir,'..\\images\\pictures\\')
#img = cv2.imread('..\\images\\pictures\\2.jpg')
root = Tk()
root.title('车牌号识别')
frame = ttk.Frame(root)
frame.pack(side=TOP)
imglabelframe = ttk.LabelFrame(frame,text='图片区域')
imglabelframe.pack(side=LEFT,padx=10,pady=10)
canvas = Canvas(imglabelframe,background='white',width=600,height=600)
canvas.pack(side=TOP,padx=10,pady=10)
oplableframe = ttk.LabelFrame(frame,text='结果区域')
oplableframe.pack(side=TOP,padx=10,pady=10)
resultlabelframe = ttk.LabelFrame(oplableframe,text='识别照片')
resultlabelframe.pack(side=TOP,padx=10,pady=10)
ttk.Label(resultlabelframe,text='车牌号：').pack(side=LEFT,padx=10,pady=10)
carnum = Canvas(resultlabelframe,background='white',width=200,height=50)
carnum.pack(side=LEFT,padx=10,pady=10)
resultlabelframe = ttk.LabelFrame(oplableframe,text='识别结果')
resultlabelframe.pack(side=TOP,padx=10,pady=10)
ttk.Label(resultlabelframe,text='车牌号：').pack(side=LEFT,padx=50,pady=10)
result = StringVar()
result.set('豫A 12345')
ttk.Entry(resultlabelframe,textvariable=result).pack(side=LEFT,padx=10,pady=10)
buttonlabelframe = ttk.LabelFrame(oplableframe,text='功能')
buttonlabelframe.pack(side=TOP,padx=10,pady=10)
ttk.Button(buttonlabelframe,text='摄像头',command=open_capture).pack(side=TOP,padx=100,pady=10)
ttk.Button(buttonlabelframe,text='图片',command=open_img).pack(side=TOP,padx=100,pady=10)
ttk.Button(buttonlabelframe,text='视频',command=open_video).pack(side=TOP,padx=100,pady=10)
ttk.Button(buttonlabelframe,text='识别',command= lambda :carnum_recognize(type)).pack(side=TOP,padx=100,pady=10)
root.mainloop()
