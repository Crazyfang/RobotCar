# encoding:utf-8

import cv2
import os
import numpy as np
import time
import logging
from nn.good_classifier import GoodClassifier
import math
from multiprocessing import Manager, Process, Lock, Pipe, Queue

class RobotCarHandle():
    """A  robotcar classification.

    Args:
        None.

    """

    def __init__(self, filepath=None, gaosi_sign=0):
        self.queue_dispose = Queue()

        self.queue_write = Queue()

        # 拍摄前延时时间
        self.taking_photo_time = 0.25

        # 每次拍摄次数
        self.taking_photo_times = 2

        # 图像存储路径
        if filepath:
            self.photo_path = './recentphoto/{}/'.format(filepath)
        else:
            self.photo_path = './picture/'


        self.end = 7
        print(self.end)

        self.gaosi_sign = gaosi_sign
        if self.gaosi_sign:
            print('高斯模式开启')
        else:
            print('原图模式开启')

        # 黄色小木块初始颜色阈值
        self.lower_yellow = np.array([15, 150, 80])
        self.upper_yellow = np.array([50, 255, 255])

        # 颜色调整值
        self.color_plus_adjust_value = 5
        self.color_sub_adjust_value = 3

        # 当次识别出小黄块列表
        self.new_color_materials = []

        # 需抓取小黄块列表
        self.color_materials = []

        # 图片切割信息
        self.verticalline = [500, 2100]
        self.transverseline = [200, 1000, 1944]

        # NeuralNetwork
        # self.Nn = NeuralNetwork()
        # 切图保存路径
        if not filepath:
            self.cut_pic_path = './picture/Cut/'
        else:
            self.cut_pic_path = './recentphoto/{}/Cut'.format(filepath)

        # 图片预处理列表 [识别度, 物品名称, 照片序号， 上下编号{上-0|下-1}, 验证结果{True|False}]
        self.pre_items = []

        # 已处理图片数量(Max to 36)
        self.process_number = 0

        # B,C,D区物品列表
        self.second_data = []

        # xml模板所在文件夹
        self.xml_Path = './cascade/'

        # 查询列表
        # self.manager = Manager()
        # self.Ready_To_Search = self.manager.list()
        # 查询进程结束标志
        self.isfinsh = True

        # 临时文件锁
        self.temp_lock = Lock()

        # 临时文件
        self.tempfile = './files/data.tmp'

        # 结果文件
        self.resultfile = './files/data.res'

        # 完成文件
        self.finishfile = './files/Finish'

        if not os.path.exists('./files'):
            os.mkdir('./files')
        else:
            pass

        if not os.path.exists(self.photo_path):
            os.mkdir(self.photo_path)
        else:
            pass

        if not os.path.exists(self.cut_pic_path):
            os.mkdir(self.cut_pic_path)
        else:
            pass

        self.data_pipe = Pipe(True)
        self.result_pipe = Pipe(True)

    # 开启摄像头
    def open_camera(self):
        self.cap = cv2.VideoCapture(1)

        #设置图像尺寸
        self.cap.set(3, 2592)
        self.cap.set(4, 1944)
        
        # 打开摄像头
        if not self.cap.isOpened():
            self.cap.open()

        # 预拍照
        ret, img = self.cap.read()
        cv2.imwrite(self.photo_path + 'temp.png', img)

        # 显示摄像头打开成功信息
        if self.cap.isOpened():
            print('Open Camera Successful')

    # 摄像头拍照
    def camera_takephoto(self, order_number):
        # 显示正在拍照信息
        print('Taking photo')

        # 拍摄并保存图像
        for i in range(0, self.taking_photo_times):
            ret, img = self.cap.read()
            time.sleep(self.taking_photo_time)
            cv2.imwrite(self.photo_path + '%s.png' % order_number, img)

    # 关闭摄像头
    def close_camera(self):
        print('关闭摄像头')
        self.cap.release()

    # A区图片动态阈值处理
    def image_handle_dynamic_change(self):
        try:
            # 最小过剩剩余方块列表
            Maxmin_new_color_materials = []
            # 修正次数
            Count = 0

            while len(self.color_materials) < 6 and Count < 15:
                # 清空当次小黄列表
                self.new_color_materials = []
                
                # 识别A区图片
                for i in range(1, 7):
                    str = self.photo_path + '%d.png' % i
                    self.findcolorfools(str, i)

                # 判断当前识别列表
                if len(self.color_materials) + len(self.new_color_materials) <= 6:
                    for i in self.new_color_materials:
                        self.color_materials.append(i)
                    if len(self.color_materials) < 6:
                        # 颜色阈值扩大
                        adjust_value = self.color_plus_adjust_value
                        self.lower_yellow = np.array([15, self.lower_yellow[1] - adjust_value, self.lower_yellow[2] - adjust_value])
                        self.upper_yellow = np.array([40, self.upper_yellow[1] + adjust_value, self.upper_yellow[1] + adjust_value])
                else:
                    # 颜色阈值缩小
                    adjust_value = self.color_sub_adjust_value
                    self.lower_yellow = np.array([15, self.lower_yellow[1] + adjust_value, self.lower_yellow[2] + adjust_value])
                    self.upper_yellow = np.array([40, self.upper_yellow[1] - adjust_value, self.upper_yellow[1] - adjust_value])

                    # 比较并保存剩余方块列表(取最小量)
                    if len(Maxmin_new_color_materials) == 0 or self.new_color_materials < Maxmin_new_color_materials:
                        if len(self.new_color_materials) != 0:
                            Maxmin_new_color_materials = self.new_color_materials[:]

                # 识别次数增加1
                Count += 1

                # 输出信息
                print('-------Loop %d Result-------' % Count)
                print('Yellow Cube List: '),
                print(self.color_materials)
                print('Yellow Color Threshold: '),
                print(self.lower_yellow),
                print(self.upper_yellow)

            # 增加最小剩余方块列表
            if len(self.color_materials) < 6:
                for Item in Maxmin_new_color_materials:
                    if Item not in self.color_materials:
                        self.color_materials.append(Item)

            # 对抓取列表进行排序
            self.color_materials = sorted(self.color_materials)
            print('----------Yellow Cube Result----------')
            print(self.color_materials)
        except Exception as e:
            print(e)

    # 颜色识别动态阈值模块
    def findcolorfools(self, img_src, picture_sign):
        # 读取图片
        img_all = cv2.imread(img_src)
        try:
            hsvs = cv2.cvtColor(img_all, cv2.COLOR_BGR2HSV)

            # 上下货栏
            for i in range(2):
                img = img_all.copy()
                if i == 0: #上栏
                    #判断是否已经在抓取列表中
                    for Item in self.color_materials:
                        if Item[1] == 2 * picture_sign - 1:
                            return
                    # 切图并颜色识别
                    crop_img = img[int(self.transverseline[0]):int(self.transverseline[1]),int(self.verticalline[0]):int(self.verticalline[1])]
                    hsv = hsvs[int(self.transverseline[0]):int(self.transverseline[1]),int(self.verticalline[0]):int(self.verticalline[1])]
                    mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
                    imgage, contours, hi = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    if contours:
                        a = contours[0]
                        x, y, w, h = cv2.boundingRect(a)
                        for cnt in contours:
                            x, y, w, h = cv2.boundingRect(cnt)
                            if w + h > 100:
                                self.new_color_materials.append(['A', 2 * picture_sign - 1 + i, "yellow cube"])
                                break
                else: #下栏
                    #判断是否已经在抓取列表中
                    for Item in self.color_materials:
                        if Item[1] == 2 * picture_sign:
                            return
                    # 切图并颜色识别
                    crop_img = img[int(self.transverseline[1]):int(self.transverseline[2]),int(self.verticalline[0]):int(self.verticalline[1])]
                    hsv = hsvs[int(self.transverseline[1]):int(self.transverseline[2]),int(self.verticalline[0]):int(self.verticalline[1])]
                    mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
                    imgage, contours, hi = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    if contours:
                        a = contours[0]
                        x, y, w, h = cv2.boundingRect(a)
                        for cnt in contours:
                            x, y, w, h = cv2.boundingRect(cnt)
                            if w + h > 100:
                                self.new_color_materials.append(['A', 2 * picture_sign - 1 + i, "yellow cube"])
                                break
        except Exception as e:
            print(e)

    # A区图片处理 - 固定阈值
    def image_handle_fixed_value(self):
        try:
            # 1-6张图为黄色物块
            for picture_sign in range(1, 7):
                str = self.photo_path + '%d.png' % picture_sign
                img_all = cv2.imread(str)
                hsvs = cv2.cvtColor(img_all, cv2.COLOR_BGR2HSV)

                # adjust = 0

                # 识别上半部分
                img = img_all.copy()
                crop_img = img[int(self.transverseline[0]):int(self.transverseline[1]),
                           int(self.verticalline[0]):int(self.verticalline[1])]
                hsv = hsvs[int(self.transverseline[0]):int(self.transverseline[1]),
                      int(self.verticalline[0]):int(self.verticalline[1])]
                mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
                imgage, contours, hi = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                if contours:
                    a = contours[0]
                    x, y, w, h = cv2.boundingRect(a)
                    for cnt in contours:
                        # contour_area_temp = np.fabs(cv2.contourArea(cnt))
                        x, y, w, h = cv2.boundingRect(cnt)
                        if w + h > 300:
                            # if w+h > adjust:
                            #     adjust = w + h
                            # cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            self.color_materials.append(["A", 2 * picture_sign - 1, "yellow cube"])
                            # print(w + h)
                            break
                # cv2.imshow('123', crop_img)
                # cv2.waitKey(0)
                # print(adjust)

                # adjust = 0
                # 识别下半部分
                img = img_all.copy()
                crop_img = img[int(self.transverseline[1]):int(self.transverseline[2]),
                           int(self.verticalline[0]):int(self.verticalline[1])]
                hsv = hsvs[int(self.transverseline[1]):int(self.transverseline[2]),
                      int(self.verticalline[0]):int(self.verticalline[1])]
                mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
                imgage, contours, hi = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                if contours:
                    a = contours[0]
                    x, y, w, h = cv2.boundingRect(a)
                    for cnt in contours:
                        # contour_area_temp = np.fabs(cv2.contourArea(cnt))
                        x, y, w, h = cv2.boundingRect(cnt)
                        if w + h > 300:
                        #     if w+h > adjust:
                        #         adjust = w + h
                        #     cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                            self.color_materials.append(["A", 2 * picture_sign, "yellow cube"])
                            # print(w + h)
                            break
                # cv2.imshow('123', crop_img)
                # cv2.waitKey(0)
        except Exception as e:
            print(e)

    # 返回A区结果
    def return_first_result(self):
        print(sorted(self.color_materials, reverse=True))
        return sorted(self.color_materials,reverse=True)

    # B,C,D区图片处理
    def second_process_queue(self, order_number):

        print('\033[0;37;40m')

        # 启动进程
        if self.isfinsh:
            if os.path.exists(self.tempfile):
                os.remove(self.tempfile)
            if os.path.exists(self.resultfile):
                os.remove(self.resultfile)
            if os.path.exists(self.finishfile):
                os.remove(self.finishfile)
            self.isfinsh = False
            p = Process(target=self.start_search_multiprocess, args=(self.queue_dispose, self.queue_write))
            p.daemon = True
            p.start()

        if order_number != 0:
            path = self.photo_path + '%d.png' % order_number
            all_img = cv2.imread(path)

            # 保存上半张图片
            img = all_img.copy()
            crop_img = img[int(self.transverseline[0]):int(self.transverseline[1]),
                       int(self.verticalline[0]):int(self.verticalline[1])]
            cv2.imwrite(self.cut_pic_path + '%d_0.png' % order_number, crop_img)

            # 输出图片保存信息
            self.queue_dispose.put([order_number, 0])
            print('Picture: %s%d_0.png Save Successful' % (self.cut_pic_path, order_number))

            # 保存下半张图片
            img = all_img.copy()
            crop_img = img[int(self.transverseline[1]):int(self.transverseline[2]),
                       int(self.verticalline[0]):int(self.verticalline[1])]
            cv2.imwrite(self.cut_pic_path + '%d_1.png' % order_number, crop_img)

            # 输出图片保存信息
            self.queue_dispose.put([order_number, 1])
            print('Picture: %s%d_1.png Save Successful' % (self.cut_pic_path, order_number))


    def start_search_multiprocess(self, queue_dispose, queue_write):

        v = GoodClassifier(shopping_list_path='./shopping_list.txt', weight_path='./weight.h5')

        sign = 1
        while sign:
            while not queue_dispose.empty():

                value = queue_dispose.get()

                print('开始处理%d-%d数据' % (value[0], value[1]))

                name, confidence = v.single_recognize(self.cut_pic_path + '%d_%d.png' % (value[0], value[1]))

                print('%d-%d识别结果: %s-%s' % (value[0], value[1], name, confidence))

                queue_write.put([confidence, name, value[0], value[1]])

                if value[0] == 7 and value[1] == 1:
                    print('处理结束！！！！！')
                    sign = 0
                    break


    def final_process_queue(self):
        Lists = []
        while not self.queue_write.empty():
            item = self.queue_write.get()
            Lists.append([item[0], item[1], item[2], item[3]])
        # 对结果排序
        Lists = sorted(Lists, reverse=True)
        # 输出识别结果
        print(Lists)

        # 已经获得的物品名称列表
        Already_Get = []
        # 已经获得的物品位置([区号, 编号])
        Already_Get_Position = []
        # 需抓取的列表，[区号, 编号, 物品名称] - ['B', 3, 'badminton']
        return_list = []

        # 遍历初步识别的物品列表
        for Item in Lists:
            # [识别度, 物品名称, 照片序号， 上下编号{上-0|下-1}] 转 [区号, 编号, 物品名称]
            FItem = self.callistcontent(Item)

            if Item[1] in Already_Get:
                continue

            name = Item[1]

            # 确认结果
            if name != 'other' and name != 'yellow cube':
                # 暂时不验证，直接加入抓取列表的物品
                if name == 'wired mouse':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                elif name == 'badminton':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                elif name == 'steel ball':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                elif name == 'tennis ball':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                elif name == 'white pingpang ball':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                # elif name == 'zhong hua pencil':
                #     return_list.append(FItem)
                #     Already_Get.append(name)
                #     Already_Get_Position.append(FItem[:-1])
                elif name == 'gold jia duo bao':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                elif name == 'mimi':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                elif name == 'yang le duo':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                else:
                    # 需要验证的列表
                    try:

                        # if name == 'gold jia duo bao':
                        #     print('开始验证加多宝')
                        #     if self.jiaduobao_confirm(FItem[0], FItem[1]):
                        #         print('验证成功')
                        #         return_list.append(FItem)
                        #         Already_Get.append(name)
                        #         Already_Get_Position.append(FItem[:-1])
                        # if name == 'mimi':
                        #     print('开始验证咪咪')
                        #     if self.mimi_confirm(FItem[0], FItem[1]):
                        #         print('咪咪验证成功')
                        #         return_list.append(FItem)
                        #         Already_Get.append(name)
                        #         Already_Get_Position.append(FItem[:-1])
                        # elif name == 'yang le duo':
                        #     print('开始验证养乐多')
                        #     if self.yangleduo_confirm(FItem[0], FItem[1]):
                        #         print('养乐多验证成功')
                        #         return_list.append(FItem)
                        #         Already_Get.append(name)
                        #         Already_Get_Position.append(FItem[:-1])
                        if name == 'zhong hua pencil':
                            print('开始验证中华铅笔')
                            if self.pencil_confirm(FItem[0], FItem[1]):
                                print('中华铅笔验证成功')
                                return_list.append(FItem)
                                Already_Get.append(name)
                                Already_Get_Position.append(FItem[:-1])
                        # elif name == 'tennis ball':
                        #     print('开始验证网球')
                        #     if self.tennis_confirm(FItem[0], FItem[1]):
                        #         print('网球验证成功')
                        #         return_list.append(FItem)
                        #         Already_Get.append(name)
                        #         Already_Get_Position.append(FItem[:-1])
                        # elif name == 'white pingpang ball':
                        #     print('开始验证乒乓球')
                        #     if self.pingpong_confirm(FItem[0], FItem[1]):
                        #         print('乒乓球验证成功')
                        #         return_list.append(FItem)
                        #         Already_Get.append(name)
                        #         Already_Get_Position.append(FItem[:-1])
                    except Exception as e:
                        print(e)

        # 判断中华铅笔是已获得，否则全照片判断铅笔位置
        if not 'zhong hua pencil' in Already_Get:
            try:
                Pencil = self.find_pencil(Already_Get_Position)
                if Pencil:
                    return_list.append(Pencil)
                    Already_Get.append('zhong hua pencil')
                    Already_Get_Position.append(Pencil[:-1])
            except Exception as e:
                print(e)

        # # 判断养乐多是已获得，否则全照片判断养乐多位置
        # if not 'yang le duo' in Already_Get:
        #     try:
        #         Yangleduo = self.find_yangleduo(Already_Get_Position)
        #         if Yangleduo:
        #             return_list.append(Yangleduo)
        #             Already_Get.append('yang le duo')
        #             Already_Get_Position.append(Yangleduo[:-1])
        #     except Exception as e:
        #         print(e)
        #
        # # 判断咪咪是已获得，否则全照片判断咪咪位置
        # if not 'mimi' in Already_Get:
        #     try:
        #         Mimi = self.find_mimi(Already_Get_Position)
        #         if Mimi:
        #             return_list.append(Mimi)
        #             Already_Get.append('mimi')
        #             Already_Get_Position.append(Mimi[:-1])
        #     except Exception as e:
        #         print(e)
        #
        # # 判断加多宝是已获得，否则全照片判断加多宝位置
        # if not 'gold jia duo bao' in Already_Get:
        #     try:
        #         Jiaduobao = self.find_jiaduobao(Already_Get_Position)
        #         if Jiaduobao:
        #             return_list.append(Jiaduobao)
        #             Already_Get.append('gold jia duo bao')
        #             Already_Get_Position.append(Jiaduobao[:-1])
        #     except Exception as e:
        #         print(e)
        #
        # # 判断乒乓是已获得，否则全照片判断乒乓位置
        # if not 'white pingpang ball' in Already_Get:
        #     try:
        #         Pingpong = self.find_pingpong(Already_Get_Position)
        #         if Pingpong:
        #             return_list.append(Pingpong)
        #             Already_Get.append('white pingpang ball')
        #             Already_Get_Position.append(Pingpong[:-1])
        #     except Exception as e:
        #         print(e)
        #
        # # 判断网球是已获得，否则全照片判断网球位置
        # if not 'tennis ball' in Already_Get:
        #     try:
        #         Tennis = self.find_tennis(Already_Get_Position)
        #         if Tennis:
        #             return_list.append(Tennis)
        #             Already_Get.append('tennis ball')
        #             Already_Get_Position.append(Tennis[:-1])
        #     except Exception as e:
        #         print(e)

        # 已排序抓取列表
        sorted_return_list = []

        # 首先加入中华铅笔
        for Item in return_list:
            if Item[2] == 'zhong hua pencil':
                sorted_return_list.append(Item)
                return_list.remove(Item)
                break

        # 已抓取列表倒序排列
        return_list = sorted(return_list, reverse=True)

        # 已抓取列表中物品加入已排序抓取列表
        for Item in return_list:
            sorted_return_list.append(Item)

        # 计算未识别列表
        Not_Get = []
        for i in ['gold jia duo bao', 'zhong hua pencil', 'mimi', 'yang le duo', 'white pingpang ball', 'steel ball',
                  'badminton', 'tennis ball', 'wired mouse']:
            if i not in Already_Get:
                Not_Get.append(i)

        print('-----------------------Sorted Return List-----------------------')
        print(sorted_return_list)
        print('---------------------------Already Get--------------------------')
        print(Already_Get)
        print('-----------------------------Not Get----------------------------')
        print(Not_Get)

        # 生成抓取列表
        self.second_data = sorted_return_list[:]

    # B,C,D区图片处理
    def second_process(self, order_number):
        print('\033[0;37;40m')

        # 启动进程
        if self.isfinsh:
            if os.path.exists(self.tempfile):
                os.remove(self.tempfile)
            if os.path.exists(self.resultfile):
                os.remove(self.resultfile)
            if os.path.exists(self.finishfile):
                os.remove(self.finishfile)
            self.isfinsh = False
            p = Process(target=self.start_search, args=(self.tempfile,))
            p.daemon = True
            p.start()

        path = self.photo_path + '%d.png' % order_number
        all_img = cv2.imread(path)

        # 保存上半张图片
        img = all_img.copy()
        crop_img = img[int(self.transverseline[0]):int(self.transverseline[1]),int(self.verticalline[0]):int(self.verticalline[1])]
        cv2.imwrite(self.cut_pic_path + '%d_0.png' % order_number, crop_img)
        # 输出图片保存信息
        print('Picture: %s%d_0.png Save Successful' % (self.cut_pic_path, order_number))
        # 添加进程任务
        self.temp_lock.acquire()
        try:
            # self.Ready_To_Search.append([order_number, 0])
            print('Write To File: %d-0' % (order_number))
            fs = open(self.tempfile, 'a+')
            fs.write("%d-0\n" % order_number)
            fs.close()
        finally:
            self.temp_lock.release()

        # 保存下半张图片
        img = all_img.copy()
        crop_img = img[int(self.transverseline[1]):int(self.transverseline[2]),int(self.verticalline[0]):int(self.verticalline[1])]
        cv2.imwrite(self.cut_pic_path + '%d_1.png' % order_number, crop_img)
        # 输出图片保存信息
        print('Picture: %s%d_1.png Save Successful' % (self.cut_pic_path, order_number))
        # 添加进程任务
        self.temp_lock.acquire()
        try:
            # self.Ready_To_Search.append([order_number, 1])
            print('Write To File: %d-1' % (order_number))
            fs = open(self.tempfile, 'a+')
            fs.write("%d-1\n" % order_number)
            fs.close()
        finally:
            self.temp_lock.release()

    # 查询列表并组织识别进程函数
    def start_search(self, tempfile):
        print("Recongnition Process Start")
        # 已经查询列表
        Now_List = []
        v = GoodClassifier(shopping_list_path='./shopping_list.txt', weight_path='./weight.h5')
        while True:
            if os.path.exists(self.finishfile):
                print("Recongnition Process End")
                break

            if not os.path.exists(self.tempfile):
                time.sleep(0.5)
                continue
            # if len(self.Ready_To_Search) == 0:
            #     print(self.Ready_To_Search)
            #     time.sleep(0.5)
            #     continue
            # self.temp_lock.acquire()
            # try:
            #     Item = self.Ready_To_Search.pop()
            #     print(Item)
            # finally:
            #     self.temp_lock.release()
            self.temp_lock.acquire()
            tmpFile = open(tempfile)
            try:
                while True:
                    str = tmpFile.readline()[:-1]
                    if str == "":
                        break
                    Item = str.split('-', 1)
                    if Item not in Now_List:
                        Now_List.append(Item)
                        break
                if str == "":
                    time.sleep(0.5)
                    continue
            finally:
                tmpFile.close()
                self.temp_lock.release()
            self.tf_noconfirm(int(Item[0]), int(Item[1]), v)

    # 用TranserFlow识别并验证
    def tf_confirm(self, order_number, position_number, v):
        print('开始处理%d-%d数据' % (order_number, position_number))

        name, confidence = v.single_recognize(self.cut_pic_path + '%d_%d.png' % (order_number, position_number))
        # name = 'wired mouse'
        # confidence = 0.9
        print('%d-%d识别结果: %s-%s' % (order_number, position_number, name, confidence))

        Result = []
        if name != 'other' and name!= 'yellow cube':
            #暂时不验证，直接加入抓取列表的物品
            if name == 'wired mouse':
                Result = [confidence, name, order_number, position_number, True]
            elif name == 'badminton':
                Result = [confidence, name, order_number, position_number, True]
            elif name == 'steel ball':
                Result = [confidence, name, order_number, position_number, True]
            # elif name == 'tennis ball':
            #     Result = [confidence, name, order_number, position_number, True]
            elif name == 'white pingpang ball':
                Result = [confidence, name, order_number, position_number, True]
            else:
                #需要验证的列表
                try:
                    #[识别度, 物品名称, 照片序号， 上下编号{上-0|下-1}] 转 [区号, 编号, 物品名称]
                    FItem = self.callistcontent([confidence, name, order_number, position_number])

                    # if name == 'white pingpang ball':
                    #     if self.pingpong_confirm(FItem[0], FItem[1]):
                    #         self.pre_items.append([confidence, name, order_number, position_number, True])
                    #     else
                    #         self.pre_items.append([confidence, name, order_number, position_number, False])
                    # elif name == 'gold jia duo bao':
                    if name == 'gold jia duo bao':
                        if self.jiaduobao_confirm(FItem[0], FItem[1]):
                            Result = [confidence, name, order_number, position_number, True]
                        else:
                            Result = [confidence, name, order_number, position_number, False]
                    elif name == 'mimi':
                        if self.mimi_confirm(FItem[0], FItem[1]):
                            Result = [confidence, name, order_number, position_number, True]
                        else:
                            Result = [confidence, name, order_number, position_number, False]
                    elif name == 'yang le duo':
                        if self.yangleduo_confirm(FItem[0], FItem[1]):
                            Result = [confidence, name, order_number, position_number, True]
                        else:
                            Result = [confidence, name, order_number, position_number, False]
                    elif name == 'zhong hua pencil':
                        if self.pencil_confirm(FItem[0], FItem[1]):
                            print('铅笔验证通过')
                            Result = [confidence, name, order_number, position_number, True]
                        else:
                            print('铅笔验证失败')
                            Result = [confidence, name, order_number, position_number, False]
                    elif name == 'tennis ball':
                        if self.tennis_confirm(FItem[0], FItem[1]):
                            print('网球验证通过')
                            Result = [confidence, name, order_number, position_number, True]
                        else:
                            print('网球验证失败')
                            Result = [confidence, name, order_number, position_number, False]
                except Exception as e:
                    print(e)

        if Result != []:
            # self.pre_items.append(Result)
            fs = open(self.resultfile, 'a+')
            fs.write("%f-%s-%d-%d-%s\n" % (Result[0], Result[1], Result[2], Result[3], Result[4]))
            # fs.write("%f-%s-%d-%d-%s\n" % (confidence, name, order_number, position_number, True))
            fs.close()

        print('%d-%d数据处理完成' % (order_number, position_number))

        if order_number == 7 and position_number == 1:
            fin = open(self.finishfile, 'w')
            fin.close()

    # 返回B,C,D区结果
    def return_second_result(self):
        print(self.second_data)
        return self.second_data

    # 最终处理
    def final_process(self):
        # 等待线程处理结束
        # while Count < 6:
        #     if not os.path.exists(self.resultfile):
        #         time.sleep(1)
        #         continue
        #     Count = 0
        #     tmpFile = open(self.resultfile)
        #     try:
        #         while True:
        #             str = tmpFile.readline()[:-1]
        #             if str == '':
        #                 break
        #             Count += 1
        #             Item = str.split('-', 4)
        #         if str == "":
        #             continue
        #     finally:
        #         tmpFile.close()
        #     time.sleep(1)
        while not os.path.exists(self.finishfile):
            time.sleep(0.5)

        tmpFile = open(self.resultfile)
        try:
            while True:
                str = tmpFile.readline()[:-1]
                if str == '':
                    break
                Item = str.split('-', 4)
                self.pre_items.append([float(Item[0]), Item[1], int(Item[2]), int(Item[3]), Item[4] == 'True'])
        finally:
            tmpFile.close()

        # while len(self.pre_items) != 36:
        #     time.sleep(0.3)

        # 对结果排序
        Lists = sorted(self.pre_items, reverse=True)
        # 输出识别结果
        print(Lists)

        #已经获得的物品名称列表
        Already_Get = []
        #已经获得的物品位置([区号, 编号])
        Already_Get_Position = []
        #需抓取的列表，[区号, 编号, 物品名称] - ['B', 3, 'badminton']
        return_list = []

        #遍历初步识别的物品列表
        for Item in Lists:
            #[识别度, 物品名称, 照片序号， 上下编号{上-0|下-1}] 转 [区号, 编号, 物品名称]
            FItem = self.callistcontent(Item[:-1])

            if Item[1] in Already_Get:
                continue
            elif Item[4]:
                return_list.append(FItem)
                Already_Get.append(Item[1])
                Already_Get_Position.append(FItem[:-1])

        #判断中华铅笔是已获得，否则全照片判断铅笔位置
        if not 'zhong hua pencil' in Already_Get:
            try:
                Pencil = self.find_pencil(Already_Get_Position)
                if Pencil:
                    return_list.append(Pencil)
                    Already_Get.append('zhong hua pencil')
                    Already_Get_Position.append(Pencil[:-1])
            except Exception as e:
                print(e)

        #判断养乐多是已获得，否则全照片判断养乐多位置
        if not 'yang le duo' in Already_Get:
            try:
                Yangleduo = self.find_yangleduo(Already_Get_Position)
                if Yangleduo:
                    return_list.append(Yangleduo)
                    Already_Get.append('yang le duo')
                    Already_Get_Position.append(Yangleduo[:-1])
            except Exception as e:
                print(e)

        #判断咪咪是已获得，否则全照片判断咪咪位置
        if not 'mimi' in Already_Get:
            try:
                Mimi = self.find_mimi(Already_Get_Position)
                if Mimi:
                    return_list.append(Mimi)
                    Already_Get.append('mimi')
                    Already_Get_Position.append(Mimi[:-1])
            except Exception as e:
                print(e)

        #判断加多宝是已获得，否则全照片判断加多宝位置
        if not 'gold jia duo bao' in Already_Get:
            try:
                Jiaduobao = self.find_jiaduobao(Already_Get_Position)
                if Jiaduobao:
                    return_list.append(Jiaduobao)
                    Already_Get.append('gold jia duo bao')
                    Already_Get_Position.append(Jiaduobao[:-1])
            except Exception as e:
                print(e)

        # # 判断乒乓是已获得，否则全照片判断乒乓位置
        # if not 'white pingpang ball' in Already_Get:
        #     try:
        #         Pingpong = self.find_pingpong(Already_Get_Position)
        #         if Pingpong:
        #             return_list.append(Pingpong)
        #             Already_Get.append('white pingpang ball')
        #             Already_Get_Position.append(Pingpong[:-1])
        #     except Exception as e:
        #         print(e)

        # 判断网球是已获得，否则全照片判断网球位置
        if not 'tennis ball' in Already_Get:
            try:
                Tennis = self.find_tennis(Already_Get_Position)
                if Tennis:
                    return_list.append(Tennis)
                    Already_Get.append('tennis ball')
                    Already_Get_Position.append(Tennis[:-1])
            except Exception as e:
                print(e)
        
        #已排序抓取列表
        sorted_return_list = []
        
        #首先加入中华铅笔
        for Item in return_list:
            if Item[2] == 'zhong hua pencil':
                sorted_return_list.append(Item)
                return_list.remove(Item)
                break

        # 已抓取列表倒序排列
        return_list = sorted(return_list, reverse = True)

        # 已抓取列表中物品加入已排序抓取列表
        for Item in return_list:
            sorted_return_list.append(Item)

        # 计算未识别列表
        Not_Get = []
        for i in ['gold jia duo bao', 'zhong hua pencil', 'mimi', 'yang le duo', 'white pingpang ball', 'steel ball', 'badminton', 'tennis ball', 'wired mouse']:
            if i not in Already_Get:
                Not_Get.append(i)

        print('-----------------------Sorted Return List-----------------------')
        print(sorted_return_list)
        print('---------------------------Already Get--------------------------')
        print(Already_Get)
        print('-----------------------------Not Get----------------------------')
        print(Not_Get)

        # 生成抓取列表
        self.second_data = sorted_return_list[:]

    # 用TranserFlow识别 - 不带验证 - 配套 final_process_confirm()
    def tf_noconfirm(self, order_number, position_number, v):
        print('开始处理%d-%d数据' % (order_number, position_number))

        name, confidence = v.single_recognize(self.cut_pic_path + '%d_%d.png' % (order_number, position_number))
        # name = 'wired mouse'
        # confidence = 0.9
        print('%d-%d识别结果: %s-%s' % (order_number, position_number, name, confidence))

        fs = open(self.resultfile, 'a+')
        fs.write("%f-%s-%d-%d\n" % (confidence, name, order_number, position_number))
        fs.close()

        print('%d-%d数据处理完成' % (order_number, position_number))

        if order_number == 7 and position_number == 1:
            fin = open(self.finishfile, 'w')
            fin.close()

    # 最终处理 - 带确认
    def final_process_confirm(self):
        while not os.path.exists(self.finishfile):
            time.sleep(0.5)

        tmpFile = open(self.resultfile)
        try:
            while True:
                str = tmpFile.readline()[:-1]
                if str == '':
                    break
                Item = str.split('-', 3)
                self.pre_items.append([float(Item[0]), Item[1], int(Item[2]), int(Item[3])])
        finally:
            tmpFile.close()

        # 对结果排序
        Lists = sorted(self.pre_items, reverse=True)
        # 输出识别结果
        print(Lists)

        # 已经获得的物品名称列表
        Already_Get = []
        # 已经获得的物品位置([区号, 编号])
        Already_Get_Position = []
        # 需抓取的列表，[区号, 编号, 物品名称] - ['B', 3, 'badminton']
        return_list = []

        # 遍历初步识别的物品列表
        for Item in Lists:
            # [识别度, 物品名称, 照片序号， 上下编号{上-0|下-1}] 转 [区号, 编号, 物品名称]
            FItem = self.callistcontent(Item)

            if Item[1] in Already_Get:
                continue

            name = Item[1]

            # 确认结果
            if name != 'other' and name != 'yellow cube':
                # 暂时不验证，直接加入抓取列表的物品
                if name == 'wired mouse':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                elif name == 'badminton':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                elif name == 'steel ball':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                elif name == 'tennis ball':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                elif name == 'white pingpang ball':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                else:
                    # 需要验证的列表
                    try:

                        if name == 'gold jia duo bao':
                            print('开始验证加多宝')
                            if self.jiaduobao_confirm(FItem[0], FItem[1]):
                                print('验证成功')
                                return_list.append(FItem)
                                Already_Get.append(name)
                                Already_Get_Position.append(FItem[:-1])
                        elif name == 'mimi':
                            print('开始验证咪咪')
                            if self.mimi_confirm(FItem[0], FItem[1]):
                                print('咪咪验证成功')
                                return_list.append(FItem)
                                Already_Get.append(name)
                                Already_Get_Position.append(FItem[:-1])
                        elif name == 'yang le duo':
                            print('开始验证养乐多')
                            if self.yangleduo_confirm(FItem[0], FItem[1]):
                                print('养乐多验证成功')
                                return_list.append(FItem)
                                Already_Get.append(name)
                                Already_Get_Position.append(FItem[:-1])
                        elif name == 'zhong hua pencil':
                            print('开始验证中华铅笔')
                            if self.pencil_confirm(FItem[0], FItem[1]):
                                print('中华铅笔验证成功')
                                return_list.append(FItem)
                                Already_Get.append(name)
                                Already_Get_Position.append(FItem[:-1])
                        # elif name == 'tennis ball':
                        #     print('开始验证网球')
                        #     if self.tennis_confirm(FItem[0], FItem[1]):
                        #         print('网球验证成功')
                        #         return_list.append(FItem)
                        #         Already_Get.append(name)
                        #         Already_Get_Position.append(FItem[:-1])
                        # elif name == 'white pingpang ball':
                        #     print('开始验证乒乓球')
                        #     if self.pingpong_confirm(FItem[0], FItem[1]):
                        #         print('乒乓球验证成功')
                        #         return_list.append(FItem)
                        #         Already_Get.append(name)
                        #         Already_Get_Position.append(FItem[:-1])
                    except Exception as e:
                        print(e)

        # 判断中华铅笔是已获得，否则全照片判断铅笔位置
        if not 'zhong hua pencil' in Already_Get:
            try:
                Pencil = self.find_pencil(Already_Get_Position)
                if Pencil:
                    return_list.append(Pencil)
                    Already_Get.append('zhong hua pencil')
                    Already_Get_Position.append(Pencil[:-1])
            except Exception as e:
                print(e)

        # 判断养乐多是已获得，否则全照片判断养乐多位置
        if not 'yang le duo' in Already_Get:
            try:
                Yangleduo = self.find_yangleduo(Already_Get_Position)
                if Yangleduo:
                    return_list.append(Yangleduo)
                    Already_Get.append('yang le duo')
                    Already_Get_Position.append(Yangleduo[:-1])
            except Exception as e:
                print(e)

        # 判断咪咪是已获得，否则全照片判断咪咪位置
        if not 'mimi' in Already_Get:
            try:
                Mimi = self.find_mimi(Already_Get_Position)
                if Mimi:
                    return_list.append(Mimi)
                    Already_Get.append('mimi')
                    Already_Get_Position.append(Mimi[:-1])
            except Exception as e:
                print(e)

        # 判断加多宝是已获得，否则全照片判断加多宝位置
        if not 'gold jia duo bao' in Already_Get:
            try:
                Jiaduobao = self.find_jiaduobao(Already_Get_Position)
                if Jiaduobao:
                    return_list.append(Jiaduobao)
                    Already_Get.append('gold jia duo bao')
                    Already_Get_Position.append(Jiaduobao[:-1])
            except Exception as e:
                print(e)

        # # 判断乒乓是已获得，否则全照片判断乒乓位置
        # if not 'white pingpang ball' in Already_Get:
        #     try:
        #         Pingpong = self.find_pingpong(Already_Get_Position)
        #         if Pingpong:
        #             return_list.append(Pingpong)
        #             Already_Get.append('white pingpang ball')
        #             Already_Get_Position.append(Pingpong[:-1])
        #     except Exception as e:
        #         print(e)

        # 判断网球是已获得，否则全照片判断网球位置
        if not 'tennis ball' in Already_Get:
            try:
                Tennis = self.find_tennis(Already_Get_Position)
                if Tennis:
                    return_list.append(Tennis)
                    Already_Get.append('tennis ball')
                    Already_Get_Position.append(Tennis[:-1])
            except Exception as e:
                print(e)

        # 已排序抓取列表
        sorted_return_list = []

        # 首先加入中华铅笔
        for Item in return_list:
            if Item[2] == 'zhong hua pencil':
                sorted_return_list.append(Item)
                return_list.remove(Item)
                break

        # 已抓取列表倒序排列
        return_list = sorted(return_list, reverse=True)

        # 已抓取列表中物品加入已排序抓取列表
        for Item in return_list:
            sorted_return_list.append(Item)

        # 计算未识别列表
        Not_Get = []
        for i in ['gold jia duo bao', 'zhong hua pencil', 'mimi', 'yang le duo', 'white pingpang ball', 'steel ball',
                  'badminton', 'tennis ball', 'wired mouse']:
            if i not in Already_Get:
                Not_Get.append(i)

        print('-----------------------Sorted Return List-----------------------')
        print(sorted_return_list)
        print('---------------------------Already Get--------------------------')
        print(Already_Get)
        print('-----------------------------Not Get----------------------------')
        print(Not_Get)

        # 生成抓取列表
        self.second_data = sorted_return_list[:]

    # B,C,D区图片处理 - 用PIPE
    def second_process_pipe(self, order_number):
        print('\033[0;37;40m')

        # 启动进程
        if self.isfinsh:
            self.isfinsh = False
            p = Process(target = self.start_search_pipe, args=(self.tempfile, self.data_pipe, self.result_pipe))
            p.daemon = True
            p.start()

        if order_number != 0:
            path = self.photo_path + '%d.png' % order_number
            all_img = cv2.imread(path)

            # 保存上半张图片
            img = all_img.copy()
            crop_img = img[int(self.transverseline[0]):int(self.transverseline[1]),int(self.verticalline[0]):int(self.verticalline[1])]
            if self.gaosi_sign:
                crop_img = self.gaosi(crop_img)

            # cv2.imwrite(self.cut_pic_path + '%d_0.png' % order_number, crop_img)
            # # 输出图片保存信息
            # print('Picture: %s%d_0.png Save Successful' % (self.cut_pic_path, order_number))

            # 添加进程任务
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            crop_img = cv2.resize(crop_img, (299, 299))
            self.data_pipe[0].send([order_number, 0, crop_img])

            # 保存下半张图片
            img = all_img.copy()
            crop_img = img[int(self.transverseline[1]):int(self.transverseline[2]),int(self.verticalline[0]):int(self.verticalline[1])]
            if self.gaosi_sign:
                crop_img = self.gaosi(crop_img)

            # cv2.imwrite(self.cut_pic_path + '%d_1.png' % order_number, crop_img)
            # # 输出图片保存信息
            # print('Picture: %s%d_1.png Save Successful' % (self.cut_pic_path, order_number))

            # 添加进程任务
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            crop_img = cv2.resize(crop_img, (299, 299))
            self.data_pipe[0].send([order_number, 1, crop_img])

    # 查询列表并组织识别进程函数 - 用PIPE
    def start_search_pipe(self, tempfile, datapipe, resultpipe):
        print("Recongnition Process Start")

        v = GoodClassifier(shopping_list_path='./shopping_list.txt', weight_path='./weight.h5')

        try:
            while True:
                Item = datapipe[1].recv()
                if Item[0] == 0:
                    print("Recongnition Process End")
                    break
                # print(Item)
                self.tf_noconfirm_pipe(int(Item[0]), int(Item[1]), v, datapipe, resultpipe, Item[2])
        except EOFError:
            datapipe.close()

    # 用TranserFlow识别 - 不带验证 - 用PIPE - 配套 final_process_confirm_pipe()
    def tf_noconfirm_pipe(self, order_number, position_number, v, datapipe, resultpipe, img):
        print('开始处理%d-%d数据' % (order_number, position_number))

        # name, confidence = v.single_recognize(self.cut_pic_path + '%d_%d.png' % (order_number, position_number))
        name, confidence = v.single_recognize(img)
        print('%d-%d识别结果: %s-%s' % (order_number, position_number, name, confidence))

        # 送出结果
        resultpipe[0].send([confidence, name, order_number, position_number])

        print('%d-%d数据处理完成' % (order_number, position_number))

        if order_number == self.end and position_number == 1:
            datapipe[0].send([0, 0, None])
            resultpipe[0].send([0, 0, None])

    # 最终处理 - 带确认 - 用PIPE
    def final_process_confirm_pipe(self):
        try:
            while True:
                Item = self.result_pipe[1].recv()

                if Item[0] == 0:
                    break

                self.pre_items.append([float(Item[0]), Item[1], int(Item[2]), int(Item[3])])
        finally:
            self.data_pipe[0].close()
            self.result_pipe[0].close()

        # 对结果排序
        Lists = sorted(self.pre_items, reverse=True)
        # 输出识别结果
        print('Final_Process: ', Lists)

        # 已经获得的物品名称列表
        Already_Get = []
        # 已经获得的物品位置([区号, 编号])
        Already_Get_Position = []
        # 需抓取的列表，[区号, 编号, 物品名称] - ['B', 3, 'badminton']
        return_list = []

        # 遍历初步识别的物品列表
        for Item in Lists:
            # [识别度, 物品名称, 照片序号， 上下编号{上-0|下-1}] 转 [区号, 编号, 物品名称]
            FItem = self.callistcontent(Item)

            if Item[1] in Already_Get:
                continue

            name = Item[1]

            # 确认结果
            if name != 'other' and name != 'yellow cube':
                # 暂时不验证，直接加入抓取列表的物品
                if name == 'wired mouse':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                elif name == 'badminton':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                elif name == 'steel ball':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                elif name == 'tennis ball':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                elif name == 'white pingpang ball':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                elif name == 'gold jia duo bao':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                elif name == 'zhong hua pencil':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                elif name == 'mimi':
                    return_list.append(FItem)
                    Already_Get.append(name)
                    Already_Get_Position.append(FItem[:-1])
                else:
                    # 需要验证的列表
                    try:

                        # if name == 'gold jia duo bao':
                        #     print('开始验证加多宝')
                        #     if self.jiaduobao_confirm(FItem[0], FItem[1]):
                        #         print('验证成功')
                        #         return_list.append(FItem)
                        #         Already_Get.append(name)
                        #         Already_Get_Position.append(FItem[:-1])
                        # if name == 'mimi':
                        #     print('开始验证咪咪')
                        #     if self.mimi_confirm(FItem[0], FItem[1]):
                        #         print('咪咪验证成功')
                        #         return_list.append(FItem)
                        #         Already_Get.append(name)
                        #         Already_Get_Position.append(FItem[:-1])
                        if name == 'yang le duo':
                            print('开始验证养乐多')
                            if self.yangleduo_confirm(FItem[0], FItem[1]):
                                print('养乐多验证成功')
                                return_list.append(FItem)
                                Already_Get.append(name)
                                Already_Get_Position.append(FItem[:-1])
                        # elif name == 'zhong hua pencil':
                        #     print('开始验证中华铅笔')
                        #     if self.pencil_confirm(FItem[0], FItem[1]):
                        #         print('中华铅笔验证成功')
                        #         return_list.append(FItem)
                        #         Already_Get.append(name)
                        #         Already_Get_Position.append(FItem[:-1])
                        # elif name == 'tennis ball':
                        #     print('开始验证网球')
                        #     if self.tennis_confirm(FItem[0], FItem[1]):
                        #         print('网球验证成功')
                        #         return_list.append(FItem)
                        #         Already_Get.append(name)
                        #         Already_Get_Position.append(FItem[:-1])
                        # elif name == 'white pingpang ball':
                        #     print('开始验证乒乓球')
                        #     if self.pingpong_confirm(FItem[0], FItem[1]):
                        #         print('乒乓球验证成功')
                        #         return_list.append(FItem)
                        #         Already_Get.append(name)
                        #         Already_Get_Position.append(FItem[:-1])
                    except Exception as e:
                        print(e)

        # 判断中华铅笔是已获得，否则全照片判断铅笔位置
        if not 'zhong hua pencil' in Already_Get:
            try:
                Pencil = self.find_pencil(Already_Get_Position)
                if Pencil:
                    return_list.append(Pencil)
                    Already_Get.append('zhong hua pencil')
                    Already_Get_Position.append(Pencil[:-1])
            except Exception as e:
                print(e)

        # 判断养乐多是已获得，否则全照片判断养乐多位置
        if not 'yang le duo' in Already_Get:
            try:
                Yangleduo = self.find_yangleduo(Already_Get_Position)
                if Yangleduo:
                    return_list.append(Yangleduo)
                    Already_Get.append('yang le duo')
                    Already_Get_Position.append(Yangleduo[:-1])
            except Exception as e:
                print(e)

        # # 判断咪咪是已获得，否则全照片判断咪咪位置
        # if not 'mimi' in Already_Get:
        #     try:
        #         Mimi = self.find_mimi(Already_Get_Position)
        #         if Mimi:
        #             return_list.append(Mimi)
        #             Already_Get.append('mimi')
        #             Already_Get_Position.append(Mimi[:-1])
        #     except Exception as e:
        #         print(e)

        # # 判断加多宝是已获得，否则全照片判断加多宝位置
        # if not 'gold jia duo bao' in Already_Get:
        #     try:
        #         Jiaduobao = self.find_jiaduobao(Already_Get_Position)
        #         if Jiaduobao:
        #             return_list.append(Jiaduobao)
        #             Already_Get.append('gold jia duo bao')
        #             Already_Get_Position.append(Jiaduobao[:-1])
        #     except Exception as e:
        #         print(e)

        # # 判断乒乓是已获得，否则全照片判断乒乓位置
        # if not 'white pingpang ball' in Already_Get:
        #     try:
        #         Pingpong = self.find_pingpong(Already_Get_Position)
        #         if Pingpong:
        #             return_list.append(Pingpong)
        #             Already_Get.append('white pingpang ball')
        #             Already_Get_Position.append(Pingpong[:-1])
        #     except Exception as e:
        #         print(e)

        # 判断网球是已获得，否则全照片判断网球位置
        # if not 'tennis ball' in Already_Get:
        #     try:
        #         Tennis = self.find_tennis(Already_Get_Position)
        #         if Tennis:
        #             return_list.append(Tennis)
        #             Already_Get.append('tennis ball')
        #             Already_Get_Position.append(Tennis[:-1])
        #     except Exception as e:
        #         print(e)

        # 已排序抓取列表
        sorted_return_list = []

        # 首先加入中华铅笔
        for Item in return_list:
            if Item[2] == 'zhong hua pencil':
                sorted_return_list.append(Item)
                return_list.remove(Item)
                break

        # 已抓取列表倒序排列
        return_list = sorted(return_list, reverse=True)

        # 已抓取列表中物品加入已排序抓取列表
        for Item in return_list:
            sorted_return_list.append(Item)

        # 计算未识别列表
        Not_Get = []
        for i in ['gold jia duo bao', 'zhong hua pencil', 'mimi', 'yang le duo', 'white pingpang ball', 'steel ball',
                  'badminton', 'tennis ball', 'wired mouse']:
            if i not in Already_Get:
                Not_Get.append(i)

        print('-----------------------Sorted Return List-----------------------')
        print(sorted_return_list)
        print('---------------------------Already Get--------------------------')
        print(Already_Get)
        print('-----------------------------Not Get----------------------------')
        print(Not_Get)

        lists = sorted(sorted_return_list[:])
        temporary = []
        adjust = []
        for i in lists:
            if i[2] == 'zhong hua pencil':
                temporary = i
            else:
                self.second_data.append(i)
                # self.second_data = sorted_return_list[:]
        for i, j in enumerate(self.second_data):
            if i != len(self.second_data) - 1:
                if self.second_data[i][0] == self.second_data[i + 1][0]:
                    if self.second_data[i + 1][1] - self.second_data[i][1] == 1 and self.second_data[i + 1][1] % 2 == 0:
                        adjust = self.second_data[i + 1]
                        self.second_data[i + 1] = self.second_data[i]
                        self.second_data[i] = adjust
            else:
                pass
        if temporary:
            self.second_data.append(temporary)
        else:
            pass

    # 列表转换函数 - [识别度, 物品名称, 照片序号， 上下编号{上-0|下-1}] 转 [区号, 编号, 物品名称]
    def callistcontent(self, Item):
        if (int(Item[2]) - 7) // 6 == 0:
            quhao = 'B'
        elif (int(Item[2]) - 7) // 6 == 1:
            quhao = 'C'
        else:
            quhao = 'D'
        return [quhao, (int(Item[2]) - 7) % 6 * 2 + int(Item[3]) + 1, Item[1]]

    # 铅笔确认函数
    def pencil_confirm(self, quhao, order_number):
        # 区域码
        print('铅笔验证开始')
        Number = 0

        # 计算区域码
        if quhao == 'A':
            Number += 0
            Number += int(order_number)
        elif quhao == 'B':
            Number += 12 
            Number += int(order_number)
        elif quhao == 'C':
            Number += 24
            Number += int(order_number)
        else:
            Number += 36
            Number += int(order_number)

        img_all = cv2.imread(self.photo_path + '%s.png' % str((Number + 1) // 2))
        img = img_all.copy()

        if Number % 2 == 1:
            src = img[int(self.transverseline[0]):int(self.transverseline[1]), int(self.verticalline[0]):int(self.verticalline[1])]
        else:
            src = img[int(self.transverseline[1]):int(self.transverseline[2]), int(self.verticalline[0]):int(self.verticalline[1])]

        dst = cv2.Canny(src, 50, 200)
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        adds = 0
        flag = 0
        old_data = []
        lists = []
        yuzhi = [4, 8]

        lines = cv2.HoughLinesP(dst, 1, math.pi / 180.0, 100, np.array([]), 45, 25)
        try:
            a, b, c = lines.shape
            for i in range(a):
                if abs(lines[i][0][3] - lines[i][0][1]) < 10:
                    if abs(lines[i][0][2] - lines[i][0][0]) > 150:
                        cv2.line(cdst, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]),
                                    (0, 255, 0), 3, cv2.LINE_AA)
                        lists.append([(lines[i][0][1] + lines[i][0][3]) // 2, lines[i][0][0], lines[i][0][2]])
            lists = sorted(lists, reverse=True)

            for i, j in enumerate(lists):
                if flag == 1:
                    break
                old_data.append(j)
                for q in lists[i + 1:len(lists)]:
                    if j[0] - q[0] < 50:
                        old_data.append(q)
                        adds += 1
                    else:
                        if adds >= yuzhi[Number % 2]:
                            flag = 1
                            break
                        else:
                            adds = 0
                            old_data = []
                            break
                if adds >= yuzhi[Number % 2]:
                    flag = 1
            # print(old_data)
            if old_data:
                min_x = 9999
                max_x = 0
                for i in old_data:
                    if i[1] < min_x:
                        min_x = i[1]
                    if i[2] > max_x:
                        max_x = i[2]
                max_y = old_data[0][0]
                min_y = old_data[len(old_data) - 1][0]
                # print(max_y, min_y, max_x, min_x)
                # print(max_y - min_y)
                if max_y - min_y > 20:
                    srcs = src[ min_y:max_y, min_x:max_x]
                    hsv = cv2.cvtColor(srcs, cv2.COLOR_BGR2HSV)
                    # lower_green = np.array([50, 40, 46])
                    # upper_green = np.array([110, 209, 170])
                    lower_green = np.array([50, 20, 20])
                    upper_green = np.array([110, 209, 170])
                    mask = cv2.inRange(hsv, lower_green, upper_green)
                    img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    if contours:
                        for cnt in contours:
                            x, y, w, h = cv2.boundingRect(cnt)
                            if w + h > 300:
                                print(w + h)
                                # print('zhong hua qian bi')
                                return True
                                # pass
                #             cv2.rectangle(srcs, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #     cv2.namedWindow('123')
                #     cv2.imshow("123", srcs)
                # cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
                # cv2.imshow("gray", cdst)
                # cv2.waitKey(0)
                # cv2.namedWindow('source', cv2.WINDOW_NORMAL)
                # cv2.imshow("source", src)
                # cv2.namedWindow('lines', cv2.WINDOW_NORMAL)
                # cv2.imshow("lines", cdst)
            return False
        except:
            return False

    # 铅笔全图匹配函数
    def find_pencil(self, Position):
        Qu = ['B', 'C', 'D']
        for Item in Qu:
            for order_number in range(1, 13):
                #判断是否是已识别物品位置
                if [Item, order_number] not in Position:
                    if self.pencil_confirm(Item, order_number):
                        print([Item, order_number, 'zhong hua pencil'])
                        return [Item, order_number, 'zhong hua pencil']

    # 咪咪虾条确认函数
    def mimi_confirm(self, quhao, order_number):
        Mimi_Template_Image = ["./model/92.png"]
        Yuzhi = [160]

        # 区域码
        Number = 0

        # 计算区域码
        if quhao == 'B':
            Number += 12 
            Number += int(order_number)
        elif quhao == 'C':
            Number += 24
            Number += int(order_number)
        else:
            Number += 36
            Number += int(order_number)

        try:
            img_all = cv2.imread(self.photo_path + '%s.png' % str((Number + 1) // 2))
            img = img_all.copy()

            if Number % 2 == 1:
                crop_img = img[int(self.transverseline[0]):int(self.transverseline[1]), int(self.verticalline[0]):int(self.verticalline[1])]
            else:
                crop_img = img[int(self.transverseline[1]):int(self.transverseline[2]), int(self.verticalline[0]):int(self.verticalline[1])]

            img1 = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
            img2 = cv2.imread(Mimi_Template_Image[0], 0)
            sift = cv2.xfeatures2d.SIFT_create()
            MIN_MATCH_COUNT = Yuzhi[0]

            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            # FLANN_INDEX_KDTREE = 0
            # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            # search_params = dict(checks=50)
            # flann = cv2.FlannBasedMatcher(index_params, search_params)
            # matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            good = []

            for m, n in matches:
                if m.distance < 0.80 * n.distance:
                    good.append(m)
            print(len(good), MIN_MATCH_COUNT)
            if len(good) >= MIN_MATCH_COUNT:
                return True
            else:
                return False
        except Exception as e:
            print(e)

    # 咪咪全图匹配函数
    def find_mimi(self, Position):
        Qu = ['B', 'C', 'D']
        for Item in Qu:
            for order_number in range(1, 13):
                #判断是否是已识别物品位置
                if [Item, order_number] not in Position:
                    if self.mimi_confirm(Item, order_number):
                        print([Item, order_number, 'mimi'])
                        return [Item, order_number, 'mimi']

    # 养乐多确认函数
    def yangleduo_confirm(self, quhao, order_number):
        print('养乐多确认开始')
        Yangleduo_Template_Image = ["./model/IMG_3855.JPG", "./model/1122.png"]
        Yuzhi = [100, 60]

        # 区域码
        Number = 0

        # 计算区域码
        if quhao == 'B':
            Number += 12 
            Number += int(order_number)
        elif quhao == 'C':
            Number += 24
            Number += int(order_number)
        else:
            Number += 36
            Number += int(order_number)

        try:
            img_all = cv2.imread(self.photo_path + '%s.png' % str((Number + 1) // 2))
            img = img_all.copy()

            if Number % 2 == 1:
                crop_img = img[int(self.transverseline[0]):int(self.transverseline[1]), int(self.verticalline[0]):int(self.verticalline[1])]
            else:
                crop_img = img[int(self.transverseline[1]):int(self.transverseline[2]), int(self.verticalline[0]):int(self.verticalline[1])]

            img1 = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
            img2 = cv2.imread(Yangleduo_Template_Image[(Number + 1) % 2], 0)
            sift = cv2.xfeatures2d.SIFT_create()
            MIN_MATCH_COUNT = Yuzhi[(Number + 1) % 2]

            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            # FLANN_INDEX_KDTREE = 0
            # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            # search_params = dict(checks=50)
            # flann = cv2.FlannBasedMatcher(index_params, search_params)
            # matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            print(len(good), MIN_MATCH_COUNT)
            if len(good) >= MIN_MATCH_COUNT:
                return True
            else:
                return False

        except Exception as e:
            print(e)

    # 养乐多全图匹配函数
    def find_yangleduo(self, Position):
        Qu = ['B', 'C', 'D']
        for Item in Qu:
            for order_number in range(1, 13):
                #判断是否是已识别物品位置
                if [Item, order_number] not in Position:
                    if self.yangleduo_confirm(Item, order_number):
                        return [Item, order_number, 'yang le duo']
                        # print([Item, order_number, 'yang le duo'])
                        # pass

    # 加多宝确认函数
    def jiaduobao_confirm(self, quhao, order_number):
        Jiaduobao_Template_Image = ["./model/65.png"]
        Yuzhi = [80]

        # 区域码
        Number = 0

        # 计算区域码
        if quhao == 'B':
            Number += 12 
            Number += int(order_number)
        elif quhao == 'C':
            Number += 24
            Number += int(order_number)
        else:
            Number += 36
            Number += int(order_number)

        try:
            img_all = cv2.imread(self.photo_path + '%s.png' % str((Number + 1) // 2))
            img = img_all.copy()

            if Number % 2 == 1:
                crop_img = img[int(self.transverseline[0]):int(self.transverseline[1]), int(self.verticalline[0]):int(self.verticalline[1])]
            else:
                crop_img = img[int(self.transverseline[1]):int(self.transverseline[2]), int(self.verticalline[0]):int(self.verticalline[1])]

            img1 = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
            img2 = cv2.imread(Jiaduobao_Template_Image[0], 0)
            sift = cv2.xfeatures2d.SIFT_create()
            MIN_MATCH_COUNT = Yuzhi[0]

            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            # FLANN_INDEX_KDTREE = 0
            # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            # search_params = dict(checks=50)
            # flann = cv2.FlannBasedMatcher(index_params, search_params)
            # matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            good = []

            for m, n in matches:
                if m.distance < 0.80 * n.distance:
                    good.append(m)

            print(len(good), MIN_MATCH_COUNT, quhao, order_number)
            if len(good) >= MIN_MATCH_COUNT:
                return True
            else:
                return False
        except Exception as e:
            print(e)

    # 加多宝全图匹配函数
    def find_jiaduobao(self, Position):
        Qu = ['B', 'C', 'D']
        for Item in Qu:
            for order_number in range(1, 13):
                #判断是否是已识别物品位置
                if [Item, order_number] not in Position:
                    if self.jiaduobao_confirm(Item, order_number):
                        print([Item, order_number, 'gold jia duo bao'])
                        return [Item, order_number, 'gold jia duo bao']

    # 乒乓确认函数 - 效果不好,需要加颜色检测
    def pingpong_confirm(self, quhao, order_number):
        # 区域码
        Number = 0

        # 计算区域码
        if quhao == 'B':
            Number += 12
            Number += int(order_number)
        elif quhao == 'C':
            Number += 24
            Number += int(order_number)
        else:
            Number += 36
            Number += int(order_number)

        try:
            img_all = cv2.imread(self.photo_path + '%s.png' % str((Number + 1) // 2))
            img = img_all.copy()

            crop_img = img[int(self.transverseline[0]):int(self.transverseline[2]),
                       int(self.verticalline[0]):int(self.verticalline[1])]

            sp = crop_img.shape
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(self.xml_Path + 'pingpong.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                if w + h > 200 and w + h < 400:  # 针对这个图片画出最大的外框
                    if x > sp[1] / 4 and x < sp[1] / 4 * 3:
                        if Number % 2 == 1 and y < self.transverseline[1] - self.transverseline[0]:
                            if y > (self.transverseline[1] - self.transverseline[0]) / 2:
                                img_ready = crop_img.copy()
                                img_ready = img_ready[y:y + h, x:x + w]

                                cv2.rectangle(crop_img, (x, y), (x + w, y + h), (255, 255, 255), 4)
                                # roi_gray = gray[y:y+h, x:x+w]
                                roi_color = crop_img[y:y + h, x:x + w]
                                # if self.pingpongcolor(roi_color):
                                #     return True
                                # print 'pingpong'
                                # return True
                                # cv2.imwrite('96.png', roi_color)
                                cv2.namedWindow('123', cv2.WINDOW_NORMAL)
                                cv2.imshow('123', roi_color)
                                # cv2.namedWindow('imgs', cv2.WINDOW_NORMAL)
                                # cv2.imshow('imgs', img_ready)
                                k = cv2.waitKey(0)
                                # print(self.pingpongcolor(hsv))
                        elif Number % 2 == 0 and y > self.transverseline[1] - self.transverseline[0]:
                            if y > self.transverseline[1] - self.transverseline[0] + (
                                self.transverseline[2] - self.transverseline[1]) / 3:
                                img_ready = crop_img.copy()
                                img_ready = img_ready[y:y + h, x:x + w]
                                # hsv = cv2.cvtColor(img_ready, cv2.COLOR_BGR2HSV)

                                cv2.rectangle(crop_img, (x, y), (x + w, y + h), (255, 255, 255), 4)
                                # roi_gray = gray[y:y+h, x:x+w]
                                roi_color = crop_img[y:y + h, x:x + w]
                                # if self.pingpongcolor(roi_color):
                                #     return True
                                # print 'pingpong'
                                # return True
                                # cv2.imwrite('96.png', roi_color)
                                cv2.namedWindow('123', cv2.WINDOW_NORMAL)
                                cv2.imshow('123', roi_color)
                                # cv2.namedWindow('imgs', cv2.WINDOW_NORMAL)
                                # cv2.imshow('imgs', img_ready)
                                k = cv2.waitKey(0)
                                # print(self.pingpongcolor(hsv))

            # cv2.line(crop_img, (0, (self.transverseline[1] - self.transverseline[0]) / 2), (1400, (self.transverseline[1] - self.transverseline[0]) / 2),
            # (255, 0, 0), 3, cv2.LINE_AA)
            # cv2.line(crop_img, (0, 820), (1400, 820),(0, 255, 0), 3, cv2.LINE_AA)
            # cv2.line(crop_img, (0, self.transverseline[1] - self.transverseline[0] + (self.transverseline[2] - self.transverseline[1]) / 3),
            # (1400, self.transverseline[1] - self.transverseline[0] + (self.transverseline[2] - self.transverseline[1]) / 3),(0, 0, 255), 3, cv2.LINE_AA)
            return False
        except Exception as e:
            print(e)

    # 对乒乓球进行颜色识别
    def pingpongcolor(self, img):
        GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 二值化操作
        down = 175
        ret, thresh = cv2.threshold(GrayImage, down, 255, cv2.THRESH_BINARY)

        size = 4
        kernel = np.uint8(np.zeros((size, size)))
        for x in range(size):
            kernel[x, 2] = 1
            kernel[2, x] = 1

        # 图像备份
        process = thresh.copy()

        # 图像腐蚀
        for i in range(3):
            process = cv2.erode(process, kernel)

        # 图像膨胀
        for i in range(10):
            process = cv2.dilate(process, kernel)

        # 计算白色比率
        black = 0
        white = 0
        for line in process:
            for pixel in line:
                if pixel == 0:
                    black += 1
                else:
                    white += 1
        rate = white * 1.0 / (white + black)
        print('White Rate: ', rate)

        if rate > 0.6:
            return True
        else:
            return False

    # 乒乓全图匹配函数
    def find_pingpong(self, Position):
        Qu = ['B', 'C', 'D']
        for Item in Qu:
            for order_number in range(1, 13):
                #判断是否是已识别物品位置
                if [Item, order_number] not in Position:
                    if self.pingpong_confirm(Item, order_number):
                        print([Item, order_number, 'white pingpang ball'])
                        # return [Item, order_number, 'white pingpang ball']

    def gaosi(self, img):
        kernel_size = (5, 5)
        sigma = 1.5

        return  cv2.GaussianBlur(img, kernel_size, sigma)

        # for i in range(7, 25):
        #     path = self.photo_path + '%d.png' % i
        #     img = cv2.imread(path)
        #
        #     img = cv2.GaussianBlur(img, kernel_size, sigma)
        #
        #     cv2.imwrite('./pictures/{}.png'.format(i), img)
        #     print('%d.png处理完成'% i)



# 测试函数
if __name__ == "__main__":
    # img = cv2.imread('./picture/1.png')
    # print(type(img))
    vision = RobotCarHandle(gaoso_sign=0)


    # print(vision.jiaduobao_confirm('B', 10))
    # print(vision.jiaduobao_confirm('B', 7))
    # vision.gaosi()
    # print(vision.yangleduo_confirm('C', 8))
    # vision.image_handle_fixed_value()
    # print(vision.return_first_result())
    # print(vision.pencil_confirm('A', 4))
    # vision.open_camera()
    # time.sleep(5)
    # vision.close_camera()
    # vision.find_pencil([])
    # print(vision.pencil_confirm('B', 1))
    # vision.image_handle_fixed_value()
    # print(vision.return_first_result())
    # print(vision.pencil_confirm('D', 7))
    for i in range(7, 25):
        vision.second_process_pipe(i)
    vision.final_process_confirm_pipe()
    vision.return_second_result()
    # p = vision.second_process_queue(0)