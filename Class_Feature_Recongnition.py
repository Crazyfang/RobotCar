# encoding:utf-8
import cv2
import os
import numpy as np
import threading
import time
import logging
from shopping_robot import ShoppingRobot
from neural_network import NeuralNetwork
import math
from multiprocessing import Manager, Process, Lock

class Main_Function():
    def __init__(self):
        # 拍摄前延时时间
        self.Taking_Photo_Time = 0.25
        # 每次拍摄次数
        self.Taking_Photo_Times = 2
        # 图像存储路径
        self.Photo_Path = './picture/'
        # 黄色小木块初始颜色阈值
        self.lower_yellow = np.array([15, 150, 120])
        self.upper_yellow = np.array([40, 210, 200])
        # 颜色调整值
        self.color_plus_adjust_value = 5
        self.color_sub_adjust_value = 3
        # 当次识别出小黄块列表
        self.New_Color_materials = []
        # 需抓取小黄块列表
        self.Color_materials = []
        # 图片切割信息
        self.verticalline = [500, 2000]
        self.transverseline = [140, 960, 1944]
        # NeuralNetwork
        # self.Nn = NeuralNetwork()
        # 切图保存路径
        self.Cut_Pic_Path = './picture/Cut/'
        # 图片预处理列表 [识别度, 物品名称, 照片序号， 上下编号{上-0|下-1}, 验证结果{True|False}]
        self.Pre_Items = []
        # 已处理图片数量(Max to 36)
        self.Process_Number = 0
        # B,C,D区物品列表
        self.Second_data = []
        # xml模板所在文件夹
        self.Xml_Path = './cascade/'
        # 查询列表
        # self.manager = Manager()
        # self.Ready_To_Search = self.manager.list()
        # 查询进程结束标志
        self.IsFinish = True
        # 临时文件锁
        self.temp_lock = Lock()
        # 临时文件
        self.TempFile = './files/data.tmp'
        # 结果文件
        self.ResultFile = './files/data.res'
        # 完成文件
        self.FinishFile = './files/Finish'

    #　开启摄像头
    def Open_Camera(self):
        self.cap = cv2.VideoCapture(1)

        #设置图像尺寸
        self.cap.set(3,3220)
        self.cap.set(4,2440)
        
        # 打开摄像头
        if not self.cap.isOpened():
            self.cap.open()

        # 预拍照
        ret, img = self.cap.read()
        cv2.imwrite(self.Photo_Path + 'temp.png', img)

        # 显示摄像头打开成功信息
        if self.cap.isOpened():
            print('Open Camera Successful')

    # 开启摄像头拍照
    def Camera_TakePhoto(self, order_number):
        # 显示正在拍照信息
        print('Taking photo')

        # 拍摄并保存图像
        for i in range(0, self.Taking_Photo_Times):
            ret, img = self.cap.read()
            time.sleep(self.Taking_Photo_Time)
            cv2.imwrite(self.Photo_Path + '%s.png' % order_number, img)

    # 关闭摄像头
    def Close_Camera(self):
        print('关闭摄像头')
        self.cap.release()

    # A区图片处理
    def First_Process(self):
        try:
            # 最小过剩剩余方块列表
            Maxmin_New_Color_materials = []
            # 修正次数
            Count = 0

            while len(self.Color_materials) < 6 and Count < 15:
                # 清空当次小黄列表
                self.New_Color_materials = []
                
                # 识别A区图片
                for i in range(1, 7):
                    str = self.Photo_Path + '%d.png' % i
                    self.FindColorFools(str, i)

                # 判断当前识别列表
                if len(self.Color_materials) + len(self.New_Color_materials) <= 6:
                    for i in self.New_Color_materials:
                        self.Color_materials.append(i)
                    if len(self.Color_materials) < 6:
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
                    if len(Maxmin_New_Color_materials) == 0 or self.New_Color_materials < Maxmin_New_Color_materials:
                        if len(self.New_Color_materials) != 0:
                            Maxmin_New_Color_materials = self.New_Color_materials[:]

                # 识别次数增加1
                Count += 1

                # 输出信息
                print('-------Loop %d Result-------' % Count)
                print('Yellow Cube List: '),
                print(self.Color_materials)
                print('Yellow Color Threshold: '),
                print(self.lower_yellow),
                print(self.upper_yellow)

            # 增加最小剩余方块列表
            if len(self.Color_materials) < 6:
                for Item in Maxmin_New_Color_materials:
                    if Item not in self.Color_materials:
                        self.Color_materials.append(Item)

            # 对抓取列表进行排序
            self.Color_materials = sorted(self.Color_materials)
            print('----------Yellow Cube Result----------')
            print(self.Color_materials)
        except Exception as e:
            print(e)

    # 颜色识别模块
    def FindColorFools(self, img_src, picture_sign):
        # 读取图片
        img_all = cv2.imread(img_src)
        try:
            hsvs = cv2.cvtColor(img_all, cv2.COLOR_BGR2HSV)

            # 上下货栏
            for i in range(2):
                img = img_all.copy()
                if i == 0: #上栏
                    #判断是否已经在抓取列表中
                    for Item in self.Color_materials:
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
                                self.New_Color_materials.append(['A', 2 * picture_sign - 1 + i, "yellow cube"])
                                break
                else: #下栏
                    #判断是否已经在抓取列表中
                    for Item in self.Color_materials:
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
                                self.New_Color_materials.append(['A', 2 * picture_sign - 1 + i, "yellow cube"])
                                break
        except Exception as e:
            print(e)

    # 返回A区结果
    def Return_First_Result(self):
        print(self.Color_materials)
        return self.Color_materials

    # B,C,D区图片处理
    def Second_Process(self, order_number):
        print('\033[0;37;40m')

        # 启动进程
        if self.IsFinish:

            if os.path.exists(self.TempFile):
                os.remove(self.TempFile)
            if os.path.exists(self.ResultFile):
                os.remove(self.ResultFile)
            if os.path.exists(self.FinishFile):
                os.remove(self.FinishFile)
            self.IsFinish = False
            p = Process(target = self.Start_Search, args=(self.TempFile,))
            p.daemon = True
            p.start()

        path = self.Photo_Path + '%d.png' % order_number
        all_img = cv2.imread(path)

        # 保存上半张图片
        img = all_img.copy()
        crop_img = img[int(self.transverseline[0]):int(self.transverseline[1]),int(self.verticalline[0]):int(self.verticalline[1])]
        cv2.imwrite(self.Cut_Pic_Path + '%d_0.png' % order_number, crop_img)
        # 输出图片保存信息
        print('Picture: %s%d_0.png Save Successful' % (self.Cut_Pic_Path, order_number))
        # 添加进程任务
        self.temp_lock.acquire()
        try:
            # self.Ready_To_Search.append([order_number, 0])
            print('Write To File: %d-0' % (order_number))
            fs = open(self.TempFile, 'a+')
            fs.write("%d-0\n" % order_number)
            fs.close()
        finally:
            self.temp_lock.release()

        # 保存下半张图片
        img = all_img.copy()
        crop_img = img[int(self.transverseline[1]):int(self.transverseline[2]),int(self.verticalline[0]):int(self.verticalline[1])]
        cv2.imwrite(self.Cut_Pic_Path + '%d_1.png' % order_number, crop_img)
        # 输出图片保存信息
        print('Picture: %s%d_1.png Save Successful' % (self.Cut_Pic_Path, order_number))
        # 添加进程任务
        self.temp_lock.acquire()
        try:
            # self.Ready_To_Search.append([order_number, 1])
            print('Write To File: %d-1' % (order_number))
            fs = open(self.TempFile, 'a+')
            fs.write("%d-1\n" % order_number)
            fs.close()
        finally:
            self.temp_lock.release()

    # 查询列表并组织识别进程函数
    def Start_Search(self, tempFile):
        print("Recongnition Process Start")
        # 已经查询列表
        Now_List = []
        v = NeuralNetwork()
        while True:
            if os.path.exists(self.FinishFile):
                print("Recongnition Process End")
                break

            if not os.path.exists(self.TempFile):
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
            tmpFile = open(tempFile)
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
            self.Tf_Confirm(int(Item[0]), int(Item[1]), v)

    # 用TranserFlow识别并验证
    def Tf_Confirm(self, order_number, position_number, v):
        print('开始处理%d-%d数据' % (order_number, position_number))

        name, confidence = v.single_recognize(self.Cut_Pic_Path + '%d_%d.png' % (order_number, position_number))
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
            elif name == 'tennis ball':
                Result = [confidence, name, order_number, position_number, True]
            elif name == 'white pingpang ball':
                Result = [confidence, name, order_number, position_number, True]
            else:
                #需要验证的列表
                try:
                    #[识别度, 物品名称, 照片序号， 上下编号{上-0|下-1}] 转 [区号, 编号, 物品名称]
                    FItem = self.CalListContent([confidence, name, order_number, position_number])

                    # if name == 'white pingpang ball':
                    #     if self.Pingpong_Confirm(FItem[0], FItem[1]):
                    #         self.Pre_Items.append([confidence, name, order_number, position_number, True])
                    #     else
                    #         self.Pre_Items.append([confidence, name, order_number, position_number, False])
                    # elif name == 'gold jia duo bao':
                    if name == 'gold jia duo bao':
                        if self.Jiaduobao_Confirm(FItem[0], FItem[1]):
                            Result = [confidence, name, order_number, position_number, True]
                        else:
                            Result = [confidence, name, order_number, position_number, False]
                    elif name == 'mimi':
                        if self.Mimi_Confirm(FItem[0], FItem[1]):
                            Result = [confidence, name, order_number, position_number, True]
                        else:
                            Result = [confidence, name, order_number, position_number, False]
                    elif name == 'yang le duo':
                        if self.Yangleduo_Confirm(FItem[0], FItem[1]):
                            Result = [confidence, name, order_number, position_number, True]
                        else:
                            Result = [confidence, name, order_number, position_number, False]
                    elif name == 'zhong hua pencil':
                        if self.Pencil_Confirm(FItem[0], FItem[1]):
                            print('铅笔验证通过')
                            Result = [confidence, name, order_number, position_number, True]
                        else:
                            print('铅笔验证失败')
                            Result = [confidence, name, order_number, position_number, False]
                except Exception as e:
                    print(e)

        if Result != []:
            # self.Pre_Items.append(Result)
            fs = open(self.ResultFile, 'a+')
            fs.write("%f-%s-%d-%d-%s\n" % (Result[0], Result[1], Result[2], Result[3], Result[4]))
            # fs.write("%f-%s-%d-%d-%s\n" % (confidence, name, order_number, position_number, True))
            fs.close()

        print('%d-%d数据处理完成' % (order_number, position_number))

        if order_number == 24 and position_number == 1:
            fin = open(self.FinishFile, 'w')
            fin.close()

    # 返回B,C,D区结果
    def Return_Second_Result(self):
        print(self.Second_data)
        return self.Second_data

    # 最终处理
    def Final_Process(self):
        # 等待线程处理结束
        # while Count < 6:
        #     if not os.path.exists(self.ResultFile):
        #         time.sleep(1)
        #         continue
        #     Count = 0
        #     tmpFile = open(self.ResultFile)
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
        while not os.path.exists(self.FinishFile):
            time.sleep(0.5)

        tmpFile = open(self.ResultFile)
        try:
            while True:
                str = tmpFile.readline()[:-1]
                if str == '':
                    break
                Item = str.split('-', 4)
                self.Pre_Items.append([float(Item[0]), Item[1], int(Item[2]), int(Item[3]), Item[4] == 'True'])
        finally:
            tmpFile.close()

        # while len(self.Pre_Items) != 36:
        #     time.sleep(0.3)

        # 对结果排序
        Lists = sorted(self.Pre_Items, reverse=True)
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
            FItem = self.CalListContent(Item[:-1])

            if Item[1] in Already_Get:
                continue
            elif Item[4]:
                return_list.append(FItem)
                Already_Get.append(Item[1])
                Already_Get_Position.append(FItem[:-1])

        #判断中华铅笔是已获得，否则全照片判断铅笔位置
        if not 'zhong hua pencil' in Already_Get:
            try:
                Pencil = self.Find_Pencil(Already_Get_Position)
                if Pencil:
                    return_list.append(Pencil)
                    Already_Get.append('zhong hua pencil')
                    Already_Get_Position.append(Pencil[:-1])
            except Exception as e:
                print(e)

        #判断养乐多是已获得，否则全照片判断养乐多位置
        if not 'yang le duo' in Already_Get:
            try:
                Yangleduo = self.Find_Yangleduo(Already_Get_Position)
                if Yangleduo:
                    return_list.append(Yangleduo)
                    Already_Get.append('yang le duo')
                    Already_Get_Position.append(Yangleduo[:-1])
            except Exception as e:
                print(e)

        #判断咪咪是已获得，否则全照片判断咪咪位置
        if not 'mimi' in Already_Get:
            try:
                Mimi = self.Find_Mimi(Already_Get_Position)
                if Mimi:
                    return_list.append(Mimi)
                    Already_Get.append('mimi')
                    Already_Get_Position.append(Mimi[:-1])
            except Exception as e:
                print(e)

        #判断加多宝是已获得，否则全照片判断加多宝位置
        if not 'gold jia duo bao' in Already_Get:
            try:
                Jiaduobao = self.Find_Jiaduobao(Already_Get_Position)
                if Jiaduobao:
                    return_list.append(Jiaduobao)
                    Already_Get.append('gold jia duo bao')
                    Already_Get_Position.append(Jiaduobao[:-1])
            except Exception as e:
                print(e)

        # # 判断乒乓是已获得，否则全照片判断乒乓位置
        # if not 'white pingpang ball' in Already_Get:
        #     try:
        #         Pingpong = self.Find_Pingpong(Already_Get_Position)
        #         if Pingpong:
        #             return_list.append(Pingpong)
        #             Already_Get.append('white pingpang ball')
        #             Already_Get_Position.append(Pingpong[:-1])
        #     except Exception as e:
        #         print(e)

        # # 判断网球是已获得，否则全照片判断网球位置
        # if not 'tennis ball' in Already_Get:
        #     try:
        #         Tennis = self.Tennis_Confirm(Already_Get_Position)
        #         if Tennis:
        #             return_list.append(Tennis)
        #             Already_Get.append('tennis ball')
        #             Already_Get_Position.append(Tennis[:-1])
        #     except Exception as e:
        #         print(e)
        
        #已排序抓取列表
        sorted_return_list = []
        
        #首先加入中华铅笔
        for Item in return_list:
            if Item[2] == 'zhong hua pencil':
                sorted_return_list.append(Item)
                return_list.remove(Item)
                break

        #已抓取列表倒序排列
        return_list = sorted(return_list, reverse = True)

        #已抓取列表中物品加入已排序抓取列表
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
        self.Second_data = sorted_return_list[:]

    # 列表转换函数 - [识别度, 物品名称, 照片序号， 上下编号{上-0|下-1}] 转 [区号, 编号, 物品名称]
    def CalListContent(self, Item):
        if (Item[2] - 7) // 6 == 0:
            quhao = 'B'
        elif (Item[2] - 7) // 6 == 1:
            quhao = 'C'
        else:
            quhao = 'D'
        return [quhao, (Item[2] - 7) % 6 * 2 + Item[3] + 1, Item[1]]

    # 铅笔确认函数
    def Pencil_Confirm(self, quhao, order_number):
        # 区域码
        print('铅笔验证开始')
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

        img_all = cv2.imread(self.Photo_Path + '%s.png' % str((Number + 1) // 2))
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
                        if adds >= 8:
                            flag = 1
                            break
                        else:
                            adds = 0
                            old_data = []
                            break
                if adds >= 8:
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
                    lower_green = np.array([50, 40, 46])
                    upper_green = np.array([110, 209, 170])
                    mask = cv2.inRange(hsv, lower_green, upper_green)
                    img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    if contours:
                        a = contours[0]
                        x, y, w, h = cv2.boundingRect(a)

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
                # cv2.namedWindow('source', cv2.WINDOW_NORMAL)
                # cv2.imshow("source", src)
                # cv2.namedWindow('lines', cv2.WINDOW_NORMAL)
                # cv2.imshow("lines", cdst)
                # cv2.waitKey(0)
            return False
        except:
            return False

    # 铅笔全图匹配函数
    def Find_Pencil(self, Position):
        Qu = ['B', 'C', 'D']
        for Item in Qu:
            for order_number in range(1, 13):
                #判断是否是已识别物品位置
                if [Item, order_number] not in Position:
                    if self.Pencil_Confirm(Item, order_number):
                        return [Item, order_number, 'zhong hua pencil']

    # 咪咪虾条确认函数
    def Mimi_Confirm(self, quhao, order_number):
        Mimi_Template_Image = ["./model/92.png"]
        Yuzhi = [150]

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
            img_all = cv2.imread(self.Photo_Path + '%s.png' % str((Number + 1) // 2))
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

            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)

            good = []

            for m, n in matches:
                if m.distance < 0.80 * n.distance:
                    good.append(m)

            if len(good) >= MIN_MATCH_COUNT:
                return True
            else:
                return False
        except Exception as e:
            print(e)

    # 咪咪全图匹配函数
    def Find_Mimi(self, Position):
        Qu = ['B', 'C', 'D']
        for Item in Qu:
            for order_number in range(1, 13):
                #判断是否是已识别物品位置
                if [Item, order_number] not in Position:
                    if self.Mimi_Confirm(Item, order_number):
                        return [Item, order_number, 'mimi']

    # 养乐多确认函数
    def Yangleduo_Confirm(self, quhao, order_number):
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
            img_all = cv2.imread(self.Photo_Path + '%s.png' % str((Number + 1) // 2))
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
    def Find_Yangleduo(self, Position):
        Qu = ['B', 'C', 'D']
        for Item in Qu:
            for order_number in range(1, 13):
                #判断是否是已识别物品位置
                if [Item, order_number] not in Position:
                    if self.Yangleduo_Confirm(Item, order_number):
                        return [Item, order_number, 'yang le duo']
                        # print([Item, order_number, 'yang le duo'])
                        # pass

    # 加多宝确认函数
    def Jiaduobao_Confirm(self, quhao, order_number):
        Jiaduobao_Template_Image = ["./model/111.jpg"]
        Yuzhi = [90]

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
            img_all = cv2.imread(self.Photo_Path + '%s.png' % str((Number + 1) // 2))
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
    def Find_Jiaduobao(self, Position):
        Qu = ['B', 'C', 'D']
        for Item in Qu:
            for order_number in range(1, 13):
                #判断是否是已识别物品位置
                if [Item, order_number] not in Position:
                    if self.Jiaduobao_Confirm(Item, order_number):
                        print([Item, order_number, 'gold jia duo bao'])
                        return [Item, order_number, 'gold jia duo bao']

    # 乒乓确认函数 - 效果不好,需要加颜色检测
    def Pingpong_Confirm(self, quhao, order_number):
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
            img_all = cv2.imread(self.Photo_Path + '%s.png' % str((Number + 1) // 2))
            img = img_all.copy()

            crop_img = img[int(self.transverseline[0]):int(self.transverseline[2]), int(self.verticalline[0]):int(self.verticalline[1])]

            sp = crop_img.shape
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(self.Xml_Path + 'pingpong.xml')
            faces = face_cascade.detectMultiScale(gray, 1.25, 5)
            for (x,y,w,h) in faces:
                if w+h > 200 and w+h < 400:    #针对这个图片画出最大的外框
                    if x > sp[1] / 4 and x < sp[1] / 4 * 3:
                        if Number % 2 == 1 and y < self.transverseline[1] - self.transverseline[0]:
                            if y > (self.transverseline[1] - self.transverseline[0]) / 2:
                                # cv2.rectangle(crop_img,(x,y),(x+w,y+h),(255,255,255),4)
                                # roi_gray = gray[y:y+h, x:x+w]
                                # roi_color = crop_img[y:y+h, x:x+w]
                                # print 'pingpong'
                                return True
                        elif Number % 2 == 0 and y > self.transverseline[1] - self.transverseline[0]:
                            if y > self.transverseline[1] - self.transverseline[0] + (self.transverseline[2] - self.transverseline[1]) / 3:
                                # cv2.rectangle(crop_img,(x,y),(x+w,y+h),(255,255,255),4)
                                # roi_gray = gray[y:y+h, x:x+w]
                                # roi_color = crop_img[y:y+h, x:x+w]
                                # print 'pingpong'
                                return True
            # cv2.line(crop_img, (0, (self.transverseline[1] - self.transverseline[0]) / 2), (1400, (self.transverseline[1] - self.transverseline[0]) / 2),(255, 0, 0), 3, cv2.LINE_AA)
            # cv2.line(crop_img, (0, 820), (1400, 820),(0, 255, 0), 3, cv2.LINE_AA)
            # cv2.line(crop_img, (0, self.transverseline[1] - self.transverseline[0] + (self.transverseline[2] - self.transverseline[1]) / 3), (1400, self.transverseline[1] - self.transverseline[0] + (self.transverseline[2] - self.transverseline[1]) / 3),(0, 0, 255), 3, cv2.LINE_AA)
            # cv2.namedWindow('imgs', cv2.WINDOW_NORMAL)
            # cv2.imshow('imgs',crop_img)
            # k = cv2.waitKey(0)

            return False
        except Exception as e:    
            print(e)

    # 乒乓全图匹配函数
    def Find_Pingpong(self, Position):
        Qu = ['B', 'C', 'D']
        for Item in Qu:
            for order_number in range(1, 13):
                #判断是否是已识别物品位置
                if [Item, order_number] not in Position:
                    if self.Pingpong_Confirm(Item, order_number):
                        # print([Item, order_number, 'white pingpang ball'])
                        return [Item, order_number, 'white pingpang ball']

    # 网球确认函数 - 效果不好,颜色检测部分需要修改
    def Tennis_Confirm(self, quhao, order_number):
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
            img_all = cv2.imread(self.Photo_Path + '%s.png' % str((Number + 1) // 2))
            img = img_all.copy()

            crop_img = img[int(self.transverseline[0]):int(self.transverseline[2]), int(self.verticalline[0]):int(self.verticalline[1])]

            sp = crop_img.shape
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(self.Xml_Path + 'tennis.xml')
            faces = face_cascade.detectMultiScale(gray, 1.15, 5)
            for (x,y,w,h) in faces:
                if w+h > 250 and w+h < 800:
                    if x > sp[1] / 5 and x < sp[1] / 5 * 4:
                        if Number % 2 == 1 and y < self.transverseline[1] - self.transverseline[0]:
                            if y > (self.transverseline[1] - self.transverseline[0]) / 4:
                                # cv2.rectangle(crop_img,(x,y),(x+w,y+h),(255,255,255),4)
                                # roi_gray = gray[y:y+h, x:x+w]
                                # roi_color = crop_img[y:y+h, x:x+w]
                                srcs = crop_img[x:y, x+w:y+h]
                                hsv = cv2.cvtColor(srcs, cv2.COLOR_BGR2HSV)
                                lower_green = np.array([50, 40, 46])
                                upper_green = np.array([90, 209, 170])
                                mask = cv2.inRange(hsv, lower_green, upper_green)
                                img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                                if contours:
                                    a = contours[0]
                                    x, y, w, h = cv2.boundingRect(a)

                                    for cnt in contours:
                                        x, y, w, h = cv2.boundingRect(cnt)
                                        if w + h > 200:
                                            # print 'Tennis'
                                            # cv2.rectangle(crop_img,(x,y),(x+w,y+h),(255,0,0),10)
                                            # roi_gray = gray[y:y+h, x:x+w]
                                            # roi_color = crop_img[y:y+h, x:x+w]
                                            return True
                        elif Number % 2 == 0 and y > self.transverseline[1] - self.transverseline[0]:
                            if y > self.transverseline[1] - self.transverseline[0] + (self.transverseline[2] - self.transverseline[1]) / 4:
                                srcs = crop_img[ x:y, x+w:y+h]
                                hsv = cv2.cvtColor(srcs, cv2.COLOR_BGR2HSV)
                                lower_green = np.array([50, 40, 46])
                                upper_green = np.array([90, 209, 170])
                                mask = cv2.inRange(hsv, lower_green, upper_green)
                                img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                                if contours:
                                    a = contours[0]
                                    x, y, w, h = cv2.boundingRect(a)
                                    for cnt in contours:
                                        x, y, w, h = cv2.boundingRect(cnt)
                                        if w + h > 200:
                                            # print 'Tennis'
                                            # cv2.rectangle(crop_img,(x,y),(x+w,y+h),(255,0,0),10)
                                            # roi_gray = gray[y:y+h, x:x+w]
                                            # roi_color = crop_img[y:y+h, x:x+w]
                                            return True
            # cv2.line(crop_img, (0, 820), (1400, 820),(0, 255, 0), 3, cv2.LINE_AA)
            # cv2.namedWindow('imgs', cv2.WINDOW_NORMAL)
            # cv2.imshow('imgs',crop_img)
            # k = cv2.waitKey(0)

            return False
        except Exception as e:
            print(e)

    # 网球全图匹配函数
    def Find_Tennis(self, Position):
        Qu = ['B', 'C', 'D']
        for Item in Qu:
            for order_number in range(1, 13):
                #判断是否是已识别物品位置
                if [Item, order_number] not in Position:
                    if self.Tennis_Confirm(Item, order_number):
                        # print([Item, order_number, 'tennis ball'])
                        return [Item, order_number, 'tennis ball']

# 测试函数
if __name__ == "__main__":
    vision = Main_Function()

    # for i in range(7, 25):
    #     vision.Second_Process(i)
    # vision.Tf_Confirm(7, 1)
    # 对结果排序
    # Lists = sorted(vision.Pre_Items,reverse=True)
    # 输出识别结果
    # vision.Final_Process()
    vision.Find_Jiaduobao([])
    # print(vision.Pencil_Confirm("C", 9))
    # vision.IsFinish = True