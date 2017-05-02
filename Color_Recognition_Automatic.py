# encoding:utf-8
import cv2
import os
import numpy as np
import time
import logging

class Main_Function():
    def __init__(self):
        # 当次识别出小黄块列表
        self.New_Color_materials = []
        # 需抓取小黄块列表
        self.Color_materials = []
        # 黄色小木块初始颜色阈值
        self.lower_yellow = np.array([15, 150, 120])
        self.upper_yellow = np.array([40, 210, 200])

    # 六张图片处理
    def Main_Process(self, path):
        try:
            # 最小过剩剩余方块列表
            Maxmin_New_Color_materials = []
            # 修正次数
            Count = 0

            while len(self.Color_materials) < 6 and Count < 15:
                # 清空当次小黄列表
                self.New_Color_materials = []
                for i in range(1, 7):
                    str = path + '%d.png' % i
                    self.FindColorFools(str, i)
                if len(self.Color_materials) + len(self.New_Color_materials) <= 6:
                    for i in self.New_Color_materials:
                        self.Color_materials.append(i)
                    if len(self.Color_materials) < 6:
                        self.lower_yellow = np.array([15, self.lower_yellow[1] - 5, self.lower_yellow[2] - 5])
                        self.upper_yellow = np.array([40, self.upper_yellow[1] + 5, self.upper_yellow[1] + 5])
                else:
                    self.lower_yellow = np.array([15, self.lower_yellow[1] + 3, self.lower_yellow[2] + 3])
                    self.upper_yellow = np.array([40, self.upper_yellow[1] - 3, self.upper_yellow[1] - 3])
                    if len(Maxmin_New_Color_materials) == 0 or self.New_Color_materials < Maxmin_New_Color_materials:
                        if len(self.New_Color_materials) != 0:
                            Maxmin_New_Color_materials = self.New_Color_materials[:]
                Count += 1
                print 'New Yellow Cube List:', self.New_Color_materials
                print 'Maxmin New Color materials', Maxmin_New_Color_materials
                print 'Yellow Cube List: ', self.Color_materials
                print 'Yellow Color Lower Threshold: ', self.lower_yellow
                print 'Yellow Color Upper Threshold: ', self.upper_yellow
            if len(self.Color_materials) < 6:
                for i in Maxmin_New_Color_materials:
                    if i not in self.Color_materials:
                        self.Color_materials.append(i)
            self.Color_materials = sorted(self.Color_materials)
        except Exception as e:
            print(e)

    # 颜色识别模块
    def FindColorFools(self, img_src, picture_sign):
        start = time.clock()
        img_all = cv2.imread(img_src)
        sp = img_all.shape
        verticalline = [673, 1900]
        transverseline = [100, 940, 1944]
        try:
            hsvs = cv2.cvtColor(img_all, cv2.COLOR_BGR2HSV)
            for i in range(2):
                img = img_all.copy()
                if i == 0:
                    for Item in self.Color_materials:
                        if Item[1] == 2 * picture_sign - 1:
                            return
                    crop_img = img[int(transverseline[0]):int(transverseline[1]),int(verticalline[0]):int(verticalline[1])]
                    hsv = hsvs[int(transverseline[0]):int(transverseline[1]),int(verticalline[0]):int(verticalline[1])]
                    mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
                    imgage, contours, hi = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    if contours:
                        a = contours[0]
                        x, y, w, h = cv2.boundingRect(a)
                        for cnt in contours:
                            x, y, w, h = cv2.boundingRect(cnt)
                            if w + h > 100:
                                #Modified - Variable From Color_materials to self.New_Color_materials
                                self.New_Color_materials.append(['A', 2 * picture_sign - 1 + i, "yellow cube"])
                                break
                else:
                    for Item in self.Color_materials:
                        if Item[1] == 2 * picture_sign:
                            return
                    crop_img = img[int(transverseline[1]):int(transverseline[2]),int(verticalline[0]):int(verticalline[1])]
                    hsv = hsvs[int(transverseline[1]):int(transverseline[2]),int(verticalline[0]):int(verticalline[1])]
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

        end = time.clock()
        print('Running time: %s Seconds' % (end - start))

    # 返回A区结果
    def return_result(self):
        print self.Color_materials

# 测试函数
if __name__ == "__main__":
    vision = Main_Function()
    vision.Main_Process('./picture/')
    vision.return_result()