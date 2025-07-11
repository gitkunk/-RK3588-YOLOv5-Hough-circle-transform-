'''
 ┌─────────────────────────────────────────────────────────────┐
 │┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
 ││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
 │├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
 ││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
 │├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
 ││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
 │├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
 ││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
 │└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
 │      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
 │      └───┴─────┴───────────────────────┴─────┴───┘          │
 └─────────────────────────────────────────────────────────────┘

Author: lucas
Date: 2022-05-13 00:03:00
LastEditTime: 2022-10-16 12:57:01
LastEditors: lucas
Description: 仪表识别核心
FilePath: \MeterReadV2\MeterClass.py
CSDN:https://blog.csdn.net/qq_27545821?spm=1000.2115.3001.5343
github: https://github.com/glasslucas00?tab=repositories
'''
from math import sqrt
import cv2
import numpy as np
import os
import random 
import glob

#基本方法工具
class Functions:
    @staticmethod
    def GetClockAngle(v1, v2): 
         # 2个向量模的乘积 ,返回夹角
        TheNorm = np.linalg.norm(v1)*np.linalg.norm(v2)
        # 叉乘
        rho = np.rad2deg(np.arcsin(np.cross(v1, v2)/TheNorm))
        # 点乘
        theta = np.rad2deg(np.arccos(np.dot(v1,v2)/TheNorm))
        if rho > 0:
            return  360-theta
        else:
            return theta
    @staticmethod
    def Disttances(a, b):
        #返回两点间距离
        x1, y1 = a
        x2, y2 = b
        Disttances = int(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        return Disttances

    @staticmethod
    def couputeMean(deg):
        #对数据进行处理，提取均值
        """
        :funtion :
        :param b:
        :param c:
        :return:
        """
        if (True):
            # new_nums = list(set(deg)) #剔除重复元素
            mean = np.mean(deg)
            var = np.var(deg)
            # print("原始数据共", len(deg), "个\n", deg)
            '''
            for i in range(len(deg)):
                print(deg[i],'→',(deg[i] - mean)/var)
                #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据
            '''
            # print("中位数:",np.median(deg))
            percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
            # print("分位数：", percentile)
            # 以下为箱线图的五个特征值
            Q1 = percentile[0]  # 上四分位数
            Q3 = percentile[2]  # 下四分位数
            IQR = Q3 - Q1  # 四分位距
            ulim = Q3 + 2.5 * IQR  # 上限 非异常范围内的最大值
            llim = Q1 - 1.5 * IQR  # 下限 非异常范围内的最小值

            new_deg = []
            uplim = []
            for i in range(len(deg)):
                if (llim < deg[i] and deg[i] < ulim):
                    new_deg.append(deg[i])
            # print("清洗后数据共", len(new_deg), "个\n", new_deg)
        new_deg = np.mean(new_deg)

        return new_deg

#检测方法
class MeterDetection:
    def __init__(self,path):
        self.k = None               #斜率
        self.image=path
        self.height_width_channels = self.image.shape
        self.circleimg=None
        self.panMask=None           #霍夫圆检测切割的表盘图片
        self.poniterMask =None      #指针图片
        self.numLineMask=None       #刻度线图片
        self.centerPoint=None       #中心点[x,y]
        self.farPoint=None          #指针端点[x,y]
        self.zeroPoint = [0, 0]     # 0刻度起始点[x,y]
        self.finalPoint = [0, 0]    # 终止刻度点[x,y]
        self.kmax = 0
        self.kmin = 0
        self.r=None                 #半径
        self.divisionValue=100/270  #分度值

    def ImgCutCircle(self):
        #截取表盘区域，滤除背景
        img=self.image
        dst = cv2.pyrMeanShiftFiltering(img, 10, 100)
        cimage = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(cimage, cv2.HOUGH_GRADIENT, 1, 80, param1=100, param2=20, minRadius=80, maxRadius=0)
        circles = np.uint16(np.around(circles))  # 把类型换成整数
        r_1 = circles[0, 0, 2]
        c_x = circles[0, 0, 0]
        c_y = circles[0, 0, 1]#cv2.HoughCircles的返回值，分别是(圆心横坐标，圆心纵坐标，半径)
        circle = np.ones(img.shape, dtype="uint8")
        circle = circle * 255
        cv2.circle(circle, (c_x, c_y), int(r_1), 0, -1)#画圆
        bitwiseOr = cv2.bitwise_or(img, circle)
        self.cirleData = [r_1, c_x, c_y]
        self.panMask=bitwiseOr
       
        return bitwiseOr

    def ContoursFilter(self):
        #对轮廓进行筛选
        """
        :funtion : 提取刻度线，指针
        :param a: 高斯滤波 GaussianBlur，自适应二值化adaptiveThreshold，闭运算
        :param b: 轮廓寻找 findContours，
        :return:lineSet,new_needleset
        """
        r_1, c_x, c_y = self.cirleData

        img = self.panMask.copy()#上一步cv2.bitwise_or处理之后的图像
        img = cv2.GaussianBlur(img, (3, 3), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(~gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)
        
        #轮廓查找，根据版本不同，返回参数不同
        if cv2.__version__ >'4.0.0':
            contours, hier = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            aa,contours, hier = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cntset = []  # 刻度线轮廓集合
        cntareas = []  # 刻度线面积集合

        needlecnt = []  # 指针轮廓集合
        needleareas = []  # 指针面积集合
        radiusLength = [r_1 * 0.6, r_1 * 0.8] # 半径范围

        localtion = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)#在感兴趣区域周围画最小面积矩形框
            # print(rect)
            #（中心点坐标，（宽度，高度）,旋转的角度）=   = rect
            a, (w, h), c = rect  
            w = int(w)
            h = int(h)
            ''' 满足条件:“长宽比例”，“面积”'''
            if h == 0 or w == 0:
                pass
            else:
                dis = Functions.Disttances((c_x, c_y), a)#计算轮廓外接矩形的中心坐标a与圆心点c_x, c_y之间的距离
                if (radiusLength[0] < dis and radiusLength[1] > dis):# 如果距离0.6*半径<dis<半径 radiusLength = [r_1 * 0.6, r_1 * 1] # 半径范围
                    #矩形筛选
                    if h / w > 4 or w / h > 4:
                        localtion.append(dis)
                        cntset.append(cnt)#刻度线轮廓 集合
                        cntareas.append(w * h)#刻度线面积集合
                else:
                    if w > r_1 / 2 or h > r_1 / 2:#如果0.5*半径<dis<0.6*半径，指针的特征?
                        needlecnt.append(cnt)
                        needleareas.append(w * h)
        cntareas = np.array(cntareas)#np.array()的作用就是把列表转化为数组，也可以说是用来产生数组。
        areasMean = Functions.couputeMean(cntareas)  # 中位数，上限区
        new_cntset = []
        # 面积
        for i, cnt in enumerate(cntset):#刻线轮廓集合cntset，这句话是返回cntset的索引给i,具体元素值给cnt
            if (cntareas[i] <= areasMean * 5.2 and cntareas[i] >= areasMean * 0.8):#刻度线的面积在中位数的0.8倍到1.5倍之间
                new_cntset.append(cnt)#将满足条件的新点给 new_cntset

        self.r = np.mean(localtion)#self.r的作用是什么
        mask = np.zeros(img.shape[0:2], np.uint8)
        self.poniterMask = cv2.drawContours(mask, needlecnt, -1, (255, 255, 255), -1)  # 生成掩膜，画出needlecnt指针掩膜
        mask = np.zeros(img.shape[0:2], np.uint8)
        self.numLineMask = cv2.drawContours(mask, new_cntset, -1, (255, 255, 255), -1)  # 生成刻度线掩膜
        self.new_cntset=new_cntset
        
        return new_cntset
    
    def FitNumLine(self):
        """ 轮廓拟合直线"""
        lineSet = []  # 拟合线集合
        img=self.image.copy()

        height,width,channels = self.height_width_channels

        for cnt in self.new_cntset:
            rect = cv2.minAreaRect(cnt)
            # 获取矩形四个顶点，浮点型
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.polylines(img, [box], True, (0, 255, 0), 1)  # pic，这句话作用是什么？
            output = cv2.fitLine(cnt, 2, 0, 0.001, 0.001)#拟合直线函数，输出output为4维，前两维代表拟合出的直线的方向，后两位代表直线上的一点
#拟合后直线点的斜率k和偏移b
            k = output[1] / output[0]
            k = round(k[0], 2)
            b = output[3] - k * output[2]
            b = round(b[0], 2)
            x1 = 1
            x2 = img.shape[0]
            y1 = int(k * x1 + b)
            y2 = int(k * x2 + b)
            # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            #lineSet:刻度线拟合直线数组，k斜率 b
            lineSet.append([k, b])  # 求中心点的点集[k,b]
            if box[0][0] < width / 2 and box[1][0] < width / 2 and box[2][0] < width / 2 and box[3][0] < width / 2:
                if self.kmax > k:
                    self.kmax = k
                    if box[0][1] > box[1][1] and box[0][1] > box[2][1] and box[0][1] > box[3][1]:
                        self.zeroPoint = box[0]
                    if box[1][1] > box[0][1] and box[1][1] > box[2][1] and box[1][1] > box[3][1]:
                       self.zeroPoint = box[1]
                    if box[2][1] > box[3][1] and box[2][1] > box[0][1] and box[2][1] > box[1][1]:
                       self.zeroPoint = box[2]
                    if box[3][1] > box[2][1] and box[3][1] > box[1][1] and box[3][1] > box[0][1]:
                       self.zeroPoint = box[3]

            if box[0][0] > width / 2 and box[1][0] > width / 2 and box[2][0] > width / 2 and box[3][0] > width / 2:
                if self.kmin < k:
                    self.kmin = k
                    if box[0][1] > box[1][1] and box[0][1] > box[2][1] and box[0][1] > box[3][1]:
                       self.finalPoint = box[0]
                    if box[1][1] > box[0][1] and box[1][1] > box[2][1] and box[1][1] > box[3][1]:
                       self.finalPoint = box[1]
                    if box[2][1] > box[3][1] and box[2][1] > box[0][1] and box[2][1] > box[1][1]:
                       self.finalPoint = box[2]
                    if box[3][1] > box[2][1] and box[3][1] > box[1][1] and box[3][1] > box[0][1]:
                       self.finalPoint = box[3]
        self.lineSet=lineSet
        return lineSet

    def getIntersectionPoints(self):
        #获取刻度线交点
        img = self.image
        lineSet=self.lineSet
        w, h, c = img.shape
        xlist=[]
        ylist=[]
        if len(lineSet) > 2:
            np.random.shuffle(lineSet)
            lkb = int(len(lineSet) / 2)
            kb1 = lineSet[0:lkb]
            kb2 = lineSet[lkb:(2 * lkb)]
            kb1sample = random.sample(kb1, int(len(kb1) / 2))
            kb2sample = random.sample(kb2, int(len(kb2) / 2))
        else:
            kb1sample = lineSet[0]
            kb2sample = lineSet[1]
        for i, wx in enumerate(kb1sample):
            for wy in kb2sample:
                k1, b1 = wx
                k2, b2 = wy
                try:
                    if (b2 - b1) == 0:
                        b2 = b2 - 0.1
                    if (k1 - k2) == 0:
                        k1 = k1 - 0.1
                    x = (b2 - b1) / (k1 - k2)
                    y = k1 * x + b1
                    x = int(round(x))
                    y = int(round(y))
                except:
                    x = (b2 - b1 - 0.01) / (k1 - k2 + 0.01)
                    y = k1 * x + b1
                    x = int(round(x))
                    y = int(round(y))
                if x < 0 or y < 0 or x > w or y > h:
                    break
                xlist.append(x)
                ylist.append(y)
        cx=int(np.mean(xlist))
        cy=int(np.mean(ylist))
        self.centerPoint=[cx,cy]
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), 2)
        return img

    def FitPointerLine(self):
        #拟合指针直线段
        img =self.poniterMask
        lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength=int(self.r / 2), maxLineGap=2)
        dmax=0
        pointerLine=[]
        #最长的线段为指针
        for line in lines:
            x1, y1, x2, y2 = line[0]
            d1=Functions.Disttances((x1, y1),(x2, y2))
            if(d1>dmax):
                dmax=d1
                pointerLine=line[0]      
        x1, y1, x2, y2 = pointerLine
        d1=Functions.Disttances((x1, y1),(self.centerPoint[0],self.centerPoint[1]))
        d2=Functions.Disttances((x2, y2),(self.centerPoint[0],self.centerPoint[1]))
        if d1 > d2:
            self.farPoint = [x1, y1]
            self.k = (y2 - y1) / (x2 - x1)
        else:
            self.farPoint = [x2, y2]
            self.k = (y1 - y2) / (x1 - x2)

    def Readvalue(self,range,angle):
        try:
            self.ImgCutCircle()
            self.ContoursFilter()
            self.FitNumLine()
            self.getIntersectionPoints()
            self.FitPointerLine()
            v1=[self.zeroPoint[0]-self.centerPoint[0],self.centerPoint[1]-self.zeroPoint[1]]            # 中心点与0刻度点向量
            v2=[self.farPoint[0]-self.centerPoint[0],self.centerPoint[1]-self.farPoint[1]]              # 中心点与指针点向量
            v3 = [self.finalPoint[0] - self.centerPoint[0], self.centerPoint[1] - self.finalPoint[1]]   # 中心点与终刻度点向量
            theta = Functions.GetClockAngle(v1,v2)                    #指针点与0刻度点角度
            self.divisionValue = Functions.GetClockAngle(v1, v3)      #角度量程
            readValue = (range / self.divisionValue) * theta  # (表盘量程/对应角度) * 指针点与0刻度点角度
            print(f" 矫正前:表盘读数:{readValue:.2f},刻度量程:{range:.2f}KPa,指针角度:{theta:.2f},角度量程:{self.divisionValue:.2f}")
            readValue = (range / angle) * theta                #(表盘量程/对应角度) * 指针点与0刻度点角度
            print(f" 矫正后:表盘读数:{readValue:.2f},刻度量程:{range:.2f}KPa,指针角度:{theta:.2f},角度量程:{angle:.2f}")
            return readValue
        except Exception as e:# 写一个except
            print("程序错误：",e)

if __name__ =="__main__":  

    #多张图片，修改输入文件夹

    # imglist=glob.glob('input/*.jpg')  
    # for imgpath in  imglist: 
    #     A=MeterDetection(imgpath)
    #     A.Readvalue()
    #一张图片
    imgpath='1.jpg'
    A=MeterDetection(imgpath)
    readValue=A.Readvalue()
