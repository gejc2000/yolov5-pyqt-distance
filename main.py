import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from main_win.win import Ui_mainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon

import sys
import os
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
from utils.CustomMessageBox import MessageBox
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.plots import colors, plot_one_box, plot_one_box_PIL
from utils.plots import Annotator, colors, save_one_box

from utils.torch_utils import select_device
from utils.capnums import Camera
from dialog.rtsp_win import Window





"""""这段代码的目的是将一个二维像素坐标点（point2D）转换为三维世界坐标点，但代码中存在一些问题和潜在的混淆点。首先，我会逐行解析代码，然后指出其中的问题和可能的改进。

代码解析
函数定义和参数

convert_2D_to_3D 函数接受二维像素坐标点和其他相机参数作为输入。
返回值是未经畸变校正和经过畸变校正的三维世界坐标点。
预处理

point2D 被转换为浮点数类型，并增加一个维度使其变为 (u, v, 1) 的形式。
计算了旋转矩阵 R 和内参矩阵 IntrinsicMatrix 的逆矩阵。
畸变校正

使用 distortion_correction 函数（未在代码段中给出）对 point2D 进行畸变校正，得到 uvPoint_yes_correct。
计算三维坐标

使用逆矩阵和畸变校正后的点来计算三维坐标。
s1 被计算为 Zc（相机坐标系下的 Z 坐标），但使用了 height（假设为世界坐标系下的 Z 坐标）进行缩放，这是不准确的。
p1 是相机坐标系下的三维点。
p_c 是通过应用旋转和平移矩阵从相机坐标系转换到世界坐标系的三维点。
 """


#
# ### 坐标系转换
# def convert_2D_to_3D(point2D, R, t, IntrinsicMatrix, K, P, f, principal_point, height):
#     """
#
#     像素坐标转世界坐标
#     Args:
#         point2D: 像素坐标点
#         R: 旋转矩阵
#         t: 平移矩阵
#         IntrinsicMatrix:内参矩阵
#         K:径向畸变
#         P:切向畸变
#         f:焦距
#         principal_point:主点
#         height:Z_w
#
#     Returns:返回世界坐标系点，point3D_no_correct, point3D_yes_correct
#
#     """
#     point3D_no_correct = []
#     point3D_yes_correct = []
#
#     ##[(u1,v1),
#     #   (u2,v2)]
#
#     point2D = (np.array(point2D, dtype='float32'))
#
#     # (u,v,1)
#     # point2D_op = np.hstack((point2D, np.ones((num_Pts, 1))))
#     point2D_op = np.hstack((point2D, np.array([1])))  # 2D————》3D了加一个维度
#
#     # 畸变矫正后变量
#     uvPoint_yes_correct = distortion_correction(point2D, principal_point, f, K, P)
#     ##########  前面是对像素点的矫正以及像素坐标系到图像坐标系的转换
#     # R逆矩阵
#     rMat_inv = np.linalg.inv(R)
#     # 内参矩阵的逆矩阵
#     IntrinsicMatrix_inv = np.linalg.inv(IntrinsicMatrix)
#
#     # uvPoint变量切换即可
#     uvPoint = point2D_op
#     uvPoint_yes_correct_T = uvPoint_yes_correct.T
#     ##########  图像坐标系 *平移矩阵
#     tempMat = np.matmul(rMat_inv, IntrinsicMatrix_inv)
#     #   # 畸变矫正后变量  * 变化矩阵R的-1
#     tempMat1_yes_correct = np.matmul(tempMat, uvPoint_yes_correct_T)  # mat1=R^(-1)*K^(-1)([U,V,1].T)
#     tempMat2_yes_correct = np.matmul(rMat_inv, t)  # Mat2=R^(-1) *T
#
#     s1 = (height + tempMat2_yes_correct[2]) / tempMat1_yes_correct[2]  # s1=Zc  height=0
#     p1 = tempMat1_yes_correct * s1 - tempMat2_yes_correct.T  # [Xw,Yw,Zw].T  =mat1*zc -mat2
#     p_c = np.matmul(R, p1.reshape(-1, 1)) + t.reshape(-1, 1)
#
#     return p1, p_c

### 坐标系转换
def convert_2D_to_3D(point2D, R, t, IntrinsicMatrix, K, P, f, principal_point, height):
    """

    像素坐标转世界坐标
    Args:
        point2D: 像素坐标点
        R: 旋转矩阵
        t: 平移矩阵
        IntrinsicMatrix:内参矩阵
        K:径向畸变
        P:切向畸变
        f:焦距
        principal_point:主点
        height:Z_w

    Returns:返回世界坐标系点，point3D_no_correct, point3D_yes_correct

    """
    point3D_no_correct = []
    point3D_yes_correct = []

    ##[(u1,v1),
    #   (u2,v2)]

    point2D = (np.array(point2D, dtype='float32'))

    # (u,v,1)
    # point2D_op = np.hstack((point2D, np.ones((num_Pts, 1))))
    point2D_op = np.hstack((point2D, np.array([1])))  # 2D————》3D了加一个维度

    # 畸变矫正后变量
    uvPoint_yes_correct = distortion_correction(point2D, principal_point, f, K, P)
    ##########  前面是对像素点的矫正以及像素坐标系到图像坐标系的转换
    # R逆矩阵
    rMat_inv = np.linalg.inv(R)
    # 内参矩阵的逆矩阵
    IntrinsicMatrix_inv = np.linalg.inv(IntrinsicMatrix)

    # uvPoint变量切换即可
    uvPoint = point2D_op
    uvPoint_yes_correct_T = uvPoint_yes_correct.T
    ##########  图像坐标系 *平移矩阵
    tempMat = np.matmul(rMat_inv, IntrinsicMatrix_inv)
    #   # 畸变矫正后变量  * 变化矩阵R的-1
    tempMat1_yes_correct = np.matmul(tempMat, uvPoint_yes_correct_T)  # mat1=R^(-1)*K^(-1)([U,V,1].T)

    # a = np.array([
    #     [1,2,3,4],
    #     [5,6,7,8],
    #     [9,10,11,12]
    # ])
    tempMat2_yes_correct = np.matmul(rMat_inv, t)  # Mat2=R^(-1) *T

    s1 = (height + tempMat2_yes_correct[2]) / tempMat1_yes_correct[2]  # s1=Zc  height=0
    p1 = tempMat1_yes_correct * s1 - tempMat2_yes_correct.T  # [Xw,Yw,Zw].T  =mat1*zc -mat2
    p_c = np.matmul(R, p1.reshape(-1, 1)) + t.reshape(-1, 1)

    return p1, p_c


"""函数定义：函数接受一个二维坐标点 uvPoint、主点 principal_point、焦距 f、径向畸变系数 K 和切向畸变系数 P 作为输入，并返回校正后的坐标点。

畸变计算：代码正确地使用了径向畸变和切向畸变的数学公式来计算校正后的坐标。

问题：

在计算 x_distorted 和 y_distorted 时，不应加1，因为这会改变坐标值，这通常不是畸变校正过程的一部分。
如果 f 是一个列表或数组，并且表示不同轴的焦距（例如，对于非矩形像素或鱼眼相机），则应当分别用于 x 和 y 的计算。但在大多数情况下，f 只是一个标量，表示像素焦距（在x和y方向上相同）。"""


# uvPoint[0] 和 uvPoint[1] 是图像坐标系中的点的横坐标（u）和纵坐标（v）。
#
# principal_point[0] 和 principal_point[1] 分别是图像坐标系中主点（即相机光轴与图像平面的交点）的横坐标（u0 或 cx）和纵坐标（v0 或 cy）。
#
# f[0] 和 f[1] 是焦距参数，但在大多数相机模型中，我们假设 x 和 y 方向的焦距是相同的，并且通常只使用一个焦距值 f。然而，在某些情况下（如鱼眼相机或具有非矩形像素的相机），x 和 y 方向的焦距可能不同，这时就需要使用两个不同的值 fx 和 fy。

def distortion_correction(uvPoint, principal_point, f, K, P):
    """
    畸变矫正函数：畸变发生在图像坐标系转相机坐标系
    Args:
        uvPoint: 坐标点（u，v）
        principal_point: 主点
        f: 焦距
        K: 径向畸变
        P: 切向畸变
    Returns:返回矫正坐标点
    """
    # K：径向畸变系数
    [k1, k2, k3] = K
    # p：切向畸变系数
    [p1, p2] = P
    #  在畸变校正函数中，您可能会使用归一化相机坐标系中的点 (x, y) 来计算畸变
    x = (uvPoint[0] - principal_point[0]) / f[0]
    y = (uvPoint[1] - principal_point[1]) / f[1]
    # 计算畸变校正函数矩阵
    r = x ** 2 + y ** 2
    x1 = x * (1 + k1 * r + k2 * r ** 2 + k3 * r ** 3) + 2 * p1 * y + p2 * (r + 2 * x ** 2)
    y1 = y * (1 + k1 * r + k2 * r ** 2 + k3 * r ** 3) + 2 * p2 * x + p1 * (r + 2 * y ** 2)

    x_distorted = f[0] * x1 + principal_point[0] + 1
    y_distorted = f[1] * y1 + principal_point[1] + 1

    return np.array([x_distorted, y_distorted, 1])
    ##  已知了该点的坐标也就是检测框的[x1+x2/2, y2] M1是径向畸变系数，M2切向畸变系数,那么我们这个时候就得到的矫正后的坐标【x_distorted, y_distorted, 1】


class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    # emit：detecting/pause/stop/finished/error msg
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './yolov5s.pt'
        self.current_weight = './yolov5s.pt'
        self.source = '0'
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.jump_out = False  # jump out of the loop
        self.is_continue = True  # continue/pause
        self.percent_length = 1000  # progress bar
        self.rate_check = True  # Whether to enable delay
        self.rate = 100
        self.save_fold = './result'

    @torch.no_grad()
    def run(self,
            imgsz=640,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            ):

        # Initialize
        try:
            device = select_device(device)
            half &= device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = attempt_load(self.weights, map_location=device)  # load FP32 model
            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            if half:
                model.half()  # to FP16

            # Dataloader
            if self.source.isnumeric() or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadWebcam(self.source, img_size=imgsz, stride=stride)
                # bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            count = 0
            jump_count = 0
            start_time = time.time()
            dataset = iter(dataset)

            while True:
                if self.jump_out:
                    self.vid_cap.release()
                    self.send_percent.emit(0)
                    self.send_msg.emit('Stop')
                    if hasattr(self, 'out'):
                        self.out.release()
                    break
                # change model
                if self.current_weight != self.weights:
                    # Load model
                    model = attempt_load(self.weights, map_location=device)  # load FP32 model
                    num_params = 0
                    for param in model.parameters():
                        num_params += param.numel()
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check image size
                    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                    if half:
                        model.half()  # to FP16
                    # Run inference
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                    self.current_weight = self.weights
                if self.is_continue:
                    path, img, im0s, self.vid_cap = next(dataset)
                    # jump_count += 1
                    # if jump_count % 5 != 0:
                    #     continue
                    count += 1
                    if count % 30 == 0 and count >= 30:
                        fps = int(30 / (time.time() - start_time))
                        self.send_fps.emit('fps：' + str(fps))
                        start_time = time.time()
                    if self.vid_cap:
                        percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.percent_length)
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length

                    statistic_dic = {name: 0 for name in names}
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    pred = model(img, augment=augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms,
                                               max_det=max_det)
                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        im0 = im0s.copy()
                        annotator = Annotator(im0, line_width=2, example=str(names))
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                cls_name = names[int(cls)]  # 获取类别名称
                                # if cls_name in ['car', 'truck', 'motorcycle', 'bus']:  # 检查类别名称是否在列表中
                                c = int(cls)  # 整数类别
                                statistic_dic[cls_name] += 1  # 更新统计字典
                                # 中心点
                                x1, y1, x2, y2 = xyxy  # 解包坐标
                                INPUT = [(x1 + x2) / 2, y2]  # 计算中心点

                                # 调用你的转换函数，注意这里假设你已经定义了所有需要的参数
                                p1, p_c = convert_2D_to_3D(INPUT, R, t, IntrinsicMatrix, K, P, f, principal_point,
                                                           0)

                                print("-----p1----", p1)
                                d1 = p1[0][1]  # 假设这是从p1中提取的某个值，你可能需要根据你的实际情况调整这行代码

                                print("----p_c---", type(p_c))
                                distance = float(p_c[0])  # 假设p_c是一个列表或元组，并且你想要它的第一个元素作为距离

                                # 隐藏标签或显示标签及置信度（这里也假设hide_conf是一个布尔值）
                                label = None if hide_labels else (
                                    cls_name if hide_conf else f'{cls_name} {conf:.2f} {distance:.2f}m')

                                annotator.box_label(xyxy, label, color=colors(c, True))
                    if self.rate_check:
                        time.sleep(1 / self.rate)
                    im0 = annotator.result()
                    self.send_img.emit(im0)
                    self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                    self.send_statistic.emit(statistic_dic)
                    if self.save_fold:
                        os.makedirs(self.save_fold, exist_ok=True)
                        if self.vid_cap is None:
                            save_path = os.path.join(self.save_fold,
                                                     time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                   time.localtime()) + '.jpg')
                            cv2.imwrite(save_path, im0)
                        else:
                            if count == 1:
                                ori_fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
                                if ori_fps == 0:
                                    ori_fps = 25
                                # width = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                # height = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                width, height = im0.shape[1], im0.shape[0]
                                save_path = os.path.join(self.save_fold,
                                                         time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.mp4')
                                self.out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), ori_fps,
                                                           (width, height))
                            self.out.write(im0)
                    if percent == self.percent_length:
                        print(count)
                        self.send_percent.emit(0)
                        self.send_msg.emit('finished')
                        if hasattr(self, 'out'):
                            self.out.release()
                        break

        except Exception as e:
            self.send_msg.emit('%s' % e)


class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False

        # style 1: window can be stretched
        # self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)

        # style 2: window can not be stretched
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
                            | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        # self.setWindowOpacity(0.85)  # Transparency of window

        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        # show Maximized window
        self.maxButton.animateClick(10)
        self.closeButton.clicked.connect(self.close)

        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # search models automatically
        self.comboBox.clear()
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        # yolov5 thread
        self.det_thread = DetThread()
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.det_thread.source = '0'
        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.raw_video))
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))
        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        self.fileButton.clicked.connect(self.open_file)
        self.cameraButton.clicked.connect(self.chose_cam)
        self.rtspButton.clicked.connect(self.chose_rtsp)

        self.runButton.clicked.connect(self.run_or_continue)
        self.stopButton.clicked.connect(self.stop)

        self.comboBox.currentTextChanged.connect(self.change_model)
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.checkBox.clicked.connect(self.checkrate)
        self.saveCheckBox.clicked.connect(self.is_save)
        self.load_setting()

    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def is_save(self):
        if self.saveCheckBox.isChecked():
            self.det_thread.save_fold = './result'
        else:
            self.det_thread.save_fold = None

    def checkrate(self):
        if self.checkBox.isChecked():
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False

    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.67:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='Tips', text='Loading rtsp stream', time=1000, auto=True).exec_()
            self.det_thread.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.statistic_msg('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.statistic_msg('%s' % e)

    def chose_cam(self):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='Tips', text='Loading camera', time=2000, auto=True).exec_()
            # get the number of local cameras
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.cameraButton.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).x()
            y = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).y()
            y = y + self.cameraButton.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec_(pos)
            if action:
                self.det_thread.source = action.text()
                self.statistic_msg('Loading camera：{}'.format(action.text()))
        except Exception as e:
            self.statistic_msg('%s' % e)

    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            check = 0
            savecheck = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "check": check,
                          "savecheck": savecheck
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                check = 0
                savecheck = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                check = config['check']
                savecheck = config['savecheck']
        self.confSpinBox.setValue(conf)
        self.iouSpinBox.setValue(iou)
        self.rateSpinBox.setValue(rate)
        self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check
        self.saveCheckBox.setCheckState(savecheck)
        self.is_save()

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x * 100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x / 100)
            self.det_thread.conf_thres = x / 100
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x * 100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x / 100)
            self.det_thread.iou_thres = x / 100
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            self.det_thread.rate = x * 10
        else:
            pass

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        # self.qtimer.start(3000)

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)
        if msg == "Finished":
            self.saveCheckBox.setEnabled(True)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.statistic_msg('Change model to %s' % x)

    def open_file(self):

        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                              "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('Loaded file：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.saveCheckBox.setEnabled(False)
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = 'camera' if source.isnumeric() else source
            self.statistic_msg('Detecting >> model：{}，file：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))
        else:
            self.det_thread.is_continue = False
            self.statistic_msg('Pause')

    def stop(self):
        self.det_thread.jump_out = True
        self.saveCheckBox.setEnabled(True)

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' ' + str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.resultWidget.addItems(results)

        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        self.det_thread.jump_out = True
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.confSpinBox.value()
        config['conf'] = self.iouSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox.checkState()
        config['savecheck'] = self.saveCheckBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        MessageBox(
            self.closeButton, title='Tips', text='Closing the program', time=2000, auto=True).exec_()
        sys.exit(0)


if __name__ == "__main__":
    R = np.array([[9.1119371736959609e-01, 2.4815760576991752e-02, 4.1123009064654115e-01],
                  [4.1105811256386449e-01, 1.1909647756530584e-02, 9.1153134251420498e-01],
                  [2.7517949080742898e-02, 9.9962109737505089e-01, 6.5127650722056341e-04]])
    R = R.T
    t = np.array([[1.0966499328613281e+01],
                  [4.1683087348937988e+00],
                  [8.7983322143554688e-01]])
    # 内参矩阵，转置
    IntrinsicMatrix = np.array([[14.0874, 0, 0],
                                [0, 14.7552, 0],
                                [16.5402, 14.2077, 1]])
    IntrinsicMatrix = IntrinsicMatrix.T
    # 焦距
    f = [1.9770188633212194e+03, 1.9668641721787440e+03]  # fx 和fy
    # 主点
    principal_point = [1.0126938349335526e+03, 4.7095156301902404e+02]
    # 径向畸变矩阵
    K = [-0.3746, 0.1854, -0.0514]
    # 切向畸变矩阵
    P = [0.0074, -0.0012]
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    # myWin.showMaximized()
    sys.exit(app.exec_())
