#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: face.py
Desc: 人脸识别基类
Author:yanjingang(yanjingang@mail.com)
Date: 2019/2/21 23:34
Cmd: 捕获摄像头中的人脸：      python face.py camera_face yan
     从目录图片中提取人脸：    python face.py path_face ~/Desktop/FACE/
     从指定图片中提取人脸：    python face.py image_face data/facedb/test/2.png
     从face目录生成训练数据：  python face.py train_data

     查询facedb id数量：       python face.py get_facedb
     使用facenet方式识别人脸： python face.py face_embedding

     # face_recognition
     识别人脸                   python face.py face_recognition
     捕获摄像头中的人脸          python face.py face_recognition_camera
                                    tail -f log/face_recognition_camera-* |grep catch
     分拣人脸图片到不同目录       python face.py face_sorting
                                    tail -f log/face_sorting-* |grep auto_sorting
"""

import os
import sys
import time
import copy
import random
import logging
import cv2
import tensorflow as tf
import numpy as np
import face_recognition
from threading import Thread

# PATH
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.realpath(CUR_PATH + '/../../../')
sys.path.append(BASE_PATH)

from machinelearning.lib import utils
from machinelearning.lib.pygui import PySimpleGUI as sg
import facenet


class Face():
    """人脸捕获"""
    FACE_PATH = CUR_PATH + "/data/face/"  # 捕获人脸数据保存位置
    TRAIN_PATH = CUR_PATH + "/data/train/"  # 训练数据集
    TEST_PATH = CUR_PATH + "/data/test/"  # 测试数据集

    def __init__(self, infer=None):
        """init"""
        # OpenCV人脸捕获分类器
        haarcascade = CUR_PATH + "/data/cv2_haarcascade/haarcascade_frontalface_alt2.xml"
        self.classfier = cv2.CascadeClassifier(haarcascade)
        # path init
        utils.mkdir(Face.FACE_PATH)
        utils.mkdir(Face.TRAIN_PATH)
        utils.mkdir(Face.TEST_PATH)

    def get_face(self, image=None, image_file=None, uname=None, mini_size=60, point=False):
        """识别图片中的人脸"""
        logging.info('__get_face__')
        if image is None:
            if image_file is None:
                return ()
            image = cv2.imread(image_file)  # 加载图片
            logging.info('imread: {}'.format(image_file))

        # 将当前帧转换成灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 利用分类器检测灰度图像中的人脸矩阵数
        #    Params:
        #        grey：要识别的图像数据（即使不转换成灰度也能识别，但是灰度图可以降低计算强度，因为检测的依据是哈尔特征，转换后每个点的RGB数据变成了一维的灰度，这样计算强度就减少很多）
        #        scaleFactor：图像缩放比例，可以理解为同一个物体与相机距离不同，其大小亦不同，必须将其缩放到一定大小才方便识别，该参数指定每次缩放的比例
        #        minNeighbors：对特征检测点周边多少有效点同时检测，这样可避免因选取的特征检测点太小而导致遗漏
        #        minSize：特征检测点的最小值
        #    Return:
        #        the positions of detected faces as Rect(x,y,w,h)，x、y是左上角起始坐标，h、w是高和宽
        res = self.classfier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        face_rects = []
        for face_rect in res:
            x, y, w, h = face_rect
            # print("{} {} {} {}".format(x, y , w, h))
            if w < mini_size or h < mini_size:  # 过滤太小的人脸
                continue
            if point is True:  # 高宽转为坐标
                face_rect = [x, y, x + w, y + h]
            face_rects.append(face_rect)
        if image_file is not None:  # 指定文件提取时，保存提取出的人脸到face目录
            # save path
            image_path = os.path.dirname(image_file)
            image_file = os.path.basename(image_file)
            if uname is None:
                uname = image_file.split('-')[0].strip().lower() if image_path.count('data/face') == 0 else 'unknow'
            pic_path = Face.FACE_PATH + uname
            utils.mkdir(pic_path)
            # save
            face_files = []
            for face_rect in face_rects:  # 单独框出每一张人脸
                x, y, w, h = face_rect
                if point is True:  #
                    w -= x
                    h -= y
                # 保存人脸区域到图片
                pic_name = '{}/{}-{}.jpg'.format(pic_path, uname, int(time.time() * 1000))
                cv2.imwrite(pic_name, image[y:y + h, x:x + w])  # 将当前帧含人脸部分保存为图片，注意这里存的还是彩色图片，前面检测时灰度化是为了降低计算量；这里访问的是从y位开始到y+h-1位
                face_files.append(pic_name)
            logging.info('get_face done! {} {}'.format(uname, face_files))
            return uname, face_rects, face_files

        return face_rects

    def get_camera_face(self, uname='unknow', camera_id=0, window_name="Camera(Process Q to exit)", image_infer=None, mini_size=60):
        """从监视摄像头中捕获人脸并保存"""
        uname = uname.lower()
        if image_infer is not None:  # camera infer时不保存图片
            uname = 'tmp'
        logging.info('__get_camera_face__ {}'.format(uname))
        cv2.namedWindow(window_name)  # 创建窗口
        cap = cv2.VideoCapture(camera_id)  # 打开摄像头

        # OpenCV人脸识别分类器
        haarcascade = CUR_PATH + "/data/cv2_haarcascade/haarcascade_frontalface_alt2.xml"
        classfier = cv2.CascadeClassifier(haarcascade)
        color = (0, 255, 0)  # 人脸边框颜色，RGB格式
        pic_path = Face.FACE_PATH + uname
        utils.mkdir(pic_path)
        pic_num = 0  # 捕获的人脸图片量

        while cap.isOpened():
            ok, image = cap.read()  # 读取一帧数据
            if not ok:
                break

            # 1.利用分类器检测灰度图像中的人脸矩阵数
            face_rects = self.get_face(image, mini_size=mini_size)
            for faceRect in face_rects:  # 单独框出每一张人脸
                x, y, w, h = faceRect

                # 2.保存人脸区域到图片
                pic_num += 1
                pic_name = '{}/camera-{}.jpg'.format(pic_path, int(time.time() * 1000))
                cv2.imwrite(pic_name, image[y:y + h, x:x + w])  # 将当前帧含人脸部分保存为图片，注意这里存的还是彩色图片，前面检测时灰度化是为了降低计算量；这里访问的是从y位开始到y+h-1位
                logging.debug('catch camera face: {}\t{}'.format(pic_num, pic_name))

                # 3.预测
                if image_infer is not None:
                    ret, msg, res = image_infer(pic_name)
                    if len(res) > 0 and ((res[0]['embedding'] is True and res[0]['weight'] < 1.0) or (res[0]['embedding'] is False and res[0]['weight'] > 0.85)):  # >指定概率才显示
                        res = res[0]
                        cv2.putText(image, '{} {}'.format(res['faceid'], round(float(res['weight']), 6)), (x + 3, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)  # 轮廓叠加图上标记轮廓序号

                # 4.原图画框
                # cv2.rectangle(image, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # 显示图像并等待10毫秒按键输入，输入‘q’退出程序
            cv2.imshow(window_name, image)
            c = cv2.waitKey(20)
            # print(c)
            if c & 0xFF == ord('q'):
                break

        # 释放摄像头并销毁所有窗口
        cap.release()
        cv2.destroyAllWindows()
        logging.info('get_camera_face done! user:{} num: {} path:{}'.format(uname, pic_num, pic_path))
        return pic_num

    def get_path_face(self, src_path=None, mini_size=60):
        """从目录图片里提取人脸并保存"""
        if src_path is None:
            src_path = CUR_PATH + "/data/src/"
        logging.info('__get_pic_face__ {}'.format(src_path))
        result = {}  # 捕获的人脸图片量

        for img_file in os.listdir(src_path):
            if img_file in utils.SKIP:
                continue
            logging.debug(src_path + img_file)
            # 从图片中提取人脸
            uname, face_rects, face_files = self.get_face(image_file=src_path + img_file, mini_size=mini_size)
            for face_file in face_files:
                logging.debug('catch pic face: {}'.format(face_file))
            if uname not in result:
                result[uname] = len(face_files)
            else:
                result[uname] += len(face_files)

        logging.info('get_pics_face done! {}'.format(result))
        return result

    def create_train_data(self, limit=330):
        """使用face数据生成训练数据（10%用于测试）"""
        logging.info('__create_train_data__')
        # clear
        utils.rmdir(Face.TEST_PATH)
        utils.mkdir(Face.TEST_PATH)
        utils.rmdir(Face.TRAIN_PATH)
        utils.mkdir(Face.TRAIN_PATH)
        # create
        num = {'train': 0, 'test': 0, 'users': {}}
        for uname in os.listdir(Face.FACE_PATH):
            if uname in utils.SKIP:
                continue
            if uname not in num['users']:
                num['users'][uname] = 0
            i = 0
            source_files = os.listdir(Face.FACE_PATH + uname)
            if len(source_files) / limit < 0.9:  # 数据太少的user不参与训练
                logging.warning("{} face is too little, skip to train! {} < {}".format(uname, len(source_files), limit))
                continue
            source_files = random.sample(source_files, limit)  # 每个user随机抽取limit个face
            for source_file in source_files:
                if source_file in utils.SKIP:
                    continue
                num['users'][uname] += 1
                source_file = Face.FACE_PATH + uname + '/' + source_file
                target_file = "{}.{}.jpg".format(uname, i)
                if i % 10 == 0:
                    target_file = Face.TEST_PATH + target_file
                    num['test'] += 1
                else:
                    target_file = Face.TRAIN_PATH + target_file
                    num['train'] += 1

                utils.copy_file(source_file, target_file)
                # logging.debug("copy: {} -> {}".format(source_file, target_file))

                i += 1
        logging.info('create_train_data done! {}'.format(num))
        return num


class FaceEmbedding:
    """通过建立Faceid对应特征向量的方式识别人脸ID"""

    def __init__(self):
        self.model_dir = CUR_PATH + '/model/facenet/20180402-114759'
        self.image_size = 160  # 图像占位高宽
        self.threshold = [0.8, 0.8, 0.8]
        self.factor = 0.7
        self.minsize = 20
        self.margin = 44
        self.detect_multiple_faces = True  # 识别多个人脸
        self.face = Face()
        self.faceid_embedding_db = CUR_PATH + "/data/facedb/faceid.db"
        self.facedb = None

    def init(self, sess=None, reset_facedb=False):
        """初始化"""
        logging.info("__init__")
        # 初始化session
        self.sess = sess
        # 加载facenet最新模型
        facenet.load_model(self.model_dir)
        logging.info("load facenet model done!")
        # 重新生成FaceDB
        if reset_facedb is True:
            self.create_face_db()
        # 加载FaceDB字典
        self.facedb = self.load_face_db()
        logging.info("load facedb done!")

    def create_face_db(self, img_path=None):
        """创建FaceDB字典"""
        logging.info("__create_face_db__")
        if img_path is None:
            img_path = CUR_PATH + '/data/facedb/faceid/'
        facedb = []
        for img_file in os.listdir(img_path):
            if img_file in utils.SKIP:
                continue
            img_file = img_path + img_file
            # logging.debug("img_file: {}".format(img_file))
            faces = self.get_face_embedding(img_file=img_file)
            # print(faces)
            if len(faces) == 0:
                logging.warning("get_face_embedding empty!  img_file: {}".format(img_file))
                continue
            facedb.append(faces)
        # save facedb
        utils.pickle_dump(facedb, self.faceid_embedding_db)
        # update facedb
        self.facedb = facedb
        logging.info("create facedb done!")
        return True if len(facedb) > 0 else False

    def load_face_db(self):
        """加载FaceID.db特征字典"""
        logging.info("__load_face_db__")
        embeddings = []
        try:
            embeddings = utils.pickle_load(self.faceid_embedding_db)
        except:
            logging.error("load face db fail! {}\n{}".format(self.faceid_embedding_db, utils.get_trace()))
        logging.info("load face db done!  {}".format(len(embeddings)))
        return embeddings

    def get_faceids(self, img_file):
        """计算指定图片中的人脸与facedb中各faceid的距离（距离越小越像, <0.6可以确信是同一个人, >1不可信）"""
        logging.info("__get_faceids__")
        # 图片特征向量
        faces_embed = self.get_face_embedding(img_file=img_file)
        # print(faces_embed)
        for embedding in faces_embed:
            embedding['distance'] = self.ecuclidean_distance(embedding)
            embedding['faceid'] = embedding['distance'][0][0] if len(embedding['distance']) > 0 and embedding['distance'][0][1] < 1.0 else ''
            embedding['weight'] = float(embedding['distance'][0][1]) if len(embedding['distance']) > 0 and embedding['distance'][0][1] < 1.0 else 1.0
            del (embedding['embedding'])
        logging.info("get faceids: {}".format(faces_embed))
        return faces_embed

    def ecuclidean_distance(self, embedding):
        """face特征 欧几里得距离计算"""
        # logging.info("__ecuclidean_distance__")
        # 欧几里得距离计算
        if len(embedding) == 0:
            return []

        res = {}
        for face in self.facedb:
            face = face[0]
            dist = np.sqrt(np.sum(np.square(np.subtract(face['embedding'], embedding['embedding']))))
            res[face['faceid']] = dist
        return sorted(res.items(), key=lambda d: d[1])

    def get_face_embedding(self, img_file=None, img=None, bounding_boxes=None, faceid=None):
        """通过图片对象和人脸边框生成face特征向量（单face）
            Params:
                img_file 原始图片路径
                img      原始图片对象（与img_file二选一即可）
                bounding_boxes  原始图片中人脸区域points（传入img_file时不用传）
                faceid   图片faceid（仅在create_face_db时有用）
        """
        logging.info("__get_face_embedding__")
        faces = []
        if img is None and img_file is None:
            return faces
        if img is None and img_file is not None:  # 传入的是图片路径，生成img对象
            logging.info("imread: {}".format(img_file))
            img = cv2.imread(img_file, 1)
        if img is not None and bounding_boxes is None:  # 提取图片中的人脸信息
            face_rects = self.face.get_face(img, mini_size=32, point=True)
            bounding_boxes = np.array(list(face_rects))
            logging.info("get bounding_boxes: {}".format(bounding_boxes))
        if img_file is not None:  # 传入的是图片路径时，需要转换img bgr->rgb
            # logging.debug("bgr->rgb")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faceid = os.path.basename(img_file).split('.')[0]

        face_cnt = bounding_boxes.shape[0]
        logging.debug("faces detected: {}".format(face_cnt))
        if face_cnt > 0:
            det = bounding_boxes[:, 0:4]
            # print(det)
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if face_cnt > 1:  # 多个人脸
                if self.detect_multiple_faces:
                    for i in range(face_cnt):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                    img_center = img_size / 2
                    offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                    det_arr.append(det[index, :])
            else:
                det_arr.append(np.squeeze(det))

            # print(det_arr)
            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - self.margin / 2, 0)
                bb[1] = np.maximum(det[1] - self.margin / 2, 0)
                bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = cv2.resize(cropped, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
                prewhitened = facenet.prewhiten(resized)
                faces.append({'faceid': faceid, 'rect': [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])], 'embedding': self._get_embedding(prewhitened)})
        # logging.debug("get_face_embedding done. {}".format(faces))
        # print(faces)
        return faces

    def _get_embedding(self, processed_img):
        """根据人脸数据生成face特征向量"""
        logging.debug("__get_embedding__")
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        images_placeholder = tf.image.resize_images(images_placeholder, (self.image_size, self.image_size))
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        # print(images_placeholder)
        # print(embeddings)
        # print(phase_train_placeholder)
        # get embedding
        reshaped = processed_img.reshape(-1, self.image_size, self.image_size, 3)
        feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
        # logging.debug("reshape done. {}".format(feed_dict))
        feature_vector = self.sess.run(embeddings, feed_dict=feed_dict)  # 这一行cpu耗时1.1s...
        # logging.debug("run done. {}".format(feature_vector))
        return feature_vector


class FaceRecognition():
    """开源人脸测试"""
    DATA_PATH = CUR_PATH + "/data/"
    FACE_DB_PATH = CUR_PATH + "/data/facedb/"

    def __init__(self, img_path=None):
        """初始化facedb"""
        self.facedb, self.facenames = [], []
        self.create_face_db(img_path=img_path)

    def create_face_db(self, img_path=None):
        """创建FaceDB字典"""
        if img_path is None:
            img_path = self.FACE_DB_PATH + 'faceid/'

        self.facedb, self.facenames = [], []
        for img_file in os.listdir(img_path):
            if img_file in utils.SKIP:
                continue
            image = face_recognition.load_image_file(img_path + img_file)
            emb = face_recognition.face_encodings(image)[0]
            self.facedb.append(emb)
            self.facenames.append(img_file.split('.')[0])

    def get_faceids(self, img_file=None, image=None, zoom=0.5):
        """计算指定图片中的人脸与facedb中各faceid的距离（距离越小越像, <0.6可以确信是同一个人, >1不可信）"""
        logging.info("__get_faceids__")
        faceids = []
        if image is None and img_file is None:
            logging.error("img_file and image params is empty!")
            return faceids

        if image is None:
            image = face_recognition.load_image_file(img_file)
        else:
            # (window_width, window_height, color_number) = image.shape
            # print("{},{},{}".format(window_width, window_height, color_number))
            if zoom < 1.0:
                small_frame = cv2.resize(image, (0, 0), fx=zoom, fy=zoom)
            else:
                small_frame = image
            # (window_width, window_height, color_number) = small_frame.shape
            # print("{},{},{}".format(window_width, window_height, color_number))
            image = small_frame[:, :, ::-1]
        logging.debug('___face_locations__')
        face_locations = face_recognition.face_locations(image)  # 此函数性能：zoom=1.0,1280*720=500ms; zoom=0.5, 640*360=120ms
        logging.debug('___face_encodings__')
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for i in range(len(face_encodings)):
            face_encoding = face_encodings[i]
            name = "Unknown"
            logging.debug('___face_distance__ {}'.format(i))
            distances = face_recognition.face_distance(self.facedb, face_encoding)
            # print(distances)
            if distances is not None:
                distances = list(distances)
                sort_distances = copy.deepcopy(distances)
                # print(sort_distances)
                sort_distances.sort()
                # print(sort_distances)
                idx = distances.index(sort_distances[0])
                # print(idx)
                name = self.facenames[idx]
                # print(name)
            # rect
            top, right, bottom, left = face_locations[i]
            faceids.append({'faceid': name, 'weight': float(sort_distances[0]), 'rect': (left, top, right, bottom), 'distance': distances})

        logging.info("get faceids: {}".format(faceids))
        return faceids

    def get_camera_face_cv2(self, camera_id=0, window_name="Camera Face (process Q to exit)", zoom=0.5):
        """捕获摄像头中的人脸 - 使用cv2窗口展示"""
        faceids = []
        process_this_frame = True
        video_capture = cv2.VideoCapture(camera_id)

        # cv2 window
        cv2.namedWindow(window_name, 0)
        cv2.moveWindow(window_name, 60, 0)
        ret, image = video_capture.read()
        (window_width, window_height, color_number) = image.shape
        logging.info("{} {} {}".format(window_width, window_height, color_number))
        cv2.resizeWindow(window_name, window_height, window_width)

        # detect
        while True:
            # get cap
            ret, image = video_capture.read()
            # print(ret)
            if ret is False:
                logging.warning("video_capture.read: {} {}".format(ret, image))
                continue
            # get faceid
            if process_this_frame:
                faceids = self.get_faceids(image=image)
            # show
            process_this_frame = not process_this_frame
            for face in faceids:
                left, top, right, bottom = face['rect']
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top = int(top * 1 / zoom)
                right = int(right * 1 / zoom)
                bottom = int(bottom * 1 / zoom)
                left = int(left * 1 / zoom)
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image, '{} {}'.format(face['faceid'], round(face['weight'], 2)), (left + 6, top + 12), font, 0.5, (0, 255, 0), 1)

            # show
            cv2.imshow(window_name, image)
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(200) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

    def _get_camera_face_image(self, camera_data, run_freq=100, max_face=11, zoom=0.5):
        """更新摄像头当前数据"""
        i = -1
        while True:
            logging.debug('___get_camera_face_image__')
            i += 1
            time.sleep(run_freq / 1000)
            # get cap
            ret, image = self.video_capture.read()
            # logging.debug('___get_camera_face_image__ 0')
            # print(ret)
            if ret is False:
                logging.warning("video_capture.read: {} {}".format(ret, image))
                return
            # get faceid
            camera_data['camera']['faceids'] = self.get_faceids(image=image)
            # logging.debug('___get_camera_face_image__ 1')
            # show
            for face in camera_data['camera']['faceids']:
                faceid = face['faceid'] if face['weight'] < 0.5 else 'unknown'  # 只保留距离<0.5的可信face
                left, top, right, bottom = face['rect']
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top = int(top * 1 / zoom)
                right = int(right * 1 / zoom)
                bottom = int(bottom * 1 / zoom)
                left = int(left * 1 / zoom)
                size = bottom - top
                # 人脸计数
                catch_data = camera_data['face']['catch']
                sec_max_cnt = round((1.1 - zoom) * 10, 2)  # 每秒的最多捕获face次数（与face_locations性能有关）
                check_cnt = sec_max_cnt if sec_max_cnt > 1 else sec_max_cnt + 1  # 最少2次才能入队列
                if faceid not in catch_data or time.time() - catch_data[faceid]['lasttime'] > zoom * 3 or catch_data[faceid]['cnt'] >= check_cnt * 1.5:  # 如果此face上次出现距离本次已查超过n s则重置
                    catch_data[faceid] = {'faceid': faceid, 'weight': face['weight'], 'size': size, 'cnt': 1, 'lasttime': time.time(), 'filename': ''}
                else:
                    catch_data[faceid]['cnt'] += 1
                    catch_data[faceid]['weight'] += face['weight']
                    catch_data[faceid]['size'] += size
                    catch_data[faceid]['lasttime'] = time.time()
                # 放入队列
                avg_weight = round(catch_data[faceid]['weight'] / catch_data[faceid]['cnt'], 4)
                avg_size = round(catch_data[faceid]['size'] / catch_data[faceid]['cnt'], 2)
                logging.info("catch: {} {} {} {} {} {}".format(faceid, catch_data[faceid]['cnt'], check_cnt, size, avg_size, avg_weight))
                if (catch_data[faceid]['cnt'] >= check_cnt * 2 and faceid == 'unknown') or \
                        (catch_data[faceid]['cnt'] >= check_cnt and avg_weight < 0.35) or \
                        (catch_data[faceid]['cnt'] >= check_cnt and avg_size >= 200 and avg_weight < 0.4) or \
                        (catch_data[faceid]['cnt'] >= check_cnt and 200 > avg_size >= 150 and avg_weight < 0.43) or \
                        (catch_data[faceid]['cnt'] >= check_cnt and 150 > avg_size >= 100 and avg_weight < 0.47) or \
                        (catch_data[faceid]['cnt'] >= check_cnt * 1.5 and avg_size < 100 and avg_weight < 0.49):  # 连续cnt次被识别，避免偶发错误识别（注意cnt和weight设置与zoom参数有关，zoom=1.0每秒最多2张，zoom=0.5每秒最多6张）
                    logging.info('face catched!  {} {} {}'.format(faceid, avg_size, avg_weight))
                    # logging.debug('___get_camera_face_image__ 1.1')
                    # TODO:检查眨眼
                    # 保存人脸照片
                    if catch_data[faceid]['filename'] == '' or i % 10 == 0:
                        face_file = '{}/face-{}-{}-{}.png'.format(self.DATA_PATH + 'tmp', faceid, round(avg_weight, 2), int(time.time() * 1000))
                        face_image = image[top:bottom, left:right]
                        face_image = cv2.resize(face_image, (80, 80), interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(face_file, face_image)
                        catch_data[faceid]['filename'] = face_file
                    if len(camera_data['face']['list']) > 0 and camera_data['face']['list'][-1]['faceid'] == faceid:  # 如果尾部跟当前是一个人脸，重置此人图像
                        camera_data['face']['list'][-1] = catch_data[faceid]
                        logging.info('reset last self !!! ')
                    else:
                        # clear old data
                        for j in range(len(camera_data['face']['list']) - 1, -1, -1):
                            if camera_data['face']['list'][j]['faceid'] == faceid:
                                logging.info('clear old data !!! ')
                                camera_data['face']['list'].pop(j)
                        # append
                        camera_data['face']['list'].append(catch_data[faceid])
                        if len(camera_data['face']['list']) > max_face:  # 定长
                            logging.info('max face pop !!! {}'.format(len(camera_data['face']['list'])))
                            camera_data['face']['list'].pop(0)

                # 画边框
                # logging.debug('___get_camera_face_image__ 1.2')
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                # 加face标注
                # if catch_data[faceid]['cnt'] > check_cnt * 0.5:
                # logging.debug('___get_camera_face_image__ 1.3')
                cv2.putText(image, '{} {}'.format(faceid, round(face['weight'], 2)), (left + 6, top + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

            # 保存摄像头照片
            # logging.debug('___get_camera_face_image__ 2')
            img_file = '{}/camera-{}.png'.format(self.DATA_PATH + 'tmp', int(time.time() * 1000))
            image = cv2.resize(image, (0, 0), fx=0.798, fy=0.798)
            # logging.debug('___get_camera_face_image__ 3')
            cv2.imwrite(img_file, image)
            # update
            camera_data['camera']['filename'] = img_file
            # print(camera_data)
            logging.debug('___get_camera_face_image__ done')

    def _show_camera_face_window(self, camera_data, max_face=11, warning_faceids=['mjs', 'zhu', 'yjy', 'zn']):
        """创建视频人脸监控windows窗口"""
        # create window
        face_imgs, face_labels = [], []
        for i in range(max_face):
            face_imgs.append(sg.Image(filename=self.DATA_PATH + 'infer/kuang.png', size=(80, 80), key='face' + str(i)))
            face_labels.append(sg.Text('', size=(14, 1), key='label' + str(i)))
        layout = [
            [sg.Image(filename=self.DATA_PATH + 'infer/1.png', size=(600, 400), key='camera')],
            face_imgs,
            face_labels,
            [sg.Text('filename', size=(100, 1), key='filename')]
        ]
        window = sg.Window('视频人脸监控', return_keyboard_events=True, default_element_size=(30, 2), location=(0, 0), use_default_focus=False).Layout(layout).Finalize()
        # window.FindElement('filename').Update(str(time.time()))

        # show camera_data
        while True:
            event, values = window.Read(timeout=5)  # wait 5ms for a GUI event
            if event is None or event == 'Exit':  # exit
                break
            while True:  # 刷新页面数据
                logging.debug('___show_camera_face_window__')
                if camera_data['camera']['filename']:
                    # print(camera_data['camera']['filename'])
                    window.FindElement('camera').Update(filename=camera_data['camera']['filename'])
                    window.FindElement('filename').Update(camera_data['camera']['filename'])
                    try:
                        # show face
                        for i in range(len(camera_data['face']['list'])):
                            face = camera_data['face']['list'][i]
                            window.FindElement('face' + str(i)).Update(face['filename'])
                            window.FindElement('label' + str(i)).Update(face['faceid'] + ' ' + str(round(face['weight'] / face['cnt'], 3)))
                    except:  # will get exception when Queue is empty
                        logging.warning('refresh window fail! {}'.format(utils.get_trace()))
                        break
                    window.Refresh()
                    # 重要人物刚出现时报警
                    if len(camera_data['face']['list']) > 0 and time.time() - camera_data['face']['list'][-1]['lasttime'] < 2.0 and camera_data['face']['list'][-1]['faceid'] in warning_faceids:
                        #utils.play_sound('warning0.wav')
                        Thread(target=utils.play_sound, daemon=True).start()
                logging.debug('___show_camera_face_window__ done')

        # close window
        window.Close()

    def _clear_tmp_path(self, camera_data, sleep=1000):
        """清理摄像头临时数据"""
        tmp_path = self.DATA_PATH + 'tmp'
        while True:
            logging.debug('___clear_tmp_path__')
            time.sleep(sleep / 1000)
            # get used filelist
            skips = []
            for face in camera_data['face']['list']:
                skips.append(face['filename'])
            # clear
            for file in os.listdir(tmp_path):
                # print(float(file.split('-')[-1].split('.')[0]) / 1000)
                timeout = 60.0 * 60 * 24 if file.count('face') > 0 else 30.0  # camera 30s,face 24h
                if tmp_path + '/' + file not in skips and time.time() - float(file.split('-')[-1].split('.')[0]) / 1000 > timeout:  # 30s前的无用临时照片清理
                    os.unlink(tmp_path + '/' + file)
                    logging.debug('clear tmp file: {}'.format(file))
            logging.debug('___clear_tmp_path__ done')

    def get_camera_face(self, camera_id=0, max_face=11):
        """捕获摄像头中的人脸 - 使用自定义window展示"""
        # 摄像头当前数据
        camera_data = {
            'camera': {  # 界面上方摄像头区域数据
                'filename': '',
                'faceids': []
            },
            'face': {  # 界面下方捕获人脸区域数据
                'catch': {},
                'list': [],
                'list_info': {
                    'lastfaceid': '',
                    'lasttime': 0,
                },
            }
        }

        # open camera
        self.video_capture = cv2.VideoCapture(camera_id)

        # clear tmp file
        utils.rmdir(self.DATA_PATH + 'tmp')
        utils.mkdir(self.DATA_PATH + 'tmp')

        # 后台异步线程-更新摄像头当前数据
        Thread(target=self._get_camera_face_image, args=(camera_data, 20, max_face), daemon=True).start()
        # 后台异步线程-清理摄像头临时数据
        Thread(target=self._clear_tmp_path, args=(camera_data, 2000), daemon=True).start()

        # 创建视频人脸监控windows窗口，展示摄像头当前数据
        self._show_camera_face_window(camera_data, max_face=max_face)

        # close camera
        self.video_capture.release()

    def face_sorting(self, extract_path, output_path):
        """按faceid自动分拣face到不同目录（建议调用此函数前，先进行照片extract face提取）
            Params:
                extract_path:要分拣的face文件目录
                output_path: 分拣结果输出位置
            Demo:
                faceid_path = FaceRecognition.FACE_DB_PATH + 'faceid_test/'  # 每个人物取一个face图片放到此目录用于分类
                face_reco = FaceRecognition(faceid_path)
                face_reco.face_sorting(extract_path, output_path)
        """
        i = 0
        for file in os.listdir(extract_path):
            i += 1
            source_file = extract_path + file
            if file in utils.SKIP or os.path.exists(source_file) is False:
                continue
            faceids = self.get_faceids(source_file)
            if len(faceids) == 0:
                logging.info("auto_sorting faceid not found: {}".format(source_file))
                continue
            faceid = faceids[0]['faceid']
            target_file = output_path + faceid + '/' + file
            utils.mkdir(output_path + faceid)
            ret = utils.move_file(source_file, target_file)
            logging.info("auto_sorting face file {}: {} {}  {} -> {}".format(i, faceid, round(faceids[0]['weight'], 2), source_file, target_file))
            # break

if __name__ == '__main__':
    """test"""
    ftype = 'get_camera_face'
    if len(sys.argv) >= 2:
        ftype = sys.argv[1]
    face = Face()
    fe = FaceEmbedding()

    # log init
    log_file = ftype + '-' + str(os.getpid())
    utils.init_logging(log_file=log_file, log_path=CUR_PATH)
    print("log_file: {}".format(log_file))

    # 捕获摄像头中的人脸并保存
    if ftype == 'camera_face':
        uname = 'unknow'
        if len(sys.argv) >= 3:
            uname = sys.argv[2]
        face.get_camera_face(uname)

    # 从目录图片里提取人脸并保存
    if ftype == 'path_face':
        src_path = CUR_PATH + "/data/src/"
        if len(sys.argv) >= 3:
            src_path = sys.argv[2]
        face.get_path_face(src_path)

    # 捕获指定图片中的人脸
    if ftype == 'image_face':
        img_file = CUR_PATH + '/data/facedb/test/2.png'
        if len(sys.argv) >= 3:
            img_file = CUR_PATH + '/' + sys.argv[2]
        print(img_file)
        uname, face_rects, face_files = face.get_face(image_file=img_file, mini_size=32)
        print(face_rects)
        print(face_files)

    # 使用face数据生成训练数据
    if ftype == 'train_data':
        face.create_train_data()

    # 获得facedb数据
    if ftype == 'get_facedb':
        res = fe.load_face_db()
        # print(res)
        print(len(res))

    # 使用facenet识别人脸
    if ftype == 'face_embedding':
        with tf.Graph().as_default():
            with tf.Session() as sess:
                # 0.初始化
                fe.init(sess, reset_facedb=True)
                # 1.比较faceid概率
                file_list = [
                    # CUR_PATH + '/1551324689686.jpg',
                    CUR_PATH + '/data/facedb/test/1.png',
                    # CUR_PATH + '/data/face/mjs/1550833526794.jpg',
                    # CUR_PATH + '/data/face/zhu/1550848580269.jpg',
                    # CUR_PATH + '/data/face/yan/1550832182904.jpg',
                    # CUR_PATH + '/data/face/yjy/camera-1550888618004.jpg',
                ]
                for filename in file_list:
                    res = fe.get_faceids(filename)
                    print("{} {}".format(res, filename))

    # 使用face_recognition识别人脸
    if ftype == 'face_recognition':
        # 0.初始化
        face_reco = FaceRecognition()
        # 1.比较faceid概率
        file_list = [
            CUR_PATH + '/data/facedb/test/1.png',
            CUR_PATH + '/data/face/mjs/1550833526794.jpg',
            CUR_PATH + '/data/face/zhu/1550848580269.jpg',
            CUR_PATH + '/data/face/yan/1550832182904.jpg',
            CUR_PATH + '/data/face/yjy/camera-1550888618004.jpg',
        ]
        for filename in file_list:
            res = face_reco.get_faceids(img_file=filename)
            print("{} {}".format(res, filename))

    # 使用face_recognition捕获摄像头中的人脸
    if ftype == 'face_recognition_camera':
        face_reco = FaceRecognition()
        face_reco.get_camera_face()

    # 人脸图片分拣到不同目录
    if ftype == 'face_sorting':
        faceid_path = FaceRecognition.FACE_DB_PATH + 'faceid_test/'  # 从视频extract提取的face里每个人物取一个face做faceid
        extract_path = '/Users/yanjingang/project/faceswap/data/extract/input/'  # 要分拣的视频extract目录
        output_path = '/Users/yanjingang/project/faceswap/data/extract/output/'  # 分拣结果输出位置
        face_reco = FaceRecognition(faceid_path)
        while True: #边提取边分拣
            face_reco.face_sorting(extract_path, output_path)
