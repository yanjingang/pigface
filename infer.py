#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: infer.py
Desc: 人脸识别预测
Author:yanjingang(yanjingang@mail.com)
Date: 2019/2/21 23:34
Cmd: python infer.py ./data/infer/0.png
"""

from __future__ import print_function
import sys
import os
import getopt
import logging
import paddle.fluid as fluid
from paddle.fluid.contrib.trainer import *
from paddle.fluid.contrib.inferencer import *
import numpy as np
import tensorflow as tf

# PATH
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.realpath(CUR_PATH + '/../../../')
sys.path.append(BASE_PATH)

from machinelearning.lib import utils
import train
from face import Face, FaceEmbedding, FaceRecognition


class Infer:
    """预测"""

    def __init__(self, params_dirname=CUR_PATH, use_cuda=False):
        if use_cuda and not fluid.core.is_compiled_with_cuda():
            exit('compiled is not with cuda')

        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        logging.info('param_path:' + params_dirname + '/model/resnet10')
        self.inferencer = Inferencer(infer_func=train.image_classification_network, param_path=params_dirname + '/model/resnet10', place=place)
        # FaceEmbedding
        self.face = Face()
        self.face_embed = None

    def image_infer(self, img_file=''):
        """使用模型测试"""
        if img_file == '':
            return 1, 'file_name is empty', {}

        # 识别人脸
        uname, face_rects, face_files = self.face.get_face(image_file=img_file, uname='tmp', mini_size=32)
        results = []
        for i in range(len(face_files)):
            face_file = face_files[i]
            # 预测
            img = utils.load_rgb_image(face_file)
            # logging.info(img.shape)
            result = self.inferencer.infer({'img': img})
            result = np.where(result[0][0] > 0.05, result[0][0], 0)  # 概率<5%的直接设置为0
            logging.info(result)
            label = np.argmax(result)
            # logging.info(train.label_list)
            # print(label)
            faceid = train.label_list[label]
            weight = result[label]

            print("*img: {}".format(face_file))
            print("*label: {}".format(label))
            print("*faceid: {}".format(faceid))
            print("*label weight: {}".format(weight))
            results.append({'src': img_file, 'img': face_file, 'rect': list(face_rects[i]), 'label': label, 'faceid': faceid, 'weight': float(weight)})

        return 0, '', results

    def image_face_embedding_infer(self, img_file=''):
        """使用facenet对比FaceDB距离"""
        if img_file == '':
            return 1, 'file_name is empty', {}
        results = []

        # 计算指定图片与facedb中各faceid的距离
        res = self.face_embed.get_faceids(img_file)
        for face in res:
            results.append({'src': img_file, 'embedding': True, 'faceid': face['faceid'], 'weight': face['weight'], 'rect': face['rect']})

        return 0, '', results

    def camera_infer(self, face_embedding=False):
        """摄像头监控识别"""
        if face_embedding is False:  # 使用cnn模型识别
            self.face.get_camera_face(image_infer=self.image_infer, mini_size=80)
        else:  # 使用facenet特征向量计算FaceDB距离
            self.face_embed = FaceEmbedding()
            with tf.Graph().as_default():
                with tf.Session() as sess:
                    self.face_embed.init(sess)
                    self.face.get_camera_face(image_infer=self.image_face_embedding_infer, mini_size=80)


if __name__ == '__main__':
    """infer test"""
    img_file = CUR_PATH + '/data/face/unknow/0.jpg'
    opts, args = getopt.getopt(sys.argv[1:], "p:", ["file_name="])
    if len(args) > 0 and len(args[0]) > 4:
        img_file = args[0]

    # log init
    log_file = 'infer-' + str(os.getpid())
    utils.init_logging(log_file=log_file, log_path=CUR_PATH)
    print("log_file: {}".format(log_file))

    '''
    infer = Infer()

    # infer
    ret = infer.image_infer(img_file)
    logging.info(ret)

    # camera infer
    infer.camera_infer(face_embedding=True)
    '''

    # 使用face_recognition捕获摄像头中的人脸
    face_reco = FaceRecognition()
    face_reco.get_camera_face()
