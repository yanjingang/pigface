#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: api_face.py
Desc: 人脸识别 API 封装
Demo: 
    nohup python api_face.py > log/api_face.log &
    
    http://www.yanjingang.com:8025/piglab/face?img_file=/home/work/project/pigface/data/facedb/test/1.png

    ps aux | grep api_face.py |grep -v grep| cut -c 9-15 | xargs kill -9
Author: yanjingang(yanjingang@mail.com)
Date: 2019/2/26 16:08
"""

from dp.face import Face
from dp import utils
import sys
import os
import json
import time
import cv2
import logging
import numpy as np
import tensorflow as tf
import tornado.ioloop
import tornado.web
import tornado.httpserver
from xpinyin import Pinyin

CUR_PATH = os.path.dirname(os.path.abspath(__file__))

# init face
face_reco = Face(CUR_PATH+'/data/facedb/faceid/')
pinyin = Pinyin()


class ApiFace(tornado.web.RequestHandler):
    """API逻辑封装"""

    def get(self):
        """get请求处理"""
        try:
            result = self.execute()
        except:
            logging.error('execute fail ' + utils.get_trace())
            result = {'code': 1, 'msg': '请求失败'}
        logging.info('API RES[' + self.request.path + '][' + self.request.method + ']['
                     + str(result['code']) + '][' + str(result['msg']) + '][' + str(result['data']) + ']')
        self.write(json.dumps(result))

    def post(self):
        """post请求处理"""
        try:
            result = self.execute()
        except:
            logging.error('execute fail ' + utils.get_trace())
            result = {'code': 1, 'msg': '请求失败'}
        logging.info('API RES[' + self.request.path + '][' + self.request.method + ']['
                     + str(result['code']) + '][' + str(result['msg']) + ']')
        self.write(json.dumps(result))

    def execute(self):
        """执行业务逻辑"""
        logging.info('API REQUEST INFO[' + self.request.path + '][' + self.request.method + ']['
                     + self.request.remote_ip + '][' + str(self.request.arguments) + ']')
        req_type = self.get_argument('req_type', 'face_catch')
        img_file = self.get_argument('img_file', '')
        face_info = self.get_argument('face_info', {})
        if type(face_info) is str:
            face_info = json.loads(face_info)
        logging.debug("req_type: ".format(req_type))
        logging.debug("img_file: ".format(img_file))
        logging.debug("face_info: ".format(face_info))
        if req_type == 'face_catch' and img_file == '':  # 人脸捕获识别
            return {'code': 2, 'msg': 'img_file不能为空'}
        if req_type == 'face_register' and len(face_info) == 0:  # 人脸注册
            return {'code': 3, 'msg': 'face_info不能为空'}
        res = []
        msg = 'success'

        try:
            # 人脸捕获
            if req_type == 'face_catch':
                ret = face_reco.get_faceids(img_file)
                for face in ret:
                    res.append({'src': img_file, 'faceid': face['faceid'], 'facename': face['facename'], 'weight': face['weight'], 'rect': face['rect']})
                return {'code': 0, 'msg': msg, 'data': res}
            # 人脸注册
            elif req_type == 'face_register':
                faceid = face_reco.register_faceid(face_info['src'], name=face_info['faceid'], rect=face_info['rect'])
                logging.warning("face_register {} done! facedb size:{}".format(faceid, len(face_reco.facedb)))
                if len(face_reco.facedb) == 0:
                    return {'code': 4, 'msg': '注册失败', 'data': res}
        except:
            logging.error('execute fail [' + img_file + '] ' + utils.get_trace())
            return {'code': 5, 'msg': '请求失败', 'data': res}

        # 组织返回格式
        return {'code': 0, 'msg': msg, 'data': res}


if __name__ == '__main__':
    """服务入口"""
    port = 8025

    # log init
    log_file = ApiFace.__name__.lower()  # + '-' + str(os.getpid())
    utils.init_logging(log_file=log_file, log_path=CUR_PATH)
    print("log_file: {}".format(log_file))

    # 路由
    app = tornado.web.Application(
        handlers=[
            (r'/piglab/face', ApiFace)
        ]
    )

    # init
    '''
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # 0.初始化
            face_embed.init(sess, reset_facedb=False)
    '''

    # 启动api服务
    http_server = tornado.httpserver.HTTPServer(app, xheaders=True)
    http_server.listen(port)
    tornado.ioloop.IOLoop.instance().start()
