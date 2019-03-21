#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
import face_recognition
from PIL import Image, ImageDraw

# PATH
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.realpath(CUR_PATH + '/../../../')
sys.path.append(BASE_PATH)

from machinelearning.lib import utils
from face import Face, FaceEmbedding, FaceRecognition

# log init
utils.init_logging(log_file='test', log_path=CUR_PATH)

# 把视频extract提取的face自动分拣分目录
faceid_path = FaceRecognition.FACE_DB_PATH + 'faceid_test/'  # 从视频extract提取的face里每个人物取一个face做faceid
extract_path = '/Users/yanjingang/project/faceswap/data/extract/zhu/'  # 要分拣的视频extract目录
output_path = '/Users/yanjingang/project/faceswap/data/extract/output/'  # 分拣结果输出位置
face_reco = FaceRecognition(faceid_path)
face_reco.face_sorting(extract_path, output_path)
i = 0
for file in os.listdir(extract_path):
    i +=1
    if file in utils.SKIP:
        continue
    faceids = face_reco.get_faceids(extract_path + file)
    if len(faceids) == 0:
        logging.info("TEST faceid not exists: {}".format(extract_path + file))
        continue
    faceid = faceids[0]['faceid']
    weight = round(faceids[0]['weight'], 2)
    utils.mkdir(output_path + faceid)
    ret = utils.move_file(extract_path + file, output_path + faceid + '/' + file)
    logging.info("TEST move file {}: {} {}  {} -> {}".format(i, faceid, weight, extract_path + file, output_path + faceid + '/' + file))
    # break

'''
# 定位图片中的所有人脸
print(time.time())
image = face_recognition.load_image_file(CUR_PATH + "/data/infer/2.png")
print(time.time())
emb = face_recognition.face_encodings(image)[0]
print(time.time())
face_locations = face_recognition.face_locations(image)
print(face_locations)
print(time.time())

# 识别人脸关键点：眼睛、鼻子、嘴和下巴
face_landmarks_list = face_recognition.face_landmarks(image)
print(face_landmarks_list)
print(face_landmarks_list[0].keys())
print(time.time())
fill_line = (200, 0, 0, 120)
fill_polygon = (0, 200, 0, 80)
for face_landmarks in face_landmarks_list:
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image, 'RGBA')

    d.line(face_landmarks['left_eyebrow'], fill=fill_line, width=1)
    d.line(face_landmarks['right_eyebrow'], fill=fill_line, width=1)
    d.line(face_landmarks['top_lip'], fill=fill_line, width=1)
    d.line(face_landmarks['bottom_lip'], fill=fill_line, width=1)
    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=fill_line, width=1)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=fill_line, width=1)
    d.line(face_landmarks['chin'], fill=fill_polygon)

    #d.polygon(face_landmarks['chin'], fill=fill_polygon)
    d.polygon(face_landmarks['left_eye'], fill=fill_polygon)
    d.polygon(face_landmarks['right_eye'], fill=fill_polygon)
    #d.polygon(face_landmarks['left_eyebrow'], fill=fill_polygon)
    #d.polygon(face_landmarks['right_eyebrow'], fill=fill_polygon)
    #d.polygon(face_landmarks['top_lip'], fill=fill_polygon)
    #d.polygon(face_landmarks['bottom_lip'], fill=fill_polygon)
    d.polygon(face_landmarks['nose_bridge'], fill=fill_polygon)
    #d.polygon(face_landmarks['nose_tip'], fill=fill_polygon)

    pil_image.show()


# 识别图片中的人是谁
img_yjy = face_recognition.load_image_file(CUR_PATH+"/data/face/yjy.png")
emb_yjy = face_recognition.face_encodings(img_yjy)[0]
print(emb_yjy)
print(time.time())

img = face_recognition.load_image_file(CUR_PATH+"/data/face/yan.png")
emb_yan = face_recognition.face_encodings(img)[0]
print(emb_yan)
print(time.time())

img = face_recognition.load_image_file(CUR_PATH+"/data/face/zhu.png")
emb_zhu = face_recognition.face_encodings(img)[0]
print(emb_zhu)
print(time.time())


known_face_encodings = [emb_yjy,emb_yan,emb_zhu]

distances = face_recognition.face_distance(known_face_encodings, emb)
print(distances)
print(list(distances<=0.4))
print(time.time())


results = face_recognition.compare_faces(known_face_encodings, emb, tolerance=0.4)
print(results)
print(time.time())
'''
