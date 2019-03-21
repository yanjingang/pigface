## 人脸识别

### 方式一：模型训练
捕获摄像头中的人脸:  
```
python face.py camera_face yan
```
从目录图片中提取人脸:  
```
python face.py path_face ~/Desktop/FACE/
```
从指定图片中提取人脸:  
```
python face.py image_face data/facedb/test/2-1.png
```
从face目录生成cnn训练数据:  
```
python face.py train_data
```
 

### 方式二：face embedding距离比对
使用face_embedding方式建立fFaceDB并识别人脸:  
```
python face.py face_embedding
```
追踪摄像头中的人脸并与FaceDB对比:  
```
python infer.py
```
