#1.使用opencv-python内置的对象检测配置
pip install opencv-python

find /anaconda3 -type f -name "*frontalface*"
cp /anaconda3/lib/python3.6/site-packages/cv2/data/*.xml data/cv2_data/



#2.训练自己的对象检测xml
brew install opencv@2

echo 'export PATH="/usr/local/opt/opencv@2/bin:$PATH"' >> ~/.bash_profile
export LDFLAGS="-L/usr/local/opt/opencv@2/lib"
export CPPFLAGS="-I/usr/local/opt/opencv@2/include"
export PKG_CONFIG_PATH="/usr/local/opt/opencv@2/lib/pkgconfig"

opencv_createsamples -info pos.txt -num 2000 -w 20 -h 30 -vec pos.vec

http://blog.topspeedsnail.com/archives/10511
http://www.cnblogs.com/tornadomeet/archive/2012/03/28/2420936.html