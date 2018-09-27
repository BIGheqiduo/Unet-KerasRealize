ENVIRONMENT:
Ubuntu 18.04.1
keras 2.2.2
Tensorflow-gpu 1.8.0
CUDA 9.0
CUDNN 7.1 (if not work, please try CUDNN 7.0)

"""UNet网络分割训练,测试 """
"""训练"""
1、首先是打开data.py文件，然后设置好TrainImage、trainMask、TestImage个文件夹的路径，ImgType，是否要进行数据增强。
2、然后生成的对应的npy文件
3、打开unet.py文件，然后设置好对应的参数，然后启动unet.py文件，开始训练。

"""测试"""
打开unet.py文件，然后设置好对应的参数，然后启动unet.py文件，开始训练。