import os
import numpy as np
import keras
from keras.models import Model
"""Concatenate,"""
from keras.layers import Dense, Dropout, Input, Conv2D, MaxPool2D, merge, Concatenate,  UpSampling2D
from keras.layers import concatenate

from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
"""
from keras import backend as keras
"""
from data import *
import cv2


IMG_CHANNELS = 1
IMG_ROWS, IMG_COLS = 512, 512
INPUT_SHAPR = (IMG_CHANNELS, IMG_ROWS, IMG_COLS)

ImgsTrainFilename = "ImgsTrain.npy"
ImgsMaskTrainFilename = "ImgsMaskTrain.npy"
ImgsTestFilename = "ImgsTest.npy"
ImgsMaskTestFilename = "ImgsMaskTest.npy"

NpyDataDirPath = "./npyData"
ResultsDataDirPath = "./Result"



"""Train Image Abs Path"""
"""Train Image Mask Abs Path"""
"""Test Image Abs Path"""
"""
ImgTrainFilePathAbs = os.path.join(NpyDataDirPath, ImgsTrainFilename)
ImgsMaskTrainFilePathAbs = os.path.join(NpyDataDirPath, ImgsMaskTrainFilename)
ImgsTestFilePathAbs = os.path.join(NpyDataDirPath, ImgsTestFilename)
"""

class Unet(object):

	def __init__(self, ImgRowsInput= IMG_ROWS, ImgColsInput = IMG_COLS, ImgChannels = IMG_CHANNELS):
		self.ImgRows = ImgRowsInput
		self.ImgCols = ImgColsInput
		self.ImgChannels = ImgChannels

		self.BatchSize = 1
		self.ModelEpoch = 800000

		self.ModelFileName = "Unet.hdf5"

		"""Train Image Abs Path"""
		"""Train Image Mask Abs Path"""
		"""Test Image Abs Path"""
		self.ImgTrainFilePathAbs = os.path.join(NpyDataDirPath, ImgsTrainFilename)
		self.ImgsMaskTrainFilePathAbs = os.path.join(NpyDataDirPath, ImgsMaskTrainFilename)
		self.ImgsTestFilePathAbs = os.path.join(NpyDataDirPath, ImgsTestFilename)
		self.ImgsMaskTestFilePathAbs = os.path.join(ResultsDataDirPath, ImgsMaskTestFilename)

		self.ImgsType = ".jpg"

	def LoadData(self):
		SegData = dataProcess(self.ImgRows, self.ImgCols)
		ImgsTrain, ImgsMaskTrain = SegData.loadTrainData()
		ImgsTest = SegData.loadTestData()
		return ImgsTrain, ImgsMaskTrain, ImgsTest

	def GetUnet(self):
		DataInput = Input((self.ImgRows, self.ImgCols, self.ImgChannels))

		Conv1 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(DataInput)
		print( "Conv1 shape:" + str(Conv1.shape)) 
		Conv1 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(Conv1)
		print( "Conv1 shape:" + str(Conv1.shape)) 
		Pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(Conv1)
		print( "Pool1 shape:" + str(Pool1.shape)) 

		Conv2 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(Pool1)
		print( "Conv2 shape:" + str(Conv2.shape)) 
		Conv2 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(Conv2)
		print( "Conv2 shape:" + str(Conv2.shape)) 
		Pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(Conv2)
		print( "Pool2 shape:" + str(Pool2.shape)) 


		Conv3 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(Pool2)
		print( "Conv3 shape:" + str(Conv3.shape)) 
		Conv3 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(Conv3)
		print( "Conv3 shape:" + str(Conv3.shape)) 
		Pool3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(Conv3)
		print( "Pool3 shape:" + str(Pool3.shape)) 

		Conv4 = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(Pool3)
		print( "Conv4 shape:" + str(Conv4.shape)) 
		Conv4 = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(Conv4)
		print( "Conv4 shape:" + str(Conv4.shape)) 
		Drop4 = Dropout(0.5)(Conv4)
		Pool4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(Drop4)
		print( "Pool4 shape:" + str(Pool4.shape))

		Conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(Pool4)
		print( "Conv5 shape:" + str(Conv5.shape)) 
		Conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(Conv5)
		print( "Conv5 shape:" + str(Conv5.shape)) 
		Drop5 = Dropout(0.5)(Conv5)

		Up6 = Conv2D(512, (2, 2), activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(Drop5))
		# Merge6 = merge([Drop4, Up6], mode="concat", concat_axis=3)
		#Merge6 = merge.Concatenate([Drop4, Up6], axis=-1)
		Merge6 = concatenate([Drop4, Up6], axis=3)
		Conv6 = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(Merge6)
		Conv6 = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(Conv6)

		Up7 = Conv2D(256, (2, 2), activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(Conv6))
		#Merge7 = merge([Conv3, Up7], mode="concat", concat_axis=3)
		Merge7 = concatenate([Conv3, Up7], axis=3)
		Conv7 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(Merge7)
		Conv7 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(Conv7)

		Up8 = Conv2D(128, (2, 2), activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(Conv7))
		#Merge8 = merge([Conv2, Up8], mode="concat", concat_axis=3)
		Merge8 = concatenate([Conv2, Up8], axis=3)
		Conv8 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(Merge8)
		Conv8 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(Conv8)

		Up9 = Conv2D(64, (2, 2), activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(Conv8))
		#Merge9 = merge([Conv1, Up9], mode="concat", concat_axis=3)
		Merge9 = concatenate([Conv1, Up9], axis=3)
		Conv9 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(Merge9)
		Conv9 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(Conv9)
		Conv9 = Conv2D(2, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(Conv9)
		Conv10 = Conv2D(1, (1, 1), activation="sigmoid")(Conv9)


		model = Model(input = DataInput, output = Conv10)
		model.compile(optimizer=Adam(lr = 1e-4), loss="binary_crossentropy", metrics=["accuracy"])

		return model


	def Train(self):
		print("Loading Data")
		ImgsTrain, ImgsMaskTrain, ImgsTest = self.LoadData()
		print("Loaded Data")
		model = self.GetUnet()
		print("Got Unet Model")
		#model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		modelCheckpoint = ModelCheckpoint(self.ModelFileName, monitor='loss',verbose=1, save_best_only=True)
		print("Will Save Model")

		print('Fitting model...')
		#verbose=1 have charged
		model.fit(ImgsTrain, ImgsMaskTrain, batch_size=self.BatchSize, nb_epoch=self.ModelEpoch, verbose=0,validation_split=0.2, shuffle=True, callbacks=[modelCheckpoint])
		#save model selfIdea
		model.save(self.ModelFileName)

	def Test(self):
		print("Loading Data---")
		ImgsTrain, ImgsMaskTrain, ImgsTest = self.LoadData()
		print("Loading Data Done")
		model = self.GetUnet()
		print("Got Unet Model")
		model.load_weights(self.ModelFileName)
		print("Predict Image")
		ImgsMaskTest = model.predict(ImgsTest, batch_size=1,verbose=0)
		np.save(self.ImgsMaskTestFilePathAbs, ImgsMaskTest)

	def SaveImg(self):
		print("Array To Image")
		#Imgs = np.load(self.ImgsMaskTestFilePathAbs)
		Imgs = np.load("./npyData/" + ImgsTestFilename)
		print('imgs_mask_test Num:' + str(Imgs.shape[0]))
		ImgCount = 0
		for SingleImg in Imgs:
			ImgCount = ImgCount + 1
			CurImgFileName = str(ImgCount).zfill(6) + self.ImgsType
			CurImgFileDirPathAbs = os.path.join(ResultsDataDirPath, CurImgFileName)
			print(CurImgFileDirPathAbs)
			#SingleImg = self.ImageGrayTranformation(SingleImg)
			cv2.imwrite(CurImgFileDirPathAbs, SingleImg)

	def ImageGrayTranformation(self, ImgArrayInput, ImgNewMin = 0, ImgNewMax = 255):
		ImgArrOldMax = ImgArrayInput.max()
		ImgArrOldMin = ImgArrayInput.min()

		ImgArrNewMax = ImgNewMax
		ImgArrNewMin = ImgNewMin

		ImgArrayOutput = ((ImgArrNewMax - ImgArrNewMin) / (ImgArrOldMax - ImgArrOldMin)) * (ImgArrayInput - ImgArrOldMin) + ImgArrNewMin
		return ImgArrayOutput

"""
ImgsTrainFilename = "ImgsTrain.npy"
ImgsMaskTrainFilename = "ImgsMaskTrain.npy"
ImgsTestFilename = "ImgsTest.npy"
ImgsMaskTestFilename = "ImgsMaskTest.npy"

NpyDataDirPath = "./npyData"
ResultsDataDirPath = "./Result"
"""

if __name__ == '__main__':
	_UNET_TRAIN_ = True
	_UNET_TEST_ = False

	myUnet = Unet()
	if _UNET_TRAIN_:
		myUnet.Train()
	#Test And Save ImgsMaskTest
	if _UNET_TEST_:
		myUnet.Test()
		myUnet.SaveImg()








