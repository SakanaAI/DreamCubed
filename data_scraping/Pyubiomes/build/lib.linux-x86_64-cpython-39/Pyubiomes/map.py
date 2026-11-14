import cv2
import os
from .overworld import save_map
class PyuMap:
	"""
	The base class for maps in Pyubiomes.
	Attributes:
	seed: long
		The seed the map will make a biome representation of
	areaX: int
		The X coordinate of the negative-most corner of the map
	areaZ: int
		The X coordinate of the negative-most corner of the map
	areaWidth: int
		The width of the map
	areaHeight: int
		The height of the map
	scale: int
		Scaling factor of the map
	version: int
		The version of Minecraft that the map was made in
	filename: str
		The name of the file the map will be saved to, excluding the file extension
	"""
	def __init__(self,seed, areaX, areaZ, areaWidth, areaHeight, scale, version, filename):
		'''
	Attributes:
	seed: long
		The seed the map will make a biome representation of
	areaX: int
		The X coordinate of the negative-most corner of the map
	areaZ: int
		The X coordinate of the negative-most corner of the map
	areaWidth: int
		The width of the map
	areaHeight: int
		The height of the map
	scale: int
		Scaling factor of the map
	version: int
		The version of Minecraft that the map was made in
	filename: str
		The name of the file the map will be saved to, excluding the file extension'''
		self.seed=seed
		self.areaX=areaX
		self.areaZ=areaZ
		self.areaWidth=areaWidth
		self.areaHeight=areaHeight
		self.scale=scale
		self.version=version
		self.filename=filename

	def save(self):
		'''
		Save the map as a .ppm file

		Returns: 
		PyuMap
		'''
		saveMap(self.seed, self.areaX, self.areaZ, self.areaWidth, self.areaHeight, self.scale, self.version, self.filename)
		return self
	def toPNG(self):
		'''
		Convert the .ppm file into a .png for easier viewing

		Returns: 
		void
		'''
		cv2.imwrite(self.filename+".png", cv2.imread(self.filename+".ppm"))
		os.remove(self.filename+".ppm")