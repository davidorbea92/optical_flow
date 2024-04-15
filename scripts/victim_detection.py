#!/usr/bin/env python3

from ultralytics import YOLO
import rospy
import sys
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage, PointCloud2, PointField, Image
import sensor_msgs.point_cloud2 as pc2
import ctypes
import struct
from std_msgs.msg import Header
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images


model = YOLO("/home/david/catkin_ws/src/yolov8/scripts/yolov8n.pt")

fps_cam_pub =20
fps_camera = 20

depth_pixels=10

class victim_detection(object):
	def __init__(self):
		self.cv_image=None
		self.cv_depth=None
		self.yolo_im = []
		self.model = YOLO("yolov8n.pt")
		self.results = None
		self.image_with_boxes=None
		self.z_victim=[]

		self.br = CvBridge()
		
		self.publisher_img = rospy.Publisher("camera_yolo/compressed", CompressedImage, queue_size=1)
		self.publisher_depth = rospy.Publisher("depth_yolo/compressed", CompressedImage, queue_size=1)

	def image_callback(self,im_msg):
		np_arr = np.frombuffer(im_msg.data, np.uint8)
		self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		
	def video_process(self, event=None):
		suma=0
		if self.cv_image is not None:
			unprocessed_image = self.cv_image.copy()
			self.results = self.model.predict(source=unprocessed_image, save=False, save_txt=False, verbose=False)
			for result in self.results:
				self.image_with_boxes = unprocessed_image.copy()

				for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):

					if cls == 0:

						x1, y1, x2, y2 = map(int, box)

						label = f"Prob: {conf:.2f}"

						cv2.rectangle(self.image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
						cv2.rectangle(self.cv_depth, (x1, y1), (x2, y2), (0, 255, 0), 2)

						cv2.putText(self.image_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
						cv2.putText(self.cv_depth, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

						#center_x=(x1+x2)/2
						#center_y=(y1+y2)/2
						center_x=int((x1+x2)/2)
						center_y=int((y1+y2)/2)

						#print(self.cv_depth[center_y][center_x])
						for i in range(center_x-depth_pixels,center_x+depth_pixels):
							for j in range(int(center_y-depth_pixels),int(center_y+depth_pixels)):
								suma=suma+self.cv_depth[j][i]
						self.z_victim.append(suma/(4*depth_pixels*depth_pixels))
						suma=0
	  
	  					
						
						#cv2.putText(self.image_with_boxes, str(self.z_victim[0]), (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
			
				print(self.z_victim)
				
				self.z_victim=[]
			

	def pc_callback(self,pc_msg):
		xyz=np.array([[0,0,0]])
		gen=pc2.read_points(pc_msg,skip_nans=True)

		int_data=list(gen)

		for x in int_data:
			test=x[3]
			s=struct.pack('>f',test)
			xyz=np.append(xyz,[[x[0],x[1],x[2]]], axis=0)
		

    
	def video_pub(self, event=None):
		image_msg=CompressedImage()
		image_msg.header.stamp=rospy.Time.now()
		image_msg.format="jpeg"
		
		if self.image_with_boxes is not None:
			image_msg.data=np.array(cv2.imencode('.jpg',self.image_with_boxes)[1]).tobytes()

			self.publisher_img.publish(image_msg)
	
	def depth_pub(self,event=None):
		depth_msg=CompressedImage()
		depth_msg.header.stamp=rospy.Time.now()
		depth_msg.format="jpeg"
		
		if self.cv_depth is not None:
			depth_msg.data=np.array(cv2.imencode('.jpg',self.cv_depth)[1]).tobytes()

			self.publisher_depth.publish(depth_msg)
		
	#def depth_callback(self,depth_msg):
	#	np_arr = np.frombuffer(depth_msg.data, np.uint8)
	#	self.cv_depth = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
 
	def depth_callback2(self,depth_msg):
		self.cv_depth = self.br.imgmsg_to_cv2(depth_msg, "16UC1")
		#print(self.cv_depth[240][320])
		
		
	def start(self):
		self.cv_process = rospy.Timer(rospy.Duration(1 / fps_camera), self.video_process)
		self.image_pub = rospy.Timer(rospy.Duration(1 / fps_cam_pub), self.video_pub)
		self.depth_pub = rospy.Timer(rospy.Duration(1 / fps_cam_pub), self.depth_pub)
		self.sub_image = rospy.Subscriber("camera/color/image_raw/compressed", CompressedImage, self.image_callback, queue_size=1)
		#self.sub_image2 = rospy.Subscriber("camera/color/image_raw", Image, self.image_callback2, queue_size=1)
#		self.sub_depth = rospy.Subscriber("camera/depth/image_rect_raw/compressed", CompressedImage, self.depth_callback, queue_size=1)
		self.sub_depth = rospy.Subscriber("camera/depth/image_rect_raw", Image, self.depth_callback2, queue_size=1)

	

if __name__ == "__main__":
	rospy.init_node('yolo_node', anonymous=True)
	victim1 = victim_detection()
	try:
		victim1.start()
		rospy.spin()
	except rospy.ROSInterruptException:
		pass

		