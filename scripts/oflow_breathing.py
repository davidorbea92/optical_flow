#!/usr/bin/env python3

import rospy, os, math, sys
import numpy as np
import cv2 as cv
import time
import csv
import os
from sensor_msgs.msg import CompressedImage # Image is the message type
from std_msgs.msg import Int8, String

rate_cv = 30   #frecuencia con la que se hacen calculos en la imagen
rate_camera=8  #frecuencia de publicacion de la imagen


#Frecuencia normal de respiracion entre 12 y 18 respiraciones por minuto
#Esto quiere decir que la frecuencia de respiracion debe estar entre 0.3 y 0.2

cap = cv.VideoCapture(0)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 10,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# random colors for lucas kanade circles
color = np.random.randint(0, 255, (100, 3))



class oflow(object):
    def __init__(self):

        self.mission_state=0

        self.ros_im=0
        self.center=[320,240]
        self.data1=[]
        self.pixel_dist=20
        self.samples_per_line=10
        self.t=0
        self.dist_error=50
        self.c_error=0
        self.p_reset=0.3
        self.mean=0
        self.num_datos=100
        self.file=[]

        for i in range(0,self.samples_per_line):
            start_x=int(self.center[1]-(self.samples_per_line*self.pixel_dist)/2)
            for k in range (0,self.samples_per_line):
                start_y=int(self.center[0]-(self.samples_per_line*self.pixel_dist)/2)
                self.data1.append([start_y+k*self.pixel_dist,start_x+i*self.pixel_dist])

        self.p0=np.array(self.data1,dtype=np.float32)
        self.p_init=self.p0

        
        self.data_y=[]
        self.data_t=[]
        self.data_x=[0]*self.num_datos
        self.list_adj=[0]*self.num_datos
        self.t_zero=[]
        self.frec=[]
        self.suma=0
        self.sum_y=0
        self.save_state=0
        self.file=None

        # Take first frame
        ret, self.old_frame = cap.read()
        self.old_frame=cv.resize(self.old_frame, (640, 480), interpolation = cv.INTER_LINEAR)
        self.old_gray = cv.cvtColor(self.old_frame, cv.COLOR_BGR2GRAY)
        # Create a mask image for drawing purposes
        self.mask = np.zeros_like(self.old_frame)

        self.publisher_arm = rospy.Publisher('arm/position',String,queue_size=1)
        self.publisher_img = rospy.Publisher('robot/optical_flow/compressed',CompressedImage,queue_size=1)


    def optical_flow(self, event=None):
        elapsed=time.time()-self.t
        self.t=time.time()
        
        t_zero=[]
        frec=[]
        
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')

        frame=cv.resize(frame, (640, 480), interpolation = cv.INTER_LINEAR)

        #Inicio del programa de calculo de frecuencia cardiaca
        if (self.mission_state==1):
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **lk_params)
            # Select good points
            if p1 is not None:
                good_new = p1
                good_old = self.p0

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            img = cv.add(frame, self.mask)

            
            if (p1[0][0].size>1):
                for i in range(0,int(p1.size/2)-1):
                    self.sum_y=self.sum_y+p1[i][0][1]
                self.mean=self.sum_y/(p1.size/2)

                for j in range(0,int(p1.size/2)-1):
                    dist_x=np.linalg.norm(p1[j][0]-self.data1[j])
                    if (dist_x>self.dist_error):
                        self.c_error+=1
                        if (self.c_error>p1.size*0.5*self.p_reset):
                            good_new=self.p_init
                            c_error=0
                self.c_error=0        
                self.sum_y=0
            
            

            #Codigo para estimacion de frecuencia

            self.data_y.append(self.mean)
            self.data_t.append(elapsed)

            if (len(self.data_y)>self.num_datos):
                self.data_y.pop(0)
                self.data_t.pop(0)

                suma=sum(self.data_t)
                self.data_x[int(len(self.data_t))-1]=suma-self.data_t[0]
                for i in range(len(self.data_t)-1,0,-1):
                    self.data_x[i-1]=self.data_x[i]-self.data_t[i]

            zero=(max(self.data_y)+min(self.data_y))/2
            for i in range(0,len(self.data_y)):
                self.list_adj[i]=self.data_y[i]-zero
                if (self.list_adj[i]>0 and self.list_adj[i-1]<0 and max(self.list_adj)>2):
                    t_zero.append(self.data_x[i])
            
            if (len(t_zero)>1):
                for i in range(1,len(t_zero)):
                    if (t_zero[i]!=t_zero[i-1]):
                        frec.append(1/(abs(t_zero[i]-t_zero[i-1])))   
            
            frec_resp=np.mean(frec)
            if (frec_resp>1):
                msg="Big movements"
            elif (frec_resp>0.35):
                msg="Agitated breathing"
            elif (frec_resp<0.18 or np.isnan(frec_resp)):
                msg="No movement"
            else:
                msg="Normal breathing"
            
            frame = cv.putText(frame, msg, (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)
            frame = cv.putText(frame, str(frec_resp), (50,80), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv.LINE_AA)
            
            if (self.save_state==1 and self.file is not None):
                with open(self.file, 'a', newline='') as file1:
                    self.writer = csv.writer(file1)
                    self.writer.writerow([msg,str(frec_resp)])

            # Now update the previous frame and previous points
            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)
            

        #copia la imagen al topico de ros
        self.ros_im=frame

    def saveCb(self,msg):
        if msg.data==1:
            self.save_state=1
            self.file=input("Nombre del archivo: ")
            self.file=self.file+'.csv'
            directory = r"/home/robcib/opt_flow"
            path = directory+"/"
            isExist = os.path.exists(path)
            if not isExist:
                print("Directorio no existe")
                os.makedirs(path)
                print("Directorio creado")
            else:
                print("Directorio existe")
            os.chdir(path)            
            with open(self.file, 'w', newline='') as file1:
                self.writer = csv.writer(file1)
                field = ["Message", "Breathing Frecuency"]
                self.writer.writerow(field)
        else:
            self.save_state=0
            self.file=None

    def camara_pub(self, event=None):
        #Imagen comprimida
        im_cmp=CompressedImage()
        im_cmp.header.stamp=rospy.Time.now()
        im_cmp.format="jpeg"

        im_cmp.data=np.array(cv.imencode('.jpg',self.ros_im)[1]).tostring()

        self.publisher_img.publish(im_cmp)

    def commands(self, msg):
        #Instrucciones para el robot
        self.mission_state=msg.data

        if (self.mission_state==2):
            self.publisher_arm.publish('h')
        elif (self.mission_state==0):
            self.publisher_arm.publish('s')
        elif (self.mission_state==4):
            self.publisher_arm.publish('j')
        elif (self.mission_state==5):
            self.publisher_arm.publish('ru')
        elif (self.mission_state==6):
            self.publisher_arm.publish('lu')
        elif (self.mission_state==7):
            self.publisher_arm.publish('rd')
        elif (self.mission_state==8):
            self.publisher_arm.publish('ld')
        elif (self.mission_state==9):
            self.publisher_arm.publish('od')

    def start(self):
        self.sub_commands = rospy.Subscriber('robot/commands', Int8, self.commands)
        rospy.Subscriber('save',Int8,self.saveCb)
        self.cv_process = rospy.Timer(rospy.Duration(1/rate_cv), self.optical_flow)
        self.image_pub=(rospy.Timer(rospy.Duration(1/rate_camera), self.camara_pub))

        #timer para mensajes de move_base_simple
        #self.move_base_pub=(rospy.Timer(rospy.Duration(3), self.mb_pub))        
        rospy.spin()    

if __name__ == '__main__':
    rospy.init_node('opticalflow_breath')
    oflow1=oflow()
    try:
        oflow1.start()
    except rospy.ROSInterruptException:
        pass
