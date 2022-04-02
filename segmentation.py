# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 10:57:32 2022

@author: amarb
"""


import cv2
import numpy as np
import math
import maxflow

class SelectFgBgRegion:
    def __init__(self,rgbImg):
        self.foregroundpos = []
        self.backgroundpos = []
        self.rgbImg = rgbImg
        self.grayImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2GRAY)/255
        self.ismousedown = False # true if mouse is pressed
        self.mode = True
        self.x = 0
        self.y = 0
        h, w = rgbImg.shape[0], rgbImg.shape[1]
        self.tempImg = np.zeros((h, w, 3))
    
    def drawcircle(self,event,x,y,flags,param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.ismousedown = True
            self.ix,self.iy = x,y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.ismousedown == True:
                if self.mode == True:
                    cv2.circle(self.rgbImg,(x,y),3,(255,0,0),-1)
                    cv2.circle(self.tempImg,(x,y),3,(255,0,0),-1)
                else:
                    cv2.circle(self.rgbImg,(x,y),3,(0,0,255),-1)
                    cv2.circle(self.tempImg,(x,y),3,(0,0,255),-1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.ismousedown = False
            if self.mode == True:
                cv2.circle(self.rgbImg,(x,y),3,(255,0,0),-1)
                cv2.circle(self.tempImg,(x,y),3,(255,0,0),-1)
            else:
                cv2.circle(self.rgbImg,(x,y),3,(0,0,255),-1)
                cv2.circle(self.tempImg,(x,y),3,(0,0,255),-1)

    def startselection(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',self.drawcircle)
        while(1):
            cv2.imshow('image',self.rgbImg)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('m'):
                self.mode = not self.mode   # change mode from FG to BG
            elif k == ord('q'):
                break

        
        # for foreground
        y, x = np.where(self.tempImg[:, :, 0] == 255)
        for i in range(y.shape[0]):
            self.foregroundpos.append((y[i], x[i]))
        # for background
        y, x = np.where(self.tempImg[:, :, 2] == 255)
        for i in range(y.shape[0]):
            self.backgroundpos.append((y[i], x[i]))

      
        bpos = set(self.foregroundpos)
        self.foregroundpos = list(bpos)
        rpos = set(self.backgroundpos)
        self.backgroundpos = list(rpos)

        foregroundpixval = [self.grayImg[x,y] for (x,y) in self.foregroundpos]
        backgroundpixval = [self.grayImg[x,y] for (x,y) in self.backgroundpos]
        cv2.destroyAllWindows()
        return self.foregroundpos,self.backgroundpos,foregroundpixval,backgroundpixval
    
########### functions for pdf ###############################
        
def compute_pdfs_FB(F, B):
    mean1 = np.mean(B)
    mean2 = np.mean(F)
    std1 = np.std(B)
    std2 = np.std(F)
    GaussB = (mean1,std1)
    GaussF = (mean2,std2)
    return GaussB,GaussF

######### get gaussian probability distribution ##############

def get_gauss(x,mean,std):
    f = (1./(std*math.sqrt(2*math.pi)))
    val =  f*np.exp(-0.5*np.square((x-mean)/std))
    return val

def WiFB(img,Lambda,GaussB,GaussF):

    probF = get_gauss(img, GaussF[0],eps+GaussF[1])
    probB = get_gauss(img, GaussB[0],eps+GaussB[1])
  
    WiF = -1*Lambda*np.log( np.divide(probB,(eps+probF+probB)))
    WiB = -1*Lambda*np.log( np.divide(probF,(eps +probF+probB)))

    return WiF,WiB

def set_weights_for_selected_FB(WiF,WiB, posF, posB,MIN,MAX):
    for val in posF:
        WiF[val[0],val[1]] = MAX
        WiB[val[0],val[1]] = MIN

    for val in posB:
        WiF[val[0],val[1]] = MIN
        WiB[val[0],val[1]] = MAX
        
    return WiF,WiB

def W_ij(img1,img2, sigma):
    r, c = img1.shape
    index = np.arange(r*c).reshape(r, c)

    f = -1/(2*sigma*sigma)

    L = np.roll(img2,1,axis=1)
    R = np.roll(img2,-1,axis=1)
    U = np.roll(img2,1, axis=0)
    D = np.roll(img2,-1,axis=0)
    
    L = (np.exp(f*np.square(img1-L)))
    R = (np.exp(f*np.square(img1-R)))
    U = (np.exp(f*np.square(img1-U)))
    D = (np.exp(f*np.square(img1-D)))


    L_index = np.roll(index,-1,axis=1)
    R_index = np.roll(index,1,axis=1)
    U_index = np.roll(index,-1,axis=0)
    D_index = np.roll(index,1,axis=0)

    return L,R,U,D,L_index,R_index,U_index,D_index,index

def get_graph(img, Sigma, Lambda, F_pos, B_pos, list_B, list_F):
    L,R,U,D,L_index,R_index,U_index,D_index,index = W_ij(img,img,Sigma)
    GB,GF = compute_pdfs_FB(list_F, list_B)
    WiF,WiB = WiFB(img,Lambda,GB,GF)
    WiF,WiB = set_weights_for_selected_FB(WiF,WiB, F_pos, B_pos, MIN,MAX)
    return L,R,U,D,WiF,WiB
    
if __name__ == "__main__":
    lamda = 0.4
    sigma = 0.1
    MIN = 0
    MAX = 1000
    eps = 1e-8
    
    img = cv2.imread("dog1.png")
    img = cv2.blur(img, (3,3))
    img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_CUBIC)
    originalImg = img.copy()
    regionSel = SelectFgBgRegion(img)
    fg,bg,fgval,bgval=regionSel.startselection()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
    m,n = grayImg.shape
    L,R,U,D,WiF,WiB = get_graph(grayImg, sigma, lamda, fg, bg, bgval, fgval)
    cv2.imshow("FG",WiF)
    cv2.imshow("BG", WiB)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    graph = maxflow.Graph[float]()
    nodeid = graph.add_nodes(m*n)
    for i in range(m):
            for j in range(n):
                if(j!= n-1):
                    graph.add_edge(i*m+j,i*m+j+1,0,L[i,j])
                if(i!= m-1):
                    graph.add_edge(i*m+j,(i+1)*m+j,0,D[i,j])

    for i in range(m):
        for j in range(n):
                graph.add_tedge(i*m+j,WiF[i,j],WiB[i,j])
    print(graph.maxflow())
    for i in range(m):
            for j in range(n):
                if(graph.get_segment(i*m+j)):
                    originalImg[i,j,:] = 255
    cv2.imshow("image",originalImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
