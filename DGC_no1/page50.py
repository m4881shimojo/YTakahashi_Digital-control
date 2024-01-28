# matrix enshuu 20231023 shimojo
#Digital Control P50 
#
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
#次には３次として記述した。
# ３次以外は、nと配列の大きさを変更すればよい
n=3 #3次系
#m=n*(n+1)/2
#
M=np.zeros((6,6));P=np.zeros((3,3));Y=np.zeros((3,3))
W=np.zeros(6);Z=np.zeros(6)
WW=np.zeros(6) #検証用に使う　shimojo Add

#input P,W
P=np.array([[0.6,-0.4,0.3],
            [0.7,-0.9, 0.2],
            [-0.7, 0.5, -0.3]])
W=np.array([[1.],
            [0.],
            [2.],
            [0.],
            [0.],
            [3.]])
#
W=-W
c=-1
########################################################
for b in range (0,n):
    for a in range(0,b+1):
        c=c+1
        r=-1
        for i in range(0,n):
            for j in range(0,i+1):
                r=r+1
                M[r,c]=P[a,i]*P[b,j]+P[b,i]*P[a,j]
                if a==b:
                    M[r,c]=M[r,c]/2.
                if r==c:
                    M[r,c]=M[r,c]-1.
########################################################
invM=LA.inv(M)
Z=invM.dot(W) #(3-44)
#
#generate Y (3-40)
c=-1
for b in range(0,n):
    for a in range(0,b+1):
        c=c+1
        Y[a,b]=Z[c]
        Y[b,a]=Y[a,b]

########################################################
#印刷
# #https://analytics-note.xyz/programming/numpy-printoptions/
np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止
#
print("solution =")
print("\nP=\n",P)
print("\nW=\n",W)
print("\nY=\n",Y)
#
#検算　P'YP-Y=-W
print("\n検算 P'YP-Y=-W")
#
WW=(P.T).dot(Y.dot(P))-Y # (3-37)
#
print("\nW=\n",-WW)
#end

                  


    
    
