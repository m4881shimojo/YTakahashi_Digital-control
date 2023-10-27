#digital control 
#デジタル制御　高橋安人
#図3-1複素重極α±jβのモード　page.33
# shimojo 20231015
##
#
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA
#from numpy.linalg import inv
#Digital Control P.33の図

n=100
for l in range(2):
    r=1.0-l*0.02
    Y=np.zeros(n)

    modedeg=20.
    a=np.cos(2*np.pi/360*modedeg)*r
    b=np.sin(2*np.pi/360*modedeg)*r
    #print(a,b)

    #
    #p.33のパラメータ
    P=np.array([[a,b,0,0],
            [-b,a,0,0],
            [0,1.,a,b],
            [0,0,-b,a]])
    #
    C=np.array([0,0,1.,0])
    X0=np.array([[0],
             [1.],
             [0],
            [0]])

    Xk=X0
    for k in range(0,n):
        Xk1=np.dot(P,Xk)
        #Xk1
        Y[k]=np.dot(C,Xk)
        Xk=Xk1
        #print(k,Y[k])
    #
    t=np.arange(0,n)

    if l==0: plt.plot(t,Y,'*--b')     #最終出力 
    if l==1: plt.plot(t,Y,'*--r')     #最終出力

plt.title("図3-1 複素重極α±jβのモード", fontname="MS Gothic")
plt.ylabel("Responce y(k)")
plt.xlabel("Step (k)")
    
# 表示
plt.show()    
