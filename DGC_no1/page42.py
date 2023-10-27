# matrix enshuu 20231023 shimojo
#Digital Control P42 応答の算定
#リスト3-2
#行列演算を用いての計算
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA # Linear algebra
#from numpy.linalg import inv

n=50
X=np.eye(4,1)
Y=np.zeros(n)

#図3-1のパラメータ
r=0.98
modedeg=20.
a=np.cos(2*np.pi/360*modedeg)*r
b=np.sin(2*np.pi/360*modedeg)*r
#
P=np.array([[a,b,0,0],
            [-b,a,0,0],
            [0,1.,a,b],
            [0,0,-b,a]])
Q=np.zeros((4,1))
C=np.array([0,0,1.,0])
#U=np.zeros((4,1))
U=np.zeros(n) # 1入力とした
#
# initial conditions
X0=np.array([[0],
             [1.],
             [0],
            [0]])
#
X=X0
####################################
#Caliculate response List3-2
#
for k in range(0,n):
    X=P.dot(X)+Q.dot(U[k]) #x(k+1)=Px(k)+Qu(k)
    Y[k]=C.dot(X)  #y(k)=Cx(k)        
#
####################################
t=np.arange(0,n)
plt.ylim(-10,15.)
#plt.xlim(0,11)
plt.title("図3-1 複素重極モード（list3-2の方法を使う）", fontname="MS Gothic")

plt.plot(t,Y,'-or')     #最終出力 label??

plt.ylabel("Responce y(k)")
plt.xlabel("Step (k)")
    
    # 表示
plt.show()  
