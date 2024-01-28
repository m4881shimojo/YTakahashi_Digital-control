#digital control 
#デジタル制御　高橋安人
#list 3-1 page 40. shimojo 20231021
#可制御同伴形の計算、4次系の例
# matrix enshuu 20160802 shimojo
#p40
#
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA   #invert Matrix
#
n=4 # 次数
m=1 # 入力数
P=np.eye(n,n);Q=np.eye(n,m);C=np.zeros(n) # 単位行列
Pc=np.eye(n,n);Qc=np.eye(n,m);Cc=np.zeros(n) # zero行列
E=np.eye(n,n);T=np.eye(n,n)#;T0=np.eye(4,4)
#D=np.zeros(4)
#
# input Data #page 33　図3-1のパラメータ
#計算して代入
modedeg=20.;r=0.98
a=np.cos(2*np.pi/360*modedeg)*r
b=np.sin(2*np.pi/360*modedeg)*r

P=np.array([[a,b,0.0,0.0],
            [-b,a,0.0,0.0],
            [0.0,1.0,a,b],
            [0.0,0.0,-b,a]])
Q=np.array([[0],
           [1.],
           [0],
           [0]])
C=np.array([0,0,1.,0])

Q1=np.copy(Q) #Q1 working memo
#Qc=Q #書籍の記述
#
#Calculate E
for j in range(0,n):
    E[:,j]=Q1[:,0] #Eq. 3-19
    Q1=np.dot(P,Q1)
invE=np.linalg.inv(E)
D=invE[n-1,:] #Eq.3-20 page 38
#
#Calcutate Pc,Qc,Cc in Eq.3-18
# make T matrix
for j in range(0,n):
    T[j,:]=D    #Eq.3-21
    D=np.dot(D,P) #D working memo
#Calcutate Pc
invT=np.linalg.inv(T)
Pc=np.dot(T,np.dot(P,invT))#Pc in Eq.3-18
#
# Pc行列の(n-1)行をAに代入
ac=-Pc[n-1,:]   # an,...a1
Cc=np.dot(C,invT)   # bn,...b1

##################################################
# 以下は検証のための印刷
##################################################
#https://analytics-note.xyz/programming/numpy-printoptions/
np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止

# 念のためPc,Qc,Tを印刷してみる
print("\nPc  :\n",Pc)
Qc=np.dot(T,Q)
print("\nqc  :\n",Qc)
#print("\nT:\n",T)

#　A(z),B(z)を印刷
print("\n---------Pc,Qc,Cc--------------------")
print("\nA(z):(an,an-1...a2,a1:)\n",ac)
print("\n-------------------------------------")
print("\nB(z):(bn..b1:)\n",Cc) #(3-18)
#print("\n-------------------------------------")
#
#END
##################################################
##       検証のため、Python controlで算出         #
##################################################
from control import *
# from control import *
print("\n----------Python control------------\n")
sys1ss = ss(P, Q, C, 0)
sys1tf = ss2tf(sys1ss)
print(sys1ss)
print(sys1tf)



    
