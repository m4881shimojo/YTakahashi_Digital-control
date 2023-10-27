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
P=np.eye(4,4);Q=np.eye(4,1);C=np.zeros(4) # 単位行列
Pc=np.eye(4,4);Qc=np.eye(4,1);Cc=np.zeros(4) # zero行列
E=np.eye(4,4);T=np.eye(4,4);T0=np.eye(4,4)
D=np.zeros(4)
n=4
#
A=np.eye(4,4);B=np.eye(4,4) #書籍にはない
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
#
# p41のデータを代入
#P=np.array([[0.921,0.335,0.0,0.0],
#            [-0.335,0.921,0.0,0.0],
#            [0.0,1.0,0.921,0.335],
#            [0.0,0.0,-0.335,0.921]])
Q=np.array([[0],
           [1.],
           [0],
           [0]])
C=np.array([0,0,1.,0])

Q_original=Q
#Qc=Q #書籍の記述
#
#Calculate E
for j in range(0,n):
    E[:,j]=Q[:,0] #Eq. 3-19
    Q=P.dot(Q)
E=LA.inv(E) #E=inv_E
D=E[n-1,:] #Eq.3-20
#
#Calcutate Pc,Qc,Cc in Eq.3-18
# make T matrix
for j in range(0,n):
    T[j,:]=D    #Eq.3-21
    D=D.dot(P)
#Calcutate Pc
T0=LA.inv(T) #T0=1/T
E=P.dot(T0) #P 1/T
Pc=T.dot(E) #(TP1/T) Pc in Eq.3-18

# Pc行列の(n-1)行をAに代入
A=Pc[n-1,:] # 書籍ではAの代わりにDを使っている
A=-A   # an,...a1

##################################################
# 以下は検証のための印刷
##################################################
print("\n---------Pc,Qc,Cc--------------------")
#https://analytics-note.xyz/programming/numpy-printoptions/
np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止

print("An......A1:")
print(A)
print("\n-------------------------------------")
#
Cc=C.dot(T0) #bn..b1
B=Cc # Cc行列の１行をBに代入。あまり意味がないが。。
print("Bn......B1:")
print(B)
print("\n-------------------------------------")
#
# 念のためPc,Qc,Tを印刷してみる
print("Pc  :")
print(Pc)

Qc=T.dot(Q_original)
print("\nqc  :")
print(Qc)

print("\nMatrix T:")
print(T)

###########################################
# original data
###########################################
print("\n\n-------------P,Q,C-------------------")
print("P  :")
print(P)


print("\nq  :")
print(Q_original)

print("\nc  :")
print(C)

#END



    
