#digital control 
#デジタル制御　高橋安人
#20231206 shimojo
#p101 List 6-1 リカチ式
#
#Riccati eqation 
#
#LQ制御
#1入出力系
##Riccati eqation p.101V2
#LQ制御
#list4-1 Pとqの算定

#
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
#p85のサーボ機構についてP,Qを求める
n=2 #次数
m=2 #入力数　1 or 2 入力
T=0.1 #sampling time
#
A=np.zeros((n,n));B=np.zeros((n,m));C=np.zeros((m,n)) #p58 hyou4-1
Dpq=np.eye(n,n);P=np.eye(n,n);Q0=np.eye(n,n)
#Epq=np.eye(n,n)# working memo
#
#
if m==1:
    A=np.array([[0, 1.],
                [0,-1.]])
    B=np.array([[0], #p58 hyou4-1
                [1]])
    C=np.array([[1.,0]])
else:
    A=np.array([[0, 1.],
                [0,-1.]])
    B=np.array([[1,0.], #p58 hyou4-1
                [0,1.]])
    C=np.array([[1.,0.],
                [0.,1.]])
#
#
np.set_printoptions(precision=5,  floatmode="fixed")#　=True 指数表記禁止

#input A, B, C matrix
print("\n -------------A, B and C matrix------------")
print("\nA matrix")
print(A)
print("\nB matrix")
print(B)
print("\nC matrix")
print(C)

#calculate P,Q matrix
k=0
e0=0.000001 #1.0E-6
e1=1.0
A=T*A
while e1>e0:
    k=k+1
    Dpq=A.dot(Dpq);Dpq=(1/k)*Dpq;Epq=(1/(k+1))*Dpq
    P=P+Dpq;Q0=Q0+Epq
    e1=np.sum(np.abs(Dpq))
    #print("k=",k,"error=",e1)
#
Q0=T*Q0;Q=Q0.dot(B)            
#calculate end

np.set_printoptions(precision=8, suppress=True)#　=True 指数表記禁止
# P and Q matrix was calculated
print("\n -------------P, Q and dc-gain matrix------------")
print("number of recursion=",k)
print("\nP matrix")
print(P)
print("\nQ matrix")
print(Q)
#end

#end