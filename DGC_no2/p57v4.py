# matrix enshuu 20231028 shimojo
# 20240112 control.matlab
#p57ver2 3連振動系　教科書例題通りに行った
#list4-1 Pとqの算定
#表4-1
#外力m1､m3．出力m1､m2の変位。B、Cマトリクスを見ること
#
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA
import control as ct

#
n=6 #次数
m=2 #入力数
Tsample=0.1 #sampling time
k1=1.;k2=2.;k3=1.;b1=0.01;b2=0;b3=0.01
#
A=np.zeros((n,n));B=np.zeros((n,m));C=np.zeros((m,n)) #p58 hyou4-1
P=np.eye(n,n);Q0=np.eye(n,n);Dpq=np.eye(n,n)

A=np.array([[0,1.0,0.,0.,0.,0.],
            [-k1,-b1,k1,b1,0.,0.],
            [0.,0.,0.,1.,0.,0.],
            [k1,b1,-(k1+k2),-(b1+b2),k2,b2],
            [0.,0.,0.,0.,0.,1.],
            [0.,0.,k2,b2,-(k2+k3),-(b2+b3)]])
B=np.array([[0.,0.], #p58 hyou4-1
            [1.,0.],
            [0.,0.],
            [0.,0.],
            [0.,0.],
            [0.,2.]])
C=np.array([[1.,0.,0.,0.,0.,0.],
            [0.,0.,1.2,0.,0.,0.]])
#
#np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止
np.set_printoptions(precision=3,  floatmode="fixed")#　=True 指数表記禁止
#
#input A, B, C matrix
print("\n -------------A, B and C matrix------------")
print("\nA matrix\n",A)
print("\nB matrix\n",B)
print("\nC matrix\n",C)

#calculate P,Q matrix
#Epq--> working memo
k=0
e0=1.0E-8;e1=1.0 #誤差許容値
A1=Tsample*A #A1---> working memo
while e1>e0:
    k=k+1
    Dpq=A1.dot(Dpq);Dpq=(1/k)*Dpq;Epq=(1/(k+1))*Dpq
    P=P+Dpq;Q0=Q0+Epq
    e1=np.sum(np.abs(Dpq))
    #
# while end
# get Q matrix          
Q0=Tsample*Q0; Q=np.dot(Q0,B)            
#calculate end

np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止
# P and Q matrix was calculated
print("\n -------------P, Q and dc-gain matrix------------")
print("number of recursion=",k)
print("\nP matrix\n",P)
print("\nQ matrix\n",Q)
#end

#dc gain (4-6)
#単位ステップ応答の終端状態
I=np.eye(n,n) #単位行列        
invP=np.linalg.inv(I-P) #(4-6)
G1=np.dot(C,np.dot(invP,Q))
print("\ndc-gain=\n",G1)
#
##################################################
##       検証のため、Python controlで算出         #
##################################################
#import control as ct
#
print("\n----------Python control------------\n")
#
sys=ct.ss(A,B,C,0)
#print(sys)
sysPd=ct.c2d(sys, Tsample, method='zoh')
print(sysPd)

#sys = ct.ss(P, Q, C, 0)
#sys_tf = ct.ss2tf(sys);sys_tf


