# matrix enshuu 20231028 shimojo
#p57ver2 3連振動系　教科書例題通りに行った
#list4-1 Pとqの算定
#表4-1
#外力m1､m3．出力m1､m2の変位。B、Cマトリクスを見ること
#
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
n=6 #次数
T=0.5 #sampling time
k1=1.;k2=2.;k3=1.;b1=0.01;b2=0;b3=0.01
#
A=np.zeros((n,n));B=np.zeros((n,2));C=np.zeros((2,n)) #p58 hyou4-1
D=np.eye(n,n);P=np.eye(n,n);Q0=np.eye(n,n)
W=np.zeros(6);Z=np.zeros(6);
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
# #https://analytics-note.xyz/programming/numpy-printoptions/
#np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止
np.set_printoptions(precision=3,  floatmode="fixed")#　=True 指数表記禁止

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
    D=A.dot(D);D=(1/k)*D;E=(1/(k+1))*D
    P=P+D;Q0=Q0+E
    e1=np.sum(np.abs(D))
    #print("k=",k,"error=",e1)

#
Q0=T*Q0;Q=Q0.dot(B)            
#calculate end

np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止
# P and Q matrix was calculated
print("\n -------------P, Q and dc-gain matrix------------")
print("number of recursion=",k)
print("\nP matrix")
print(P)
print("\nQ matrix")
print(Q)
#end
I=np.eye(n,n)         
invP=I-P
invP=LA.inv(invP)
G1=C.dot(invP)
G1=G1.dot(Q)
print("\ndc-gain")
print(G1)
#end