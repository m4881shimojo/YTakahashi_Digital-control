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
n=2 #次数
m=1 #入力
T_sample=4 #sampling time
#k1=1.;k2=2.;k3=1.;
a1=0.2672;a2=0.017
b1=0.;b2=1
#
A=np.zeros((n,n));B=np.zeros((n,m));C=np.zeros((m,n)) #p58 hyou4-1
Dpq=np.eye(n,n);P=np.eye(n,n);Q0=np.eye(n,n)
Epq=np.eye(n,n)
#
A=np.array([[0,1.0,],
            [-a2,-a1]])
B=np.array([[0.], #p58 hyou4-1
            [1.]])
C=np.array([[0,1.]])

#正準形
#Pc=np.eye(n,n);Qc=np.eye(n,m);Cc=np.zeros(n) # zero行列
Pc=np.zeros((n,n));Qc=np.zeros((n,m));Cc=np.zeros(n) # zero行列
Ec=np.eye(n,n);Tc=np.eye(n,n);T0=np.eye(n,n)
Dc=np.zeros(n)

#input A, B, C matrix
#calculate P,Q matrix
k=0
e0=0.000001 #1.0E-6
e1=1.0
A=T_sample*A
while e1>e0:
    k=k+1
    Dpq=A.dot(Dpq);Dpq=(1/k)*Dpq;Epq=(1/(k+1))*Dpq
    P=P+Dpq;Q0=Q0+Epq
    e1=np.sum(np.abs(Dpq))
    #print("k=",k,"error=",e1)
#
Q0=T_sample*Q0; Q=Q0.dot(B)            
#calculate end

np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止

# P and Q matrix was calculated
print("\n -------------P, Q and dc-gain matrix------------")
print("\nP matrix",P)
print("\nQ matrix",Q)
print("\nC matrix",C)
#end
I=np.eye(n,n)         
invP=I-P
invP=LA.inv(invP)
G1=C.dot(invP)
G1=G1.dot(Q)
print("\ndc-gain")
print(G1)

#end
########################################################
#同伴形（正準形）への変換
########################################################
Q_original=Q
#Qc=Q #書籍の記述
#
#Calculate E
for j in range(0,n):
    Ec[:,j]=Q[:,0] #Eq. 3-19
    Q=P.dot(Q)
E=LA.inv(Ec) #E=inv_E
Dc=Ec[n-1,:] #Eq.3-20
#
#Calcutate Pc,Qc,Cc in Eq.3-18
# make T matrix
for j in range(0,n):
    Tc[j,:]=Dc    #Eq.3-21
    Dc=Dc.dot(P)
#Calcutate Pc
T0=LA.inv(Tc) #T0=1/T
Ec=P.dot(T0) #P 1/T
Pc=Tc.dot(Ec) #(TP1/T) Pc in Eq.3-18

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

Qc=Tc.dot(Q_original)
print("\nqc  :")
print(Qc)

print("\nMatrix T:")
print(Tc)

###########################################
# original data
###########################################
print("\n\n-------------P,Q,C-------------------")
print("P  :")
print(P)


print("\nq  :")
print(Q_original)

print("\nc  :")
print(Cc)

