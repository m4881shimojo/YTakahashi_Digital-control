#digital control 
#デジタル制御　高橋安人
# matrix enshuu 20231031 shimojo
#
#p63無駄時間を含むシステムの係数行列
#list 4-2 以下は"L=0を計算"

import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
n=6 #次数
Tsamp=0.5 #sampling time
L1=0.0 #Delay_time　<--　case (a)

print("\n\n---------------------------------")
print("L1 =",L1)

k1=1.;k2=2.;k3=1.;b1=0.01;b2=0;b3=0.01
#
#list 4-1　の3連振動系 p57
A=np.zeros((n,n));B=np.zeros((n,1));C=np.zeros(n) #C ->vector

P=np.eye(n,n)
A0=np.zeros((n,n));A1=np.zeros((n,n)) #add list 4-1
B1=np.zeros((n,1));B2=np.zeros((n,1)) #add list 4-1

Q0=np.eye(n,n);Q1=np.eye(n,n);Q2=np.eye(n,n) #Q --> Q0,Q1,Q2
D=np.eye(n,n);D1=np.eye(n,n) #D --> D,D1
E=np.eye(n,n);E1=np.eye(n,n) #E --> E,E1

#
#１入出力系にする！そうしないと同伴形が作れない（ここでは+）
A=np.array([[0,1.0,0.,0.,0.,0.],
            [-k1,-b1,k1,b1,0.,0.],
            [0.,0.,0.,1.,0.,0.],
            [k1,b1,-(k1+k2),-(b1+b2),k2,b2],
            [0.,0.,0.,0.,0.,1.],
            [0.,0.,k2,b2,-(k2+k3),-(b2+b3)]])
B=np.array([[0.], #p65に説明
            [1.],
            [0.],
            [1.],
            [0.],
            [1.]])
C=np.array([1.,1.,1.,1.,1.,1.]) #p65に説明
#
########################################################################
#            次からは A,B 行列を P,Q 行列に変換する　　                  #
######################################################################## 
#calculate P,Q matrix
#
#print parameter
np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止
#
e0=1.0E-6 
e1=1.0

P=np.eye(n,n);Q0=np.eye(n,n);Q1=np.eye(n,n)
D=np.eye(n,n);D1=np.eye(n,n)

A0=Tsamp*A
A1=(Tsamp-L1)*A
#print("\nA",A)
#print("\nA0",A0)
#print("\nA1",A1)
#print("\nQ0=\n",Q0)
#print("\nB",B.shape)
#
#
#calculate P,Q0,Q1,Q2 page 57
k=0
while e1>e0:
    k=k+1
    D=A0.dot(D);D=(1/k)*D;E=(1/(k+1))*D  #Eq.(4-4)
    D1=A1.dot(D1);D1=(1/k)*D1;E1=(1/(k+1))*D1 #Eq.(4-4)
    P=P+D;Q0=Q0+E;Q1=Q1+E1  #Eq.(4-4)Eq.(4-5)
    #e1=np.sum(np.abs(D))
    e1=LA.norm(D)
    #end while
#print("k=",k,"error=",e1)
#print("\nP=",P) #表4-1のPと同じになる
#
Q0=Tsamp*Q0
Q1=(Tsamp-L1)*Q1  #Eq.(4-5) page 57
Q2=Q0-Q1
#print("\nQ0,Q1,Q2",LA.norm(Q0),LA.norm(Q1),LA.norm(Q2))
#print("\nQ012",LA.norm(Q1)+LA.norm(Q2))
#print("\nP:\n",P)
#print("\nQ0:\n",Q0)
#print("\nQ1:\n",Q1)
#print("\nQ2:\n",Q2)
#
#print("B :\n",B)
B1=Q1.dot(B);B2=Q2.dot(B) #B1<-- q1, B2<-- q2 (4-3)
#
P_calculated=P; #表4-1のPと同じはず    
#calculate end
#print("\nP=\n",P)
#print("\nQ0=\n",Q0)
print("\nq1=\n",B1) #q1=B1
print("\nq2=\n",B2) #q2=B2
# P and Q matrix was calculated
#end  
#
########################################################################
#            p40　離散系モデル　P,Q,Cをもとに同伴形に導く                  #
######################################################################## 
#状態変数ｘの選び方は唯一ではないことXc=Txとなる変換行列Tを導入
#変換行列Tを選ぶことによって同伴形に変換する(ジョルダン標準形）
#Pc行列の下端の行ベクトルからan,..a2,a1を求める
#
#今回、Pq1, Pq2の同伴形をそれぞれ求める。
#L=0の場合では、Pq2は計算されない！！
#
#n=6
#Pc=np.eye(n,n);Qc=np.eye(n,3);Cc=np.zeros((3,n))
Pc=np.eye(n,n);Qc=np.eye(n,1);Cc1=np.zeros(n)
E=np.eye(n,n);T=np.eye(n,n);T0=np.eye(n,n)
d=np.zeros(n);d1=np.zeros(n)
#Qc=Q0 #Q0 Eq.4-3
#print(".B1.ndim=\n",B1.shape)
q1=B1;q2=B2  #わかりやすいように式の記述に合わせる
#HP basicでは小文字変数の使用ができなかった？
#このため、プログラムが分かりにくくなっている
#--------------------------------------------------
#-----------11111111111111111111111111-------------
#calculate Q1.　P62遅れの前半パート
#-----------11111111111111111111111111-------------
#q=np.zeros((n,1))
P=P_calculated
#print("P :\n",P)
q=q1  #q<--q1 q1は元の値を保つ
#print(".q.shape=\n",q.shape)
print("\nq1=\n",q)
#
#Calculate E (list3-1 page 40)
for j in range(0,n):
    E[:,j]=q[:,0] #Eq. 3-19
    q=P.dot(q)

#print("E :\n",E)
E=LA.inv(E)
d=E[n-1,:] #Eq.3-20 page 38

#print("E-1 :\n",E)
#print("d :\n",d)
#Calcutate Pc,Qc,Cc in Eq.3-18
for j in range(0,n):
    T[j,:]=d    #Eq.3-21
    d=d.dot(P)
#
#print("T :\n",T)
T0=LA.inv(T)
E=P.dot(T0) 
Pc=T.dot(E)  #Pc in Eq.3-18
#Pc=T.dot(P.dot(T0))  #Pc in Eq.3-18
#print("d? :\n",d)

#d=Pc[n-1,:]
#print("d?? :\n",d)
#print("\nPc1 :\n",Pc)
a1=-Pc[n-1,:]   # an,...a1
print("a1:(An,An-1...A2,A1:)\n",a1)
Cc1=C.dot(T0) #bn..b1
print("Cc1(Bn..B1:)\n",Cc1)#(3-18)
#
# 念のためqcを計算してみる
qc1=T.dot(q1)
#print("qc1:\n",qc1)
#
#--------------------------------------------------
#----------2222222222222222222222222222222---------
#calculate Q2.　P62遅れの後半パート
#----------2222222222222222222222222222222---------
#reset
Pc=np.eye(n,n);Qc=np.eye(n,1);Cc2=np.zeros(n)
E=np.eye(n,n);T=np.eye(n,n);T0=np.eye(n,n)
d=np.zeros(n);d2=np.zeros(n)
#
P=P_calculated
q=q2  #q<-q2 q2は元の値を保つ

#print("\nq_NORM=",LA.norm(q))
print("\nq2=\n",q2)

if LA.norm(q)> e0:
#
    #Calculate E
    for j in range(0,n):
        E[:,j]=q[:,0] #Eq. 3-19
        #print("\nE :\n",E)
        q=P.dot(q)
    E=LA.inv(E)
    d=E[n-1,:] #Eq.3-20　E行列の最後行
    
    print("\nd :\n",d)
    #
    #Calcutate Pc,Qc,Cc in Eq.3-18
    for j in range(0,n):
        T[j,:]=d    #Eq.3-21　making T matrix
        #print("\nT :\n",T)
        d=d.dot(P)
    #
    T0=LA.inv(T)
    E=P.dot(T0)
    Pc=T.dot(E)  #Pc in Eq.3-18
    #print("\nPc2\n",Pc)
    #Pc=T.dot(P.dot(T0))  #Pc in Eq.3-18
    #d=Pc[n-1,:] #Eq.3-20　Pc行列の最後行
    a2=-Pc[n-1,:]   # an,...a1
    print("a2(An,An-1...A2,A1:)\n",a2)
    #print("\nC\n",C)
    #print("\nT0\n",T0)
    Cc2=C.dot(T0) #bn..b1
    print("Cc2(Bn..B1:)\n",Cc2)#(3-18)
    #
    # 念のためqcを計算してみる
    qc2=T.dot(q2)
    #print("qc2:\n",qc2)
#
#--------------------------------------------------
#--------------------------------------------------
#calculate  G(z)=(zB1(z)+B2(z))/zA(z)
#B1=Cc1,B2=Cc2
#--------------------------------------------------
#--------------------------------------------------
#
Bz=np.zeros(n+1)
#print("Bz,Cc2",Bz,Cc2)
Bz[0]=Cc1[0] #z^6の項を入れる
#print("B(z)0= ",Bz)#add 20231031
for i in range(1,n):
    Bz[i]=Cc1[i]+Cc2[i-1]
Bz[n]=Cc2[n-1] #z^0の項を入れる
#print("Calculate  G(z)=(zB1(z)+B2(z))/zA(z)")
print("\n\nB(z)= \n",Bz)
###########################################
#F(z)の極を求める
#revD=np.append(Bz,[1.0]) #１ を加える（1+a1*z^6+..)
revD=Bz
revD=revD[::-1] # 逆順表示
pol=np.poly1d(revD) #n多項式関数の決定
#pol.roots  #n多項式の根を求める関数        
print("\npolynominal:\n ",pol)
# Solve f(x) == 0.
polSolv=pol.roots
#np.set_printoptions(formatter={'float': '{:.2f}'.format})
np.set_printoptions(precision=5, floatmode="fixed")
print("\nroots:\n ", polSolv)
print("\nend ")
