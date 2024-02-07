#digital control 
#デジタル制御　高橋安人
# matrix enshuu 20231031 shimojo
#20240107 n
#p63無駄時間を含むシステムの係数行列
#list 4-2 以下は"L=0を計算"

import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
#print parameter
np.set_printoptions(precision=4, suppress=True)#　=True 指数表記禁止
#
n=6 #次数
Tsamp=0.5 #sampling time
L1=0.3 #Delay_time　<--　case (a)

print("\n\n------------ L1=",L1,"-------------------")

km1=1.;km2=2.;km3=1.;bm1=0.01;bm2=0;bm3=0.01 #3連振動系の物理パラメータ
#
#list 4-1　の3連振動系 p57
A=np.zeros((n,n));B=np.zeros((n,1));C=np.zeros(n) #C ->vector
#
#１入出力系にする！そうしないと同伴形が作れない（ここでは+）
A=np.array([[0,1.0,0.,0.,0.,0.],
            [-km1,-bm1,km1,bm1,0.,0.],
            [0.,0.,0.,1.,0.,0.],
            [km1,bm1,-(km1+km2),-(bm1+bm2),km2,bm2],
            [0.,0.,0.,0.,0.,1.],
            [0.,0.,km2,bm2,-(km2+km3),-(bm2+bm3)]])
B=np.array([[0.], #p65に説明
            [1.],
            [0.],
            [1.],
            [0.],
            [1.]])
#C=np.array([1.,1.,1.,1.,1.,1.]) #p65に説明
C=np.array([1.,1,1.,1,1.,1]) #

print("A=\n",A)
print("B=\n",B)
print("C=\n",C)
########################################################################
#            次からは A,B 行列を P,Q 行列に変換する　　                  #
######################################################################## 
#calculate P,Q matrix
e0=1.0E-8;e1=1.0 #誤差許容値
P=np.eye(n,n);Q0=np.eye(n,n);Q1=np.eye(n,n)# 単位行列！
Dpq=np.eye(n,n);D1pq=np.eye(n,n) # 単位行列！
Epq=np.zeros((n,n));E1pq=np.zeros((n,n)) # 単位行列！

A0=Tsamp*A      #Q0=Q1+Q2
A1=(Tsamp-L1)*A #Q1 区間
#
#calculate P,Q0,Q1,Q2 page 57
#Epq,E1pq <--- working
k=0
while e1>e0:
    k=k+1
    Dpq=A0.dot(Dpq);Dpq=(1/k)*Dpq;Epq=(1/(k+1))*Dpq  #Eq.(4-4)
    D1pq=A1.dot(D1pq);D1pq=(1/k)*D1pq;E1pq=(1/(k+1))*D1pq #Eq.(4-4)
    P=P+Dpq;Q0=Q0+Epq;Q1=Q1+E1pq  #Eq.(4-4)Eq.(4-5)
    e1=np.sum(np.abs(Dpq))
    #e1=LA.norm(Dpq)
    #end while
#print("\nP=",P) #表4-1のPと同じになる
#(4-5)で最後にTを掛ける処理
Q0=Tsamp*Q0 # 
Q1=(Tsamp-L1)*Q1  #Eq.(4-5) page 57
Q2=Q0-Q1 #calculate Q0=Q1+Q2

q1=np.dot(Q1,B);q2=np.dot(Q2,B)
#
P_calculated=np.copy(P); #表4-1のPと同じはず    
#calculate end
print("\n-----x(k+1)=Px(k)+q1u(k)+q2u(k-1)-----")
print("P=\n",P)
print("q1=\n",q1) #
print("q2=\n",q2) #
# P and Q matrix was calculated
#end  
#
########################################################################
# p40　離散系モデル　P,Q,Cをもとに同伴形(正準形)に導く                   #
######################################################################## 
#
#今回、Pq1, Pq2の同伴形(正準形)をそれぞれ求める。
#L=0の場合では、q2は計算されない！！
#
Pc=np.zeros((n,n));Qc=np.zeros((n,1))
E=np.zeros((n,n));T=np.zeros((n,n))
#d=np.zeros(n)#;d1=np.zeros(n)
ac1=np.zeros(n);ac2=np.zeros(n) #念のためreset
Cc1=np.zeros(n);Cc2=np.zeros(n) #念のためreset

#--------------------------------------------------
#-----------11111111111111111111111111-------------
#calculate Q1.　u(k)の部分
#         x(k+1)=Px(k)+q1u(k)+q2u(k-1)
#-----------11111111111111111111111111-------------
#
P=np.copy(P_calculated)
q=np.copy(q1)  #q<--q1 q1は元の値を保つ
#
#Calculate E (list3-1 page 40)
for j in range(0,n):
    E[:,j]=q[:,0] #Eq. 3-19
    q=np.dot(P,q)

#print("E :\n",E)
invE=np.linalg.inv(E)
d=invE[n-1,:] #Eq.3-20 page 38

#Calcutate Pc,Qc,Cc in Eq.3-18
for j in range(0,n):
    T[j,:]=d    #Eq.3-21
    d=np.dot(d,P)
#
invT=np.linalg.inv(T)
Pc=np.dot(T,np.dot(P,invT))#Pc in Eq.3-18

print("\n-----P,q,Cをもとに同伴形(正準形)を導出----- ")
ac1=-Pc[n-1,:]   # an,...a1
print("\nA1(z):(an,an-1...a2,a1:)\n",ac1)
#
Cc1=np.dot(C,invT)
print("\nB1(z):(bn..b1:)\n",Cc1) #(3-18)
#
# 念のためqcを計算してみる
qc1=np.dot(T,q1)#(3-18)
#print("qc1:\n",qc1)
#
#--------------------------------------------------
#----------2222222222222222222222222222222---------
#calculate Q2.　u(k-1)の部分
#         x(k+1)=Px(k)+q1u(k)+q2u(k-1)
#----------2222222222222222222222222222222---------
#
P=np.copy(P_calculated)
q=np.copy(q2)  #q<-q2 q2は元の値を保つ

if LA.norm(q)> e0: #L1=0　だとq2=0で無意味（逆数計算不可）
#
    #Calculate E
    for j in range(0,n):
        E[:,j]=q[:,0] #Eq. 3-19
        q=np.dot(P,q)
    #
    invE=np.linalg.inv(E)
    d=invE[n-1,:] #Eq.3-20 page 38
    #Calcutate Pc,Qc,Cc in Eq.3-18
    for j in range(0,n):
        T[j,:]=d    #Eq.3-21　making T matrix
        d=np.dot(d,P)
    
    invT=np.linalg.inv(T)
    Pc=np.dot(T,np.dot(P,invT))
    #
    ac2=-Pc[n-1,:]   # an,...a1
    print("\nA2(z):(an,an-1...a2,a1:)\n",ac2)
    #Cc1=C.dot(T0) #bn..b1
    Cc2=np.dot(C,invT)
    print("\nB2(z):(bn..b1:)\n",Cc2) #(3-18)
        
    #
    # 念のためqcを計算してみる
    #qc2=T.dot(q2)
    qc2=np.dot(T,q2)
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
#Bz[0]=Cc2[0] #z^0の項を入れる
Bz[0]=Cc2[0] #z^0の項を入れる
#print("B(z)0= ",Bz)#add 20231031
for i in range(1,n):
    Bz[i]=Cc1[i-1]+Cc2[i]
Bz[n]=Cc1[n-1] #z^0の項を入れる
print("\n----- G(z)=(zB1(z)+B2(z))/zA(z) -----")
print("\nB(z)= \n",Bz)
###########################################
#F(z)の極を求める
#revD=np.append(Bz,[1.0]) #１ を加える（1+a1*z^6+..)
revD=Bz
revD=revD[::-1] # 逆順表示
pol=np.poly1d(revD,variable = "z") #n多項式関数の決定
#pol.roots  #n多項式の根を求める関数        
print("\npolynominal:\n ",pol)
# Solve f(x) == 0.
polSolv=pol.roots
#np.set_printoptions(formatter={'float': '{:.2f}'.format})
np.set_printoptions(precision=5, floatmode="fixed")
print("\nroots:\n ", polSolv)
print("\nend ")
