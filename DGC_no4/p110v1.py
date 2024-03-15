#digital control 
#デジタル制御　高橋安人
#20231210 shimojo
#20240303 見直し
#p108 6.3節
#表6-3　成功例
#
#
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
n=3
m=1 # m個のu入力
P=np.zeros((n,n));Q=np.zeros((n,m))
W0=np.zeros((n,n));w=np.zeros((m,m))
G=np.zeros((m,n));H=np.zeros((n,n))
Eh=np.zeros((n,n))
#
knum=30 #収束計算の回数上限
YY=np.zeros((knum,n)) #plot用data array. 行方向にdata
#
T=4 #sampling period
T1=10;T2=6;r=T2/T1
p1=np.exp(-T/T1);p2=np.exp(-T/T2)
b1=1-(p1-r*p2)/(1-r)
b2=p1*p2-(p2-r*p1)/(1-r)
b3=p1*p2-(p2-r*p1)/(1-r)
p3=(b1*p2+b2)/(1-p2)
#
#図6-4のシステム
q1=-b1;q2=b1;q3=1-p2
P=np.array([[1,p1, p3],
            [0,p1,p3],
            [0,0,p2]])
Q=np.array([[q2],
            [q2],
            [q3]])
C=np.array([[1,0,0]])

#
#################################################
######　    ここからリカチ行列の収束計算     ######
#################################################
# J=Integral ( x'W0x+u'wu )dt
#　W0　-->重み状態変数x　nxn Eq.6-12
#　w   -->重み入力u
W0=np.dot(C.T,C) #Eq.6-12　対角行列となる
Dw0=np.sum(np.abs(W0))
#
w11=0.01 # 重み w=np.zeros((m,m))　対角行列となる
w=np.eye(m,m)
w=np.dot(w11,w) # 重み(mxm)
#w11=0.001 # 重み
#w11=0.0001 # 重み
#
print("\n -------------P, Q matrix------------")
#print("w={:.5g}".format(w))
print("w=",w)
print("P matrix\n",P)
print("Q matrix\n",Q)
print("q1={:.4g},q2={:.4g},q3={:.4g}".format(q1,q2,q3))
# 
H=np.copy(W0) #H=W0 こうしないとHとW0は同一のオブジェクトになる！怖いなー
k=0; e0=1.0   #e0-->収束条件(for_loopでは利用せず)
Flg=True;Kok=-1 #e0-->収束条件のチェック
#
    #while e0>1.0E-8 or k<5:　#収束チェックの時使う
for k in range(0,knum):
    #YY-->for PLOT
    YY[k,0]=G[0,0]
    YY[k,1]=G[0,1] # G--> 3x1
    YY[k,2]=G[0,2] # G--> 3x1
    #
    #E-->収束検証用
    Eh=np.copy(H) #
    #
    #P'H(k)P--->A(working memo)
    A1=np.dot(P.T,np.dot(H,P))
    #P'H(k)q-->B(working memo)
    B1=np.dot(P.T,np.dot(H,Q))
    #(w+q'H(k)q)-->W1(working memo)
    W1=w+np.dot(Q.T,np.dot(H,Q))
    #(W1)^(-1)q'H(k)P-->G
    invW1=np.linalg.inv(W1) #逆行列を求める
    G=np.dot(invW1,np.dot(Q.T,np.dot(H,P))) #Eq. 6-14
    #
    H=A1-np.dot(B1,G)+W0  #Eq.6-13
    Eh=H-Eh; e0=np.sum(np.abs(Eh))/Dw0 #H(k)の収束状況を確認のためe0を計算
    if (e0<1.0E-8) and Flg : Kok=k;Flg=False
    #収束条件チェック。但し、while_loopで使う条件。for_loopでなKnum回数計算する
    #End for k
    #
if Kok<0 :print("収束e0条件に達せずGain=\n",G)
else:print("収束Num=",Kok);print("Gain=",G)
print(";w=",w)
# 
#########################################################
#　　　　　　　　　　　　　　　PLOT 　　　　　　　　　　　 #
#########################################################
#
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(6,5)) # Figureの初期化
ax1 = plt.subplot()
#
t=np.arange(0,knum)
ax1.plot(t,YY[:,0],'-*r',label="k0")
ax1.plot(t,YY[:,1],'-*b',label="k1")
ax1.plot(t,YY[:,2],'-*g',label="k2")

#strg0="収束反復計算回数={:.3g}, e0={:.3g}".format(Kok,e0)
strg0="重み w={:.3g}".format(w11)

plt.title("図6-1 リカチ式の反復計算過程(P107のtype-0型 T=4) :"+strg0, fontname="MS Gothic")
plt.ylabel("Gain")
plt.xlabel("iterative calculation")
#
Ymax=np.amax(G); Ymin=0.0
xp=knum*3/10; yp=Ymax*8/10  #plt.textの位置座標
strg1=" Gain: K0={:.5g}, K1={:.5g}, K2={:.5g}".format(G[0,0],G[0,1],G[0,2])
plt.text(xp,yp, strg1 ) #

plt.minorticks_on()
plt.legend(loc='lower right')
plt.grid() #ax1.grid() でも良い
#
plt.show()    