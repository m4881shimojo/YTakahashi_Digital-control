#digital control 
#デジタル制御　高橋安人
#20231201 shimojo
#20240229 見直し
#p101 List 6-1 リカチ式
#
#n=2,m=2システムのGを求める　G<--2x2

#Riccati eqation 
#m=2の場合　w-->2x2
#P,Q行列は、p57_p101.pyで計算する
#
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
n=2
m=2 # m個のu入力

P=np.zeros((n,n));Q=np.zeros((n,m));C=np.zeros((m,n))
H=np.zeros((n,n));Eh=np.zeros((n,n))
G=np.zeros((m,n))
W0=np.zeros((n,n)) #;w=np.zeros((m,m))
#
knum=50 #収束計算の回数上限
YY=np.zeros((knum,4)) #plot用data array. 行方向にdata
#
Tsample=0.1 #sampling period
p=np.exp(-Tsample)
#
#P,Q行列は、p57_p101.pyで計算する
P=np.array([[1.00000000, 0.09516258],
 [0.00000000, 0.90483742]])
Q=np.array([[0.10000000, 0.00483742],
 [0.00000000, 0.09516258]])
C=np.array([[1,0],
            [0,1]])

#
#################################################
######　    ここからリカチ行列の収束計算     ######
#################################################
#Begin Calculation
k=0; e0=1.0   #e0-->収束条件(for_loopでは利用せず)
Flg=True;Kok=-1 #e0-->収束条件のチェック
#
# J=Integral ( x'W0x+u'wu )dt
#　W0　-->重み状態変数x　nxn Eq.6-12
#　w   -->重み入力u
W0=np.dot(C.T,C) #Eq.6-12　対角行列となる
#W0=np.array([[1,0],
#            [0,1]]) 
#D=np.sum(np.abs(W0))
Dw0=np.sum(np.abs(W0))
#
# 重み w=np.zeros((m,m))　対角行列となる
#w11=0.1
w11=0.01
#w11=0.001 
#w11=0.0001 # 重み
#
w=np.array([[w11,0],
            [0,w11]]) 
H=np.copy(W0) #H=W0 こうしないとHとW0は同一のオブジェクトになる！怖いなー
#
    #while e0>1.0E-8 or k<5:　#収束チェックの時使う
for k in range(0,knum):
    #YY-->for PLOT
    YY[k,0]=G[0,0]; YY[k,1]=G[0,1] 
    YY[k,2]=G[1,0]; YY[k,3]=G[1,1] 
    #
    #E-->収束検証用
    Eh=np.copy(H) #E=H=E
    #
    #P'H(k)P--->A1(working memo)
    A1=np.dot(P.T,np.dot(H,P))
    #P'H(k)q-->B1(working memo)
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
print(";w11=",w11)
# 
#########################################################
#　　　　　　　　　　　　　　　PLOT 　　　　　　　　　　　 #
#########################################################
#
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(6,4.5)) # Figureの初期化
ax1 = plt.subplot()
#
t=np.arange(0,knum)
ax1.plot(t,YY[:,0],'-*r',label="K11")
ax1.plot(t,YY[:,1],'-*g',label="K12")
ax1.plot(t,YY[:,2],'-*m',label="K21")
ax1.plot(t,YY[:,3],'-*c',label="K22")

strg0="重み w={:.5g}".format(w11)
plt.title("図6-1 リカチ式の反復計算過程(P87のサーボ系 T=0.1) :"+strg0, fontname="MS Gothic")
plt.ylabel("Gain")
plt.xlabel("iterative calculation")
#
Ymax=np.amax(G); Ymin=0.0
xp=knum*2/10; yp=Ymax*7/10  #plt.textの位置座標
strg1=" Gain:  K11={:.5g},  K12={:.5g}".format(G[0,0],G[0,1])
plt.text(xp,yp, strg1 ) #
xp=knum*2/10; yp=Ymax*6/10  #plt.textの位置座標
strg1=" Gain:  K21={:.5g},  K22={:.5g}".format(G[1,0],G[1,1])
plt.text(xp,yp, strg1 ) #

plt.minorticks_on()
plt.legend(loc='center right')
plt.grid() #ax1.grid() でも良い
#
plt.show()    