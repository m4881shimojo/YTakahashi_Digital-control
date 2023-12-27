#digital control 
#デジタル制御　高橋安人
#20231201 shimojo
#p101 List 6-1 リカチ式
#
#n2,m2システムのGを求める　G<--2x2

#Riccati eqation 
#m=2の場合　w-->2x2
#P,Q行列は、p57_p101.pyで計算する
#LQ制御
#m入出力系に拡張
#m入出力系に拡張
#
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
n=2
m=2 # m個のu入力
A=np.zeros((n,n));B=np.zeros((n,))
C=np.zeros((m,n));E=np.zeros((n,n))
G=np.zeros((m,n));H=np.zeros((n,n));P=np.zeros((n,n));Q=np.zeros((n,m))
W0=np.zeros((n,n)) ;w=np.zeros((m,m))
#
knum=50 #収束計算の回数上限
YY=np.zeros((knum,2)) #plot用data array. 行方向にdata
#
T=0.1 #sampling period
p=math.exp(-T)
#
#図5-3のシステム（伝達関数より求めたもの<--書籍）
#P[0,0]=1;P[0,1]=1-p;P[1,0]=0;P[1,1]=p # P--> 2x2
#Q[0,0]=p+T-1;Q[1,0]=1-p # Q--> 2x1
#C[0,0]=1;C[0,1]=0 # C--> 1x2

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
D=np.sum(np.abs(W0))
#
w11=0.01
#w11=0.001 # 重み w=np.zeros((m,m))　対角行列となる
#w11=0.0001 # 重み
#
w=np.array([[w11,0],
            [0,w11]]) 
H=np.copy(W0) #H=W0 こうしないとHとW0は同一のオブジェクトになる！怖いなー

############   memo ##############################################
#https://python-no-memo.blogspot.com/2020/07/numpyndarray_13.html
#代入演算子で複製すると参照渡しとなり同じメモリ領域を参照するため
# コピー先のオブジェクトとコピー元のオブジェクトに互いに影響される。
# コピー先のオブジェクトをコピー元のオブジェクトとリンクさせたくない場合は
# ndarray.copyやnumpy.copy、copyモジュールを使う。
############   以上　 #############################################
#
    #while e0>1.0E-8 or k<5:　#収束チェックの時使う
for k in range(0,knum):
    #print("k=",k,G)
    #YY-->for PLOT
    YY[k,0]=G[0,0]
    YY[k,1]=G[0,1] # G--> 2x1
    #
    #E-->収束検証用
    E=np.copy(H);k=k+1 #E=H=E
    #
    #P'H(k)P--->A(working memo)
    A=np.dot(P.T,np.dot(H,P))
    #P'H(k)q-->B(working memo)
    B=np.dot(P.T,np.dot(H,Q))
    #(w+q'H(k)q)-->W1(working memo)
    W1=w+np.dot(Q.T,np.dot(H,Q))
    #(W1)^(-1)q'H(k)P-->G
    invW1=np.linalg.inv(W1) #逆行列を求める
    G=np.dot(invW1,np.dot(Q.T,np.dot(H,P))) #Eq. 6-14
    #
    H=A-np.dot(B,G)+W0  #Eq.6-13
    E=H-E; e0=np.sum(np.abs(E))/D #H(k)の収束状況を確認のためe0を計算
    if (e0<1.0E-8) and Flg : Kok=k;Flg=False
    #収束条件チェック。但し、while_loopで使う条件。for_loopでなKnum回数計算する
    #End for k
    #
if Kok<0 :print("収束e0条件に達せず失敗")
else:print("収束Num=",Kok);print("Gain=",G)
print("Gain=",G)
# 
#########################################################
#　　　　　　　　　　　　　　　PLOT 　　　　　　　　　　　 #
#########################################################
#
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(8,6)) # Figureの初期化
ax1 = plt.subplot()
#
t=np.arange(0,knum)
ax1.plot(t,YY[:,0],'-*r')
ax1.plot(t,YY[:,1],'-*g')

#strg0="収束反復計算回数={:.3g}, e0={:.3g}".format(Kok,e0)
#strg0="重み w={:.3g}".format(w)
strg0="重み w={:.5g}".format(w11)
plt.title("図6-1 リカチ式の反復計算過程(P87のサーボ系 T=0.1) :"+strg0, fontname="MS Gothic")
plt.ylabel("Gain K1,K2")
plt.xlabel("repeatation")
#
Ymax=np.amax(G); Ymin=0.0
xp=knum*3/10; yp=Ymax*8/10  #plt.textの位置座標
strg1=" Gain: K1={:.5g},K2={:.5g}".format(G[0,0],G[0,1])
plt.text(xp,yp, strg1 ) #

# x軸に補助目盛線を設定
ax1.grid(which = "major", axis = "x", color = "blue", alpha = 0.8,
        linestyle = "--", linewidth = 1)
# y軸に目盛線を設定
ax1.grid(which = "major", axis = "y", color = "green", alpha = 0.8,
        linestyle = "--", linewidth = 1)
# 補助目盛を表示
plt.minorticks_on()
plt.grid(which="minor", color="gray", linestyle=":")
#
plt.show()    