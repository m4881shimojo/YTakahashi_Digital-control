# matrix enshuu 20231102 shimojo
#Digital Control P47
#A(z),B(z)が与えられた時の出力をPLOTする 
#Y=G(z)U
#page 64の結果について出力をPLOTする
#G(z)=B1(z)/A(z) + (1/z)B2(z)/A(z) としてプロットしてみる

import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA # Linear algebra
#
#今回の想定はp63のlist4-2の結果チェック
#自分の結果と書籍の結果が合わない。このため応答波形を描くことにする
#例題は付録A-4の例7、3連振動系である。
#運動方程式から連続系のA,b､cを求め、離散系のp､q､cに変換、そして同伴形へ変換する
#同伴形は、1入出力の変換を行う方法をp37で述べた。このため例7の3連振動系1入出力として変換する(p65)
#
#np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止
np.set_printoptions(precision=3,  floatmode="fixed")
#
n=6 #6次系
knum=50 #サンプル数
A=np.zeros(n) #A: a1,a2,....an
B=np.zeros(n) #B: b1,b2,....bn
#
Y=np.zeros((n,1))
U=np.zeros((n,1));Y=np.zeros((n,1)) 
#PLOT用の配列
YY=np.zeros(knum)# plotのデータアレイ 
YYp=np.zeros(knum)# plotのデータアレイ 
#
#memo: インデントの追加は「Ctrl + ]」
#
#input u
#
Uinp=np.zeros(knum);Uinp[5]=1. # impulse input
Uinp=np.ones(knum) #step input
for i in range(0,5):
    Uinp[i]=0

#Uinp[20]=2. #打撃を与える（遊び）
#Uinp[30]=2. #打撃を与える
#
#応答の計算
################################
# B1,B2はpage63v2.pyの実行結果 #
# Aは全てに共通(変化なし)      #
################################
#A: a1,a2,....an
#B: b1,b2,....bn
# G(z)=(b1z^(n-1)+b2^(n-2)+.....bn))/(z(^n)+a1z^(n-1)+a2^(n-2)+.....+an)
#############################
#同伴形の出力結果は上記と逆。注意！
# an,.......a2,a1
# bn,.......b2,a1
#
#今回は、
L=0.0
A=np.array([0.98511,  -4.33945,   9.03484, -11.35148,   9.08309,  -4.38535])#同伴形の出力結果
B1=np.array([-1.0915,  4.536,  -8.2534,  8.95,   -5.6703,  1.8504])#同伴形の出力結果
B2=np.array([0., 0., 0., 0., 0.,  0.])#同伴形の出力結果

#これは
L=0.1
A=np.array([0.98511,  -4.33945,   9.03484, -11.35148,   9.08309,  -4.38535])#同伴形の出力結果
B1=np.array([-0.8109,  3.4213, -6.2609,  6.827,  -4.3436,  1.4275])#同伴形の出力結果
B2=np.array([-0.2806,  1.1148, -1.9925,  2.123,  -1.3267,  0.423])#同伴形の出力結果

#これは
L=0.2
A=np.array([0.98511,  -4.33945,   9.03484, -11.35148,   9.08309,  -4.38535])#同伴形の出力結果
B1=np.array([-0.5606,  2.407,  -4.4333,  4.864,  -3.1091,  1.0297])#同伴形の出力結果
B2=np.array([-0.5309,  2.129,  -3.8201,  4.086,  -2.5611,  0.8207])#同伴形の出力結果
#これは
L=0.3
A=np.array([0.98511,  -4.33945,   9.03484, -11.35148,   9.08309,  -4.38535])#同伴形の出力結果
B1=np.array([-0.3416,  1.4967, -2.7769,  3.0678, -1.9712,  0.6584])#同伴形の出力結果
B2=np.array([-0.7499,  3.0393, -5.4765,  5.8822, -3.6991,  1.192])#同伴形の出力結果

#これは
L=0.4
A=np.array([0.98511,  -4.33945,   9.03484, -11.35148,   9.08309,  -4.38535])#同伴形の出力結果
B1=np.array([-0.1545,  0.6935, -1.2974,  1.4446, -0.9336,  0.3148])#同伴形の出力結果
B2=np.array([-0.937,   3.8426, -6.956,   7.5054, -4.7367,  1.5357])#同伴形の出力結果

##################################################################
L=0.1
A=np.array([0.98511,  -4.33945,   9.03484, -11.35148,   9.08309,  -4.38535])#同伴形の出力結果
B1=np.array([-0.8109,  3.4213, -6.2609,  6.827,  -4.3436,  1.4275])#同伴形の出力結果
B2=np.array([-0.2806,  1.1148, -1.9925,  2.123,  -1.3267,  0.423])#同伴形の出力結果
##################################################################
#
#   B1,B2はpage63v2.pyの実行結果
#

A=A[::-1];B1=B1[::-1];B2=B2[::-1] # 逆順表示(an,..a1)-->(a1,a2,....an)
#
YY=np.zeros(knum)#clear: plotのデータアレイ 
YYp=np.zeros(knum)#clear: plotのデータアレイ
y0=np.zeros(1) # こうしないとコンパイラでwaning?(理由はよくわからない)

################################################
#       G(z)=B1(z)/A(z) + B2(z)/(zA(z))        *
############## B1B1B1B1B1B1B1B1B1B1B1###########
################################################
#G(z)=B1(z)/A(z) + B2(z)/(zA(z))
#応答の算定 B1(z)/A(z)
#
#
U=np.zeros((n,1));Y=np.zeros((n,1)) #reset
y0[0]=0.
for k in range(0,knum):
    for i in range(n-1,0,-1): #一刻み毎に更新
        Y[i]=Y[i-1]
        U[i]=U[i-1]
    #
    U[0]=Uinp[k] #k番目の入力
    Y[0]=y0[0] #y0 <---y(k-1)
    #print("\n---------k=",k)
    #print(U.T,Y.T) 

    #y0=B1.dot(U)-A.dot(Y) # (3-11　)y0 <---y(k)
    y0=np.dot(B1,U)-np.dot(A,Y) # (3-11　)y0 <---y(k)
    YY[k]=y0[0] #YY　はplotのため用意した
    YYp[k]=y0[0] #YYp　はplotのため用意した
    #
    #print(k,y0)
#enk K loop
t=np.arange(0,knum)
plt.plot(t, YY, '--or',label="Q1")
# End for B1
################################################
#       G(z)=B1(z)/A(z) + B2(z)/(zA(z))        *
############## B2B2B2B2B2B2B2B2B2B2B2###########
################################################
#応答の算定  (1/z)B2(z)/A(z)
#B2は、一刻み遅れる
U=np.zeros((n,1));Y=np.zeros((n,1)) #reset
Ndelay=1 #Delay個数(1以上の場合は未確認)
y0[0]=0.
for k in range(0,knum):
    for i in range(n-1,0,-1): #一刻み毎に更新
        Y[i]=Y[i-1]
        U[i]=U[i-1]
        #   
    if k<Ndelay: U[0]=0.
    else:
        U[0]=Uinp[k-Ndelay] #delay 入力
    #
    Y[0]=y0[0] #y0 <---y(k-1)
    
     #print("\n---------k=",k)
    #print(U.T,Y.T) 

    #y0=B1.dot(U)-A.dot(Y) # (3-11　)y0 <---y(k)
    y0=np.dot(B2,U)-np.dot(A,Y) # (3-11　)y0 <---y(k
    YY[k]=y0[0] #YY　はplotのため用意した
    YYp[k]=YYp[k]+y0[0]
    #
    #print(k,y0)
#enk K loop
t=np.arange(0,knum)
plt.plot(t, YY, '--*b',label="Q2")
plt.plot(t, YYp, '--.c',label="Q1+Q2")

# End for calculation

# 以下はグラフを描くコマンド
plt.title("図4-2拡張 ３連振動号系の応答 L="+str(L), fontname="MS Gothic")
#Ymax = 200
#Ymin = -200
#plt.ylim(Ymin, Ymax)
plt.xlim(0, knum)
plt.ylabel("Responce y(k)")
plt.xlabel("Step(k)")
plt.legend() #labelの表示
plt.grid()

#
#xp = knum*4/8
#yp = Ymax*7/8  # plt.textの位置座標
#plt.text(xp, yp, "case(a)黒,case(b)赤,case(c)緑", fontname="MS Gothic") 
#plt.ylabel("Responce y(k)")
#plt.xlabel("Step (k)")

# 表示
plt.show() 
    
    
