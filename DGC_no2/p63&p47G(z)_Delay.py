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

n=6 #6次系
knum=100 #サンプル数
A=np.zeros(n) #A: a1,a2,....an
B=np.zeros(n) #B: b1,b2,....bn
#U=np.zeros((n,1));Y=np.zeros((n,1))
U=np.zeros(n)
Y=np.zeros((n,1))
YY=np.zeros(knum)# plotのデータアレイ 
YYp=np.zeros(knum)# plotのデータアレイ 
#input u

#
#memo: インデントの追加は「Ctrl + ]」
#入力
Uinp=np.zeros(knum+1)
#Uinp[0]=1. # impulse input
Uinp=np.ones(knum+1) #step input
#
U=np.zeros((n,1));Y=np.zeros((n,1)) #reset
#print("\nU :\n",U)
    
# ケース分けのパラメータ設定
#Case (d)は書籍に無い。for l in range(4):　とすれば描画

Q=np.array([[0],
            [1],
            [0],
            [1],
            [0],
            [1]])
#
C=np.array([1,1,1,1,1,1])

##input A,B
############################
# B1,B2はp64v5.pyの実行結果 #
############################
#A: a1,a2,....an
#B: b1,b2,....bn
# G(z)=(b1z^(n-1)+b2^(n-2)+.....bn))/(z(^n)+a1z^(n-1)+a2^(n-2)+.....+an)
# G(1z^(-1))=(b1z^(-1)+b2^(-2)+.....bn^(-n))/(1+a1z^(-1)+a2^(-2)+.....+an^(-n))
#同伴形の出力結果は上記と逆。注意！
# an,.......a2,a1
# bn,.......b2,a1
#以下は、p47のプログラムを用いて応答をPLOT

#############################
#今回は、L=0.1
#B1
# a1:(An,An-1...A2,A1:)
#  [  0.98511  -4.33945   9.03484 -11.35148   9.08309  -4.38535]#
# Cc1(Bn..B1:)
# # [-0.81086,  3.42126, -6.26091,  6.827,   -4.34356,  1.42747]
#B2
# a2(An,An-1...A2,A1:)
#  [  0.98511  -4.33945   9.03484 -11.35148   9.08309  -4.38535]
# #Cc2(Bn..B1:)
#  [-0.28064,  1.11477, -1.99249,  2.12299, -1.32671,  0.42297]

#これはL=0.2
#B1
#a1:(An,An-1...A2,A1:)#
# [  0.98511,  -4.33945,   9.03484, -11.35148,   9.08309,  -4.38535]
# Cc1(Bn..B1:)
# [-0.56061,  2.40699, -4.43329,  4.86402, -3.10913,  1.0297 ]
#B2
#a2(An,An-1...A2,A1:)
# [  0.98511  -4.33945   9.03484 -11.35148   9.08309  -4.38535]
# Cc2(Bn..B1:)
# [-0.5309,   2.12904, -3.82011,  4.08596, -2.56113,  0.82074]

#これはL=0.3
#B1
#a1:(An,An-1...A2,A1:)
# [  0.98511,  -4.33945,   9.03484, -11.35148,   9.08309,  -4.38535]
# Cc1(Bn..B1:)
# [-0.34158,  1.49668, -2.77686,  3.0678,  -1.97117,  0.65839]
#B2
#a2(An,An-1...A2,A1:)
# [  0.98511  -4.33945   9.03484 -11.35148   9.08309  -4.38535]
# Cc2(Bn..B1:)
# [-0.74992,  3.03934, -5.47654,  5.88218, -3.69909,  1.19205]

#これはL=0.4
#B1
#a1:(An,An-1...A2,A1:)
# [  0.98511  -4.33945   9.03484 -11.35148   9.08309  -4.38535]
# Cc1(Bn..B1:)
# [-0.15451,  0.69345, -1.29739,  1.44458, -0.93359,  0.31478]
#B2
#a2(An,An-1...A2,A1:)
# [  0.98511  -4.33945   9.03484 -11.35148   9.08309  -4.38535]
# Cc2(Bn..B1:)
# [-0.93699,  3.84257, -6.95601,  7.5054,  -4.73667,  1.53566]
#B1,B2はp64v5.pyの実行結果

L=0.4 #PLOTした時のタイトルに値が入る（それ以外の意味はない）

A=np.array([0.98511,  -4.33945,   9.03484, -11.35148,   9.08309,  -4.38535])#同伴形の出力結果
B1=np.array([-0.15451,  0.69345, -1.29739,  1.44458, -0.93359,  0.31478])#同伴形の出力結果
B2=np.array([-0.93699,  3.84257, -6.95601,  7.5054,  -4.73667,  1.53566])#同伴形の出力結果
A=A[::-1];B1=B1[::-1];B2=B2[::-1] # 逆順表示
#print(A)

#応答の算定 B1(z)/A(z)
#y(k)=-{a1y(k-1)+..+any(k-n)}+{b1 u(k-1)+..+bn u(k-n)}
#
y0=0.
for k in range(0,knum):
    for j in range(n-1,0,-1): #一刻み毎に更新
        U[j]=U[j-1]
        Y[j]=Y[j-1]
        #print(U)
            
    #print("\n---------k=",k)
    #print(U)  
    #U[0]=1. #original
    U[0]=Uinp[k] #k番目の入力
    Y[0]=y0 #y0 <---y(k-1)

    y0=B1.dot(U)-A.dot(Y) # (3-11　)y0 <---y(k)
    #y0=B1.dot(U)+A.dot(Y) # (3-11　)y0 <---y(k)
    YY[k]=y0 #YY　はplotのため用意した
    #
    #print(k,y0)
#enk K loop
#
YYp=YYp+YY     
t=np.arange(0,knum)
plt.plot(t, YY, '--*r')
# End for B1

#応答の算定  (1/z)B2(z)/A(z)
#y(k)=-{a1y(k-1)+..+any(k-n)}+{b1 u(k-2)+..+bn u(k-n-1)}
#
y0=0.
for k in range(0,knum):
    for j in range(n-1,0,-1): #一刻み毎に更新
        U[j]=U[j-1]
        Y[j]=Y[j-1]
        #print(U)
            
    #print("\n---------k=",k)
    #print(U)  
    #U[0]=1. #original
    U[0]=Uinp[k+1] #k+1番目の入力
    Y[0]=y0 #y0 <---y(k-1)

    y0=B2.dot(U)-A.dot(Y) # (3-11　)y0 <---y(k)
   # y0=B1.dot(U)+A.dot(Y) # (3-11　)y0 <---y(k)
    YY[k]=y0 #YY　はplotのため用意した
        #
    #print(k,y0)
#enk K loop
#
YYp=YYp+YY #B1出力とB2出力を足し合わせる     
t=np.arange(0,knum)
plt.plot(t, YY, '--*b')
plt.plot(t, YYp, '--*g')

# End for calculation

# 以下はグラフを描くコマンド
plt.title("図4-2拡張 ３連振動号系の応答 L="+str(L), fontname="MS Gothic")
#Ymax = 200
#Ymin = -200
#plt.ylim(Ymin, Ymax)
plt.xlim(0, knum)
plt.ylabel("Responce y(k)")
plt.xlabel("Step(k)")

#
#xp = knum*4/8
#yp = Ymax*7/8  # plt.textの位置座標
#plt.text(xp, yp, "case(a)黒,case(b)赤,case(c)緑", fontname="MS Gothic") 
#plt.ylabel("Responce y(k)")
#plt.xlabel("Step (k)")

# 表示
plt.show() 
    
    
