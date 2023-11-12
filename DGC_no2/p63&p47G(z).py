# matrix enshuu 20231102 shimojo
#Digital Control P47
#A(z),B(z)が与えられた時の出力をPLOTする 
#Y=G(z)U
#page 64の結果について出力をPLOTする

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
#input u

#
#memo: インデントの追加は「Ctrl + ]」
#入力
Uinp=np.zeros(knum)
#Uinp[0]=1. # impulse input
Uinp=np.ones(knum) #step input
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
    #今回は、a1:(An,An-1...A2,A1:)
    # [  0.98511  -4.33945   9.03484 -11.35148   9.08309  -4.38535]
    # #Cc1(Bn..B1:)
    #  [-1.09151  4.53603 -8.2534   8.94998 -5.67026  1.85044]

#A: a1,a2,....an
#B: b1,b2,....bn
    #G(z)=(b1z^(n-1)+b2^(n-2)+.....bn))/(z(^n)+a1z^(n-1)+a2^(n-2)+.....+an)
#G(1z^(-1))=(b1z^(-1)+b2^(-2)+.....bn^(-n))/(1+a1z^(-1)+a2^(-2)+.....+an^(-n))
#同伴形の出力結果は上記と逆。
#an,.......a2,a1
#bn,.......b2,a1

A=np.array([0.98511,  -4.33945,   9.03484, -11.35148,   9.08309,  -4.38535])#同伴形の出力結果
B=np.array([-1.09151,  4.53603, -8.2534,   8.94998, -5.67026,  1.85044])#同伴形の出力結果
A=A[::-1];B=B[::-1] # 逆順表示
#print(A)


#応答の算定
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

    y0=B.dot(U)-A.dot(Y) # (3-11　)y0 <---y(k)
    YY[k]=y0 #YY　はplotのため用意した
    #
    #print(k,y0)
#enk K loop
    
#
t=np.arange(0,knum)
plt.plot(t, YY, '--*g')
# End for loop (for l in range(3):)

# 以下はグラフを描くコマンド
plt.title("図4-2拡張 ３連振動号系の応答 L=0", fontname="MS Gothic")
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
    
    
