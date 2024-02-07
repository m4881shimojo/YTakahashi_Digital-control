#digital control 
#デジタル制御　高橋安人
#p57ver2 3連振動系　教科書例題通りに行った
#list4-1 Pとqの算定
#p57 微分方程式と等価の差分式への変換
#p42 応答の算定を利用
#p239　｢例７｣図A-3(d)の3連振動系を例として利用する
#p42の例では、B: 6x2,C: 2x6、となっている。
#u1,u3入力しか利用しないので良いから？
#ここではB: 6x3,C: 3x6　としてu2も使えるようにした
#page 239

import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA

n=6 #次数
m=3 #入力数
Tsample=0.5 #sampling time
k1=1.;k2=2.;k3=1.;b1=0.01;b2=0;b3=0.01
#
A=np.zeros((n,n));B=np.zeros((n,m));C=np.zeros((m,n))
P=np.eye(n,n);Q0=np.eye(n,n);Dpq=np.eye(n,n)
#
#
A=np.array([[0,1.0,0.,0.,0.,0.],
            [-k1,-b1,k1,b1,0.,0.],
            [0.,0.,0.,1.,0.,0.],
            [k1,b1,-(k1+k2),-(b1+b2),k2,b2],
            [0.,0.,0.,0.,0.,1.],
            [0.,0.,k2,b2,-(k2+k3),-(b2+b3)]])
B=np.array([[0.,0.,0.],
            [1.,0.,0.],
            [0.,0.,0.],
            [0.,1.,0.],# 0,1,0 <- change 20231028
            [0.,0.,0.],
            [0.,0.,1.]]) #<=2.0
C=np.array([[1.,0.,0.,0.,0.,0.],
            [0.,0.,1.0,0.,0.,0.], #<=1.2
            [0.,0.,0.,0.,1.,0.]])
#C行列をp240 (A-13)と同じにした            
#C=np.array([[1.,0.,0.,0.,0.,0.],
#            [0.,0.,1.,0.,0.,0.], #<=1.0
#            [0.,0.,0.,0.,1.,0.]])
#
# check input matrix
# #https://analytics-note.xyz/programming/numpy-printoptions/
#np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止
np.set_printoptions(precision=3,  floatmode="fixed")#　=True 指数表記禁止
#input A, B, C matrix
print("\n -------------A, B and C matrix------------")
print("\nA matrix\n",A)
print("\nB matrix\n",B)
print("\nC matrix\n",C)

########################################################################
#            次からは A,B 行列を P,Q 行列に変換する　　                  #
######################################################################## 
#calculate P,Q matrix
#list4-1 Pとqの算定
k=0
e0=1.0E-8;e1=1.0 #誤差許容値
A1=Tsample*A #A1---> working memo
while e1>e0:
    k=k+1
    Dpq=A1.dot(Dpq);Dpq=(1/k)*Dpq;Epq=(1/(k+1))*Dpq
    P=P+Dpq;Q0=Q0+Epq
    e1=np.sum(np.abs(Dpq))
    #
# while end
# get Q matrix          
Q0=Tsample*Q0; Q=np.dot(Q0,B)            
#calculate end
# P and Q matrix was calculated

# check calcutated matric P,Q
np.set_printoptions(precision=5, suppress=True)#　=True 指数表記禁止
# P and Q matrix was calculated
print("\n -------------P, Q and dc-gain matrix------------")
print("number of recursion=",k)
print("\nP matrix\n",P)
print("\nQ matrix\n",Q)
#end

#dc gain
#p58とは若干異なる。ｐ５８では、C,Bが異なる値を用いてるためか？
I=np.eye(n,n)
invIP=LA.inv((I-P))   
G1=C.dot(invIP)
G1=G1.dot(Q)  
print("dc gain\n",G1)

########################################################################
#            次からは  P,Q 行列を使って応答を求めます　                  #
########################################################################
knum=50 #サンプリング総数
lcase=3 # case 数（図４－２でm1への入力とm2への入力で個別plotするため）

#YY(サンプル数[knum],変位[y1,y2,y3]、ケース番号[lcase])
#
YY=np.zeros((knum,3,lcase)) #行方向に時系列データが作られる
#
#t = np.arange(0, knum) #plotの時のX軸 
#
for l in range(lcase):
    X=np.zeros((6,1)) # reset X array
    if l==0: input1=1.0;input2=0.0;input3=0.0 #m1への入力
    if l==1: input1=0.0;input2=1.0;input3=0.0 #m3への入力
    if l==2: input1=0.0;input2=0.0;input3=1.0 #m3への入力


    #U step input
    U=np.array([[input1],  #u1 input
                [input2],  #u2 input
                [input3]]) #u3 input
    # list 3-2 page42
    for k in range(0,knum):
        X=np.dot(P,X)+np.dot(Q,U)        
        YY[k,:,l]=np.transpose(np.dot(C,X))
        #print (YY[k])
    #
    #plt.plot(t,YY[:,2,l],'-or') #y1=YY[:,0],y2=YY[:,1],y3=YY[:,2]


########################################################################
#            次からはプロットするためshimojoが追加                       #
########################################################################    
#  
#参考： https://www.yutaka-note.com/entry/2020/01/02/232925
#参考：https://pystyle.info/matplotlib-line-properties/　ステップplot

fig = plt.figure(figsize = (9,5),tight_layout=True) 
t=np.arange(0,knum) #plotの時のX軸
Ymin,Ymax=-1,6

plt.suptitle("図4-2 3連振動系のステップ応答", fontname="MS Gothic")

#y1=YY[:,0,lcase],y2=YY[:,1,lcase],y3=YY[:,2,lcase]
#plt.subplot(121,title="m1の変位", fontname="MS Gothic") 

plt.subplot(131,title="Displacement of m1") 
plt.plot(t,YY[:,0,0],'-or',label="inp_m1") 
plt.plot(t,YY[:,0,1],'-*b',label="inp_m2")
plt.plot(t,YY[:,0,2],'-*g',label="inp_m3")
plt.ylim(Ymin,Ymax)
plt.ylabel("Responce y(k)")
plt.xlabel("Step (k)")
plt.grid()
plt.legend() #labelの表示

plt.subplot(132,title="Displacement of m2") #　1行目
plt.plot(t,YY[:,1,0],'-or',label="inp_m1") 
plt.plot(t,YY[:,1,1],'-*b',label="inp_m2") 
plt.plot(t,YY[:,1,2],'-*g',label="inp_m3") 
plt.ylim(Ymin,Ymax)
plt.ylabel("Responce y(k)")
plt.xlabel("Step (k)")
plt.grid()
plt.legend() #labelの表示

plt.subplot(133,title="Displacement of m3") #　1行目
plt.plot(t,YY[:,2,0],'-or',label="inp_m1") 
plt.plot(t,YY[:,2,1],'-*b',label="inp_m2") 
plt.plot(t,YY[:,2,2],'-*g',label="inp_m3") 
plt.ylim(Ymin,Ymax)
plt.ylabel("Responce y(k)")
plt.xlabel("Step (k)")
plt.grid()
plt.legend() #labelの表示


# 表示
plt.show() 
