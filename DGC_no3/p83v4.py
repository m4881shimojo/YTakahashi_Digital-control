#digital control 
#デジタル制御　高橋安人
#20240208
#p84 Fig5-2
#単一ループデジタル制御　一次プラント制御
#Eq.5-1 R(z):input Y(z):output V(z)=0
#G[z]=K(z-d)/{(z-1)(z-p)}  Eq.5-10.
# K=Kc*Kp
# Gp(z)=Kp/(z-p) プラント　＜---G(s)=1/(s+a)
# Gc(z)=Kc　ディジタル制御法則　<--P　比例制御
# これをtypeI形とする　--->Gc(z)が極(Z-1)を持つこと！
#　Gc(z)=Kc(z-d)/(z-1)
#  d-->設計の自由度を持つための定数。
# これでEq.5-10　となる
#Y[k]=G[z]/(1+G[z])*R[k]
#
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
#
##################### def root_log ###################
def root_log(wbeg,wend,dw,plot_color):
    tw=np.arange(wbeg,wend,dw)
    knum=len(tw)
    wt=np.zeros(knum)
    for i in range(knum):
        if tw[i]>0:wt[i]=10**tw[i]
        else: wt[i]=(1/10)**(-tw[i])
    #print(wt)
    labe="{:.3g}".format(wt[0])
    
    for j in range(0,knum):
        kgain=wt[j]
        a1=kgain-1-p; a2=p-d*kgain
        A=np.array([a1, a2])
        A_pol=np.array([1,a1, a2])
        
        pol=np.poly1d(A_pol) #n多項式関数の決定
        z=pol.roots  #n多項式の根を求める関数 
        if j==0:plt.plot(z.real,z.imag, plot_color,label=labe)
        else: plt.plot(z.real,z.imag, plot_color)
    
        #end root_plot
##################### def end ###################

##################### main ######################
n=2 #2次系
knum=10 #サンプル数#
r1=np.zeros(knum) # input (step) 入力
r2=np.zeros(knum) # input (ramp) 入力
y1=np.zeros(knum) #respose for r1
y2=np.zeros(knum) #respose for r2
err=np.zeros(knum)
#
##input A,B
Tsample=0.1
p=np.exp(-Tsample) #p=0.905 ,a=1
#*****************************
#   input parameter          *
#*****************************
d=0.6;kgain=1.6 #case1
#d=0.475;kgain=1.905 #case2
#d=0.3;kgain=2.5 #case3
#d=0.3;kgain=3 #case4 発振 add shimojo
#
kp=(1-p)/1.0 #(5-8)
# Kgain=Kp*Kc
# kc=kgain/kp
# 伝達関数形による応答計算 list 3-3 p47
a1=kgain-1-p; a2=p-d*kgain
b1=kgain; b2=-d*kgain
# use for E(k) calsulate
b1e=-(1+p);b2e=p # for error E(z)
#
A=np.array([a1, a2]) #a1,a2
B=np.array([b1,b2]) #b1,b2
#
#定常偏差
#eo=1/(1+G(1))
e0=1/(1+(kgain/(1-p)))
print("e0={:.3g}".format(e0))

e1=Tsample*(1-p)/(kgain*(1-d))
#e1Strg="{:.3g}".format(e1)
print("e1={:.3g}".format(e1))
#print("e1={:.3g}".format(e1))
#
# input 
for i in range(0,knum):
    r1[i]=1.0 #step input
    r2[i]=i*Tsample #rump input0,T,2T,3T....
    #
#　y(k)を生成する。伝達関数の場合、(3-32)の方が分かり易い
#　list 3-3 に代わってベタなアルゴリズムで記述する
# a1=kgain-1-p; a2=p-d*kgain; b1=kgain; b2=-d*kgain
# Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*r[k-1]+b2*r[k-2]])
#
for k in range(0,knum):
    if k==0: y1[k]=0.0; y2[k]=0.0
    elif k==1: y1[k]=-a1*y1[k-1]+(b1*r1[k-1]);y2[k]=-a1*y2[k-1]+(b1*r2[k-1])   
    else: y1[k]=-(a1*y1[k-1]+a2*y1[k-2])+(b1*r1[k-1]+b2*r1[k-2]);y2[k]=-(a1*y2[k-1]+a2*y2[k-2])+(b1*r2[k-1]+b2*r2[k-2])  
   
#誤差の計算
for k in range(0,knum):   
    err[k]=r1[k]-y1[k]
#

#########################################################
# PLOT
#########################################################
# Figure1
#########################################################
fig = plt.figure(figsize=(7.5, 4),tight_layout=True) # Figureの初期化(横ｘ縦)
# Create a figure of size 横x縦 inches, 100 dots per inch
plt.suptitle("図5-2 1次系の制御", fontname="MS Gothic")
t=np.arange(0,knum)
#########################################################
#                 subplot 1   step input                #
#########################################################
plt.subplot(121) 
plt.plot(t,r1,'-.g',label="input")  #input 
plt.plot(t,y1,'-or',label="y(k)")  # y(k)
#ax1.plot(t,err*5,'-*k',label="errorx5") # 5倍に拡大表示した！
#
Ymax=6.0; Ymin=-4.0
plt.ylim(Ymin,Ymax)
plt.ylabel("Response ")
plt.xlabel("step (k)")
#
Title_para="K="+str(kgain)+", P="+"{:.3g}".format(p)+", d="+str(d)
plt.title("図5-2 step応答 :"+Title_para, fontname="MS Gothic")
plt.legend(loc='upper right')
#plt.legend(loc='upper left')
plt.grid()
#########################################################
#                 subplot 2    ramp input               #
#########################################################
plt.subplot(122) #　
plt.plot(t,r2,'-.g',label="input")  #input 
plt.plot(t,y2,'-or',label="y(k)")  # y(k)
#ax1.plot(t,err*5,'-*k',label="errorx5") # 5倍に拡大表示した！
#
Ymax=1.0; Ymin=0.0
#plt.ylim(Ymin,Ymax,2.)
plt.ylabel("Response ")
plt.xlabel("step (k)")
#
Title_para="K="+str(kgain)+", P="+"{:.3g}".format(p)+", d="+str(d)
plt.title("図5-2 ramp応答:"+Title_para, fontname="MS Gothic")
plt.legend(loc='upper left')
plt.grid()

xp=knum*3/5; yp=Ymax*1/5  #plt.textの位置座標
#plt.text(xp,yp,"e1={:.3g}".format(e1) ) #定常偏差e1

#########################################################
#　 Figure 2   F(z)の極を求める
#########################################################
#PLOT Circle radius=1
plt.figure(figsize=(6,4.5)) #新しいウィンドウを描画
#ax2= plt.axes()
num=100
circle_x=np.zeros(num) 
circle_y=np.zeros(num) 

for k in range (0,num):
     rad=2*np.pi/num*k
     circle_x[k]=np.sin(rad);circle_y[k]=np.cos(rad)
          
plt.plot(circle_x,circle_y,'--k')#,label="unit circle")

#Plot　極
#F(z)=z(^2)+(K-1-d)z+(p-Kd) closed-loop
#F(z)=z(^2)+a2*z+a2
a1=kgain-1-p; a2=p-d*kgain
A=np.array([a1, a2])
A_pol=np.array([1,a1, a2])
pol=np.poly1d(A_pol) #n多項式関数の決定
print(kgain,p,d)
print(A);print(pol)
z=pol.roots  #n多項式の根を求める関数 
#print("Z=",z)
ppX=np.array([p,1]);ppY=np.array([0,0])
plt.scatter(ppX,ppY, color='r', s=100, marker= "$x$",label="pole")
plt.scatter(z.real,z.imag, color='m', s=100, marker= "*",label="F-pole")
plt.scatter(d,0, color='b', s=100, marker= "$+$",label="zero")

#
#Plot　極の軌跡
# 10(^wbeg)<wt[i]10(^wend)
#dw　分割単位
wbeg=-3;wend=-1;dw=0.1;plot_color="c." 
root_log(wbeg,wend,dw,plot_color)

wbeg=-1;wend=0;dw=0.05;plot_color="r." 
root_log(wbeg,wend,dw,plot_color)

wbeg=0;wend=0.5;dw=0.01;plot_color="b." 
root_log(wbeg,wend,dw,plot_color)

wbeg=0.5;wend=1;dw=0.01;plot_color="g." 
root_log(wbeg,wend,dw,plot_color)

wbeg=1;wend=3;dw=0.01;plot_color="m." 
root_log(wbeg,wend,dw,plot_color)


plt.title("図5-2 根軌跡", fontname="MS Gothic")
Ymin=-1.5;Ymax=1.5
Xmin=-2;Xmax=2
plt.ylim(Ymin,Ymax)
plt.xlim(Xmin,Xmax)
#
plt.legend(loc='upper right')
plt.grid()
#plt.tight_layout()
# 表示
plt.show()    