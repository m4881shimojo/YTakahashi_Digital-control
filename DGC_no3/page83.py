#digital control 
#デジタル制御　高橋安人
#20231112 shimojo
#plot用コマンドを変更
#
#p84 Fig5-2
#単一ループデジタル制御　一次プラント制御
#Eq.5-1 R(z):input Y(z):output V(z)=0
#G[z]=K(z-d)/{(z-1)(z-p)}  Eq.5-10.
# K=Kc*Kp
# Gp(z)=Kp/(z-p) プラント　＜---G(s)=1/(s+a)
# Gc(z)=Kc　ディジタル制御法則　<--P　比例制御
# これをtypeI形とする　--->Gc(z)が極(Z-1)を持つこと！
#　Gc(z)=Kc(z-d)/(z-1)
#  (z-d)? -->設計の自由度を持つための定数。これは聞かないとわからない
# これでEq.5-10　となる
#Y[k]=G[z]/(1+G[z])*R[k]
#
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
# Figureの初期化
fig = plt.figure(figsize=(6, 4.5)) # Figureの初期化
# Create a figure of size 8x10 inches, 100 dots per inch
#fig = plt.figure(figsize=(6, 4)) 
#
n=2 #2次系
knum=10 #サンプル数
#
r=np.zeros(knum) # input (step or ramp) 入力
y=np.zeros(knum) 
E=np.zeros(knum) 
U=np.zeros(knum) 
err=np.zeros(knum)
#
##input A,B
T=0.1
p=np.exp(-T) #p=0.905 
#d=0.6;kgain=1.6 #case1
#d=0.475;kgain=1.905 #case2
#d=0.3;kgain=2.5 #case3
d=0.3;kgain=3 #case4 発振 add shimojo
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
e1=T*(1-p)/(kgain*(1-d))
e1Strg="{:.3g}".format(e1)
print("e1="+e1Strg)
#print("e1={:.3g}".format(e1))
#
for i in range(0,knum):
    r[i]=1.0 #step input
    r[i]=i*T #rump input0,T,2T,3T....
    #r[i]=i #rump input

#　y(k)を生成する。伝達関数の場合、(3-32)の方が分かり易い
#　list 3-3 に代わってベタなアルゴリズムで記述する
# Y[k]=-(a1*Y[k-1]+a2*Y[k-2])+(b1*r[k-1]+b2*r[k-2]])
#
y[0]=0.0
y[1]=-a1*y[0]+(b1*r[0])  
#Y[2]=-(a1*Y[1]+a2*Y[0])+(b1*r[1]+b2*r[0])
for k in range(2,knum):   
    y[k]=-(a1*y[k-1]+a2*y[k-2])+(b1*r[k-1]+b2*r[k-2]) 
       #print(k,Y[k])

#誤差の計算
#　r[k]-y[k]　------> E[k]と等しくなるはず
#
err[0]=0
for k in range(0,knum):   
    err[k]=r[k]-y[k]
    #
#
# 念のため、E(z)を計算してみた
#E(z)=1/(1+G(z))
#E(k)=-(a1*E[k-1]+a2*E[k-2])+(r[k]+b1*r[k-1]+b2*r[k-2])
E[0]=0.0
E[1]=-a1*E[0]+(1+b1e*r[0])  
for k in range(2,knum):   
    E[k]=-(a1*E[k-1]+a2*E[k-2])+(r[k]+b1e*r[k-1]+b2e*r[k-2])  
    #
#
#念のため、U(k)を算出してみた    
# kp=(1-p)/1.0 #(5-8)
# Kgain=Kp*Kc
# kc=kgain/kp
#kc=kgain/kp
#print("kgain",kgain,kp,kc)
#U[0]=0.0
#for k in range(1,knum):   
#    U[k]=-(U[k-1])+kc*(E[k]-d*E[k-1])  

#########################################################
# PLOT
#########################################################

ax1= plt.axes()
t=np.arange(0,knum)

ax1.plot(t,r,'-og')  #input 
ax1.plot(t,y,'-or')  # y(k)
#ax1.plot(t,E,'-ok') #E(k)
#ax1.plot(t,U,'-ob') #U(k)
ax1.plot(t,err*5,'-ok') # 5倍に拡大表示した！
#
#Title_para="K="+str(float(kgain))+", P="+str(float(p))+", d="+str(float(d))
strgp="{:.3g}".format(p)
Title_para="K="+str(float(kgain))+", P="+strgp+", d="+str(float(d))
plt.title("図5-2 1次形の制御 :"+Title_para, fontname="MS Gothic")
#
Ymax=1.0; Ymin=0.0
#plt.ylim(Ymin,Ymax,2.)
plt.ylabel("Responce ")
plt.xlabel("step (k)")
ax1.set_xticks(np.linspace(0, knum, 11))
ax1.set_yticks(np.linspace(Ymin, Ymax,11))
#
# x軸に補助目盛線を設定
ax1.grid(which = "major", axis = "x", color = "blue", alpha = 0.8,
        linestyle = "--", linewidth = 1)
# y軸に目盛線を設定
ax1.grid(which = "major", axis = "y", color = "green", alpha = 0.8,
        linestyle = "--", linewidth = 1)
# 補助目盛を表示
plt.minorticks_on()
plt.grid(which="minor", color="gray", linestyle=":")
#ax1.grid()

xp=knum*3/5; yp=Ymax*2/5  #plt.textの位置座標
plt.text(xp,yp, "e1="+e1Strg ) #定常偏差e1
#plt.text(xp,yp, "K={:.3g}, p={:.3g}, d={:.3g}".format(kgain,p,d) )
#plt.text(xp,yp-Ymax/15, "b1,b2,b3={:.3g},{:.3g},{:.3g}".format(b1,b2,b3))

#########################################################
#　F(z)の極を求める
#########################################################

#PLOT Circle radius=1
plt.figure(figsize=(6,4.5)) #新しいウィンドウを描画
ax2= plt.axes()
num=100
circle_x=np.zeros(num) 
circle_y=np.zeros(num) 

for k in range (0,num):
     rad=2*np.pi/num*k
     circle_x[k]=np.sin(rad);circle_y[k]=np.cos(rad)
          
ax2.plot(circle_x,circle_y,'--k')

#Plot　極
a1=kgain-1-p; a2=p-d*kgain
A=np.array([a1, a2])
A_pol=np.array([1,a1, a2])
pol=np.poly1d(A_pol) #n多項式関数の決定
z=pol.roots  #n多項式の根を求める関数 
#markers1 = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8"]
#markers2 = ["s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_", "$x$"]
ax2.scatter(z.real,z.imag, color='r', s=200, marker= "$x$")

#Plot　極の軌跡
kpara_num=40
for kpara in range(0,kpara_num):
    kgain=kpara*.1
    a1=kgain-1-p; a2=p-d*kgain
    A=np.array([a1, a2])
    A_pol=np.array([1,a1, a2])
    
    pol=np.poly1d(A_pol) #n多項式関数の決定
    z=pol.roots  #n多項式の根を求める関数 

    pmod=kpara%5
    if pmod==0: ax2.plot(z.real,z.imag,'.k')
    if pmod==1: ax2.plot(z.real,z.imag,'.r')
    if pmod==2: ax2.plot(z.real,z.imag,'.g')
    if pmod==3: ax2.plot(z.real,z.imag,'.b')
    if pmod==4: ax2.plot(z.real,z.imag,'.y')

plt.title("図5-2 根軌跡", fontname="MS Gothic")
Ymin=-1.5;Ymax=1.5
Xmin=-2;Xmax=2
plt.ylim(Ymin,Ymax)
plt.xlim(Xmin,Xmax)
ax2.set_xticks(np.linspace(Xmin, Xmax, 9))
ax2.set_yticks(np.linspace(Ymin, Ymax, 9))
#
ax2.grid()
#plt.tight_layout()

# 表示
plt.show()    