#digital control 
#デジタル制御　高橋安人
#fig1.3 shimojo 20231025
#
import numpy as np
import matplotlib.pyplot as plt
#
knum=20
#Num=10
G=np.zeros(knum) #g(k)=p*^(k-1)
U=np.zeros(knum) # 入力
Y=np.zeros(knum) # 出力
YY=np.zeros((4,knum))

# input parameter
P_para=np.array([-0.5,-0.7,0.5,0.7])
#
t=np.arange(0,knum) # 0,1,....knum-1

#単位パルス入力に対する応答波形ｇ
for m in range(4):
    P=P_para[m] #-->p: g(k)=p*^(k-1)
    G[0]=1.0;G[1]=1/P
    for i in range(1,knum):        
        G[i]=P**(i-1) #g(k)=p*^(k-1)
        
        #
    #応答
    U=np.zeros(knum)
    U[1]=1
    for k in range(0,knum):
        y0=0.0
        #Num=k
        for j in range(0,k):
            y0=y0+U[j]*G[k-j] # Eq.(1-5)
        #Y[k]=y0
        YY[m,k]=y0#;print(m,k,y0)
#end m loop

#########################################################
#　　　　　　　　　　　　　　　PLOT 　　　　　　　　　　　 #
#########################################################
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(8, 6)) # Figureの初期化
#
gs = GridSpec(3, 1)  # 縦方向に3つ、横方向に１つの場所を用意
#ss1--> 場所は(0,0)、縦2つ、横１つ、を使用
ss1 = gs.new_subplotspec((0, 0), rowspan=2,colspan=1)  # ax1 を配置する領域
#ss2--> 場所は(2,0)、縦１つ横１つ、を使用
ss2 = gs.new_subplotspec((2, 0), rowspan=1, colspan=1)  # ax2 を配置する領域
#
# ax1　PLOT
ax1 = plt.subplot(ss1)
ax1.plot(t,YY[0,],'--*c',label=str(P_para[0]))
ax1.plot(t,YY[1,],'--*r',label=str(P_para[1]))
ax1.plot(t,YY[2,],'--*b',label=str(P_para[2]))
ax1.plot(t,YY[3,],'--*k',label=str(P_para[3]))
#
ax1.grid()
#plt.title("g(k), P="+str(P_para))
plt.title("インパルス入力に対する応答(g(k)=p^(k-1))", fontname="MS Gothic")
plt.legend() #labelの表示
plt.ylabel("Responce")
plt.xlabel("step k")
# ax2　PLOT
ax2 = plt.subplot(ss2)
#plt.plot(t,U, drawstyle='steps-post',color='g', linestyle='solid', marker='',label="input")
plt.plot(t,U, '-*g',drawstyle='steps-post',label="input")


ax2.grid()
plt.legend() #labelの表示
plt.ylabel("input")
plt.xlabel("step k")

#plt.ylim(-2.,2.)
#plt.xlim(0,11)
   # 表示
plt.show()    
