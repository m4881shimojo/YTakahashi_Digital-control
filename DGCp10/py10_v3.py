#digital control 
#デジタル制御　高橋安人
#list 1.1 page 10. shimojo 20160728/20231013
#
import numpy as np
import matplotlib.pyplot as plt
#Dead beat response of robo arm
#print("start")

G=np.zeros(11); U=np.zeros(11); Y=np.zeros(11); #0,1,....8,9,10　の11個
UG=np.zeros([11,11]); YY=np.zeros([11,11]);

#fig = plt.figure() # 無くて良い
# input parameter
P=-0.5; R=1.0; G[1]=1.
#G,U,Y
#単位パルス入力に対する応答波形ｇ
for i in range(2,11):
    G[i]=P*G[i-1]
    #print(i,G[i])
    #plt.plot(i,G[i], "ro")
 
#G,Y   
#応答出力がR=1となるように、入力Uを定める   
for k in range(1,11):
    U[k-1]=(R-Y[k])/G[1]
    #print(k,U[k-1]) #add shimojo for check
    #plt.plot(k,U[k-1], "bo") #add shimojo for check

    for j in range(k,11,1):  
        Y[j]+=U[k-1]*G[j-k+1]
        UG[k-1,j]=U[k-1]*G[j-k+1] #add shimojo for data_plot
        #print(j,UG[k-1,j],Y[j]) #add shimojo for check
        #plt.plot(j,UG[k-1,j],"ro") #Add shimojo for check
        #plt.plot(j,Y[j],"go") #Add shimojo  for check   
    #print(k,U[k-1],Y[k]) #これは書籍Original

    
#次からはプロットするためshimojoが追加
#応答波形は単位応答波形のsum
#YY[1]=UG[0]
#YY[2]=UG[0]+UG[1]=YY[1]+UG[1]
#YY[3]=UG[0]+UG[1]+UG[2]=YY[2]+UG[2]
#YY[4]=UG[0]+UG[1]+UG[2]+UG[3]=YY[3]+UG[3]

for i in range(1,10):
    YY[i]=YY[i-1]+UG[i-1]


x = np.arange(0,11)

#描画。サイズは横４インチ、縦8インチ、配置の自動調整
#参考： https://www.yutaka-note.com/entry/2020/01/02/232925

fig = plt.figure(figsize = (4,8),tight_layout=True) 

#長くなるので、図は6個までとした。
plt.subplot(611,title="u(0)") # 6行1列の1番目
plt.plot(x,UG[0,], '*--r')  # [marker][line][color]の順番
plt.plot(x,YY[0], '*--g')

plt.subplot(612,title="u(0)+u(1)")
plt.plot(x,UG[1,],'*--r')
plt.plot(x,YY[1], '*--g')

plt.subplot(613,title="u(0)+..+u(2)")
plt.plot(x,UG[2,],'*--r')
plt.plot(x,YY[2], '*--g')

plt.subplot(614,title="u(0)+..+u(3)")
plt.plot(x,UG[3,],'*--r')
plt.plot(x,YY[3], '*--g')

plt.subplot(615,title="u(0)+..+u(4)")
plt.plot(x,UG[4,],'*--r')
plt.plot(x,YY[4], '*--g')

plt.subplot(616,title="u(0)+..+u(5)")
plt.plot(x,UG[5,],'*--r')
plt.plot(x,YY[5], '*--g')


# plt.show()で画面に表示（Jupyter Notebookの場合は不要）
plt.show()
