#digital control 
#デジタル制御　高橋安人
#list 1-1 page 10. shimojo 20231013
#図3-1をlist1-1に基づき描く
import numpy as np
import matplotlib.pyplot as plt
#Dead beat response of robo arm
#print("start")

G=np.zeros(11); U=np.zeros(11); Y=np.zeros(11); #0,1,....8,9,10　の11個
UG=np.zeros([11,11]); YY=np.zeros([11,11]);Uin=np.zeros([11,11])
Num=11
#
# input parameter
G[0]=0.;Y[0]=0.;Y[1]=0. #今回不要だが書籍通りにする
P=-0.5; R=1.0; G[1]=1. #(1-4)

#G,U,Y
#単位パルス入力に対する応答波形ｇ
for i in range(2,Num):
    G[i]=P*G[i-1]
    Y[i]=Y[i-1] #今回不要だが書籍通りにする

#G,Y   
#応答出力y(k)=R,R=1となるように、入力Uを定める   
for k in range(1,Num):
    U[k-1]=(R-Y[k])/G[1]
#
    for j in range(k,Num):  
        Y[j]=Y[j]+U[k-1]*G[j-k+1]        
        UG[k-1,j]=U[k-1]*G[j-k+1] #add shimojo for data_plot
        #print('{:2g},{:2g},{:3g},{:3g},{:3g}'.format(k,j,Y[j],U[k-1],G[j-k+1]) )
    YY[k-1,:]=Y #add shimojo for data_plot
    Uin[k-1,k-1]=U[k-1] #add shimojo for data_plot

    #if k==3 :
    #    plt.plot(x,Y, '*--r')
    #    plt.plot(x,U, drawstyle='steps-post',color='g', linestyle='dashed', marker='o')
    #    plt.plot(x,UG[k-1,:], '*--b')
    
########################################################################
#            次からはプロットするためshimojoが追加                       #
########################################################################
#応答波形は単位応答波形のsum

flag=True #Trueで描く(意味はない。debug用に入れた)
if (flag):

    x = np.arange(0,Num) # x軸
 
     #描画。サイズは横４インチ、縦8インチ、配置の自動調整
    #参考： https://www.yutaka-note.com/entry/2020/01/02/232925
    #参考：https://pystyle.info/matplotlib-line-properties/　ステップplot
    fig = plt.figure(figsize = (12,8),tight_layout=True) 

    #plt.title("図1-3ロボットアームの有限時間整定応答", fontname="MS Gothic")
    Ymin=-1.0;Ymax=2.0
    P_ratio=str(P)
    #長くなるので、図は9個までとした。
    plt.subplot(331,title="u(0)") #　1行目
    plt.plot(x,Uin[0,:],drawstyle='steps-post',color='g', linestyle='dashed', marker='o')  # [marker][line][color]の順番
    plt.ylim(Ymin,Ymax)

    plt.subplot(332,title="ug(k), P="+str(P)) # 6行1列の1番目
    plt.plot(x,UG[0,:], '*--b')
    plt.ylim(Ymin,Ymax)

    plt.subplot(333,title="y(k)") # 
    plt.plot(x,YY[0,:], '*--r')
    plt.ylim(Ymin,Ymax)


    plt.subplot(334,title="u(1)") # 　2行目
    plt.plot(x,Uin[1,:],drawstyle='steps-post',color='g', linestyle='dashed', marker='o')  # [marker][line][color]の順番
    plt.ylim(Ymin,Ymax)

    plt.subplot(335,title="ug(k), P="+str(P)) # 
    plt.plot(x,UG[1,:], '*--b')
    plt.ylim(Ymin,Ymax)

    plt.subplot(336,title="y(k)") # 
    plt.plot(x,YY[1,:], '*--r')
    plt.ylim(Ymin,Ymax)


    plt.subplot(337,title="u(2)") # 　3行目
    plt.plot(x,Uin[2,:],drawstyle='steps-post',color='g', linestyle='dashed', marker='o')  # [marker][line][color]の順番
    plt.ylim(Ymin,Ymax)

    plt.subplot(338,title="ug(k), P="+str(P)) # 
    plt.plot(x,UG[2,:], '*--b')
    plt.ylim(Ymin,Ymax)

    plt.subplot(339,title="y(k)") # 
    plt.plot(x,YY[2,:], '*--r')
    plt.ylim(Ymin,Ymax)

else:
    print("end")

# Num_stepでの応答波形
Num_step=8 # K=8までの応答（書籍図１－３と同じにした）
U=np.zeros(11) # UをUinとしてplotさせるため。こうしなくても良いが、変更が面倒で。

for l in range(Num_step):
    U[l]=Uin[l,l]
#
fig = plt.figure(figsize = (12,3),tight_layout=True) 
plt.subplot(121,title="u(k), P="+str(P)) # 6行1列の1番目
plt.plot(x,U,drawstyle='steps-post',color='g', linestyle='dashed', marker='o')
plt.ylim(Ymin,Ymax)

plt.subplot(122,title="Y(k), Number of step="+str(Num_step)) #
plt.plot(x,YY[Num_step-1,:], '*--r') #Y(k)は1step進んだ結果になるから
plt.ylim(Ymin,Ymax)

# plt.show()で画面に表示（Jupyter Notebookの場合は不要）
plt.show()
