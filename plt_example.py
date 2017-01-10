#coding=utf-8
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
# 饼状图的画法
labels='frogs','hogs','dogs','logs'

sizes=15,20,45,10

colors='yellowgreen','gold','lightskyblue','lightcoral'

explode=0,0.2,0,0

plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)

plt.axis('equal')

plt.show()

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 基本的作图指令 用plot
x = np.arange(-5.0, 5.0, 0.02)
y1 = np.sin(x)
plt.figure(3)
plt.subplot(211)
plt.plot(x, y1)
plt.subplot(212)
# 设置x轴范围
xlim(-2.5, 2.5)
# 设置y轴范围
ylim(-1, 1)
plt.plot(x, y1)
plt.show()

#多图叠加的做法
# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)
# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t ** 2, 'bs', t, t ** 3, 'g^') # 每三个一组单元
plt.show()

# figure的用法

plt.figure(1)  # 第一张图
plt.subplot(211)  # 第一张图中的第一张子图
plt.plot([1, 2, 3])
plt.subplot(212)  # 第一张图中的第二张子图
plt.plot([4, 5, 6])
plt.figure(2)  # 第二张图
plt.plot([4, 5, 6])  # 默认创建子图subplot(111)
plt.figure(1)  # 切换到figure 1 ; 子图subplot(212)仍旧是当前图
plt.subplot(211)  # 令子图subplot(211)成为figure1的当前图
plt.title('Easy as 1,2,3')  # 添加subplot 211 的标题
plt.show()


# 做直方图
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)
# 数据的直方图
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)
print n
plt.xlabel('Smarts')
plt.ylabel('Probability')
# 添加标题
plt.title('Histogram of IQ')
# 添加文字
plt.text(60, .025, r'$mu=100, sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

#  文本注释
ax = plt.subplot(111)
t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2 * np.pi * t)
line, = plt.plot(t, s, lw=2)
plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05), )
plt.ylim(-2, 2)
plt.show()



# 导入 matplotlib 的所有内容（nympy 可以用 np 这个名字来使用）
# 创建一个 8 * 6 点（point）的图，并设置分辨率为 80
figure(figsize=(8, 6), dpi=80)
# 创建一个新的 1 * 1 的子图，接下来的图样绘制在其中的第 1 块（也是唯一的一块）
subplot(1, 1, 1)
X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)
# 绘制余弦曲线，使用蓝色的、连续的、宽度为 1 （像素）的线条
plot(X, C, color="blue", linewidth=1.0, linestyle="-")
# 绘制正弦曲线，使用绿色的、连续的、宽度为 1 （像素）的线条
plot(X, S, color="r", lw=4.0, linestyle="-")
plt.axis([-4, 4, -1.2, 1.2])
# 设置轴记号
xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
       [r'$-pi$', r'$-pi/2$', r'$0$', r'$+pi/2$', r'$+pi$'])  # 这里的相对应标注
yticks([-1, 0, +1],
       [r'$-1$', r'$0$', r'$+1$'])
# 在屏幕上显示
show()


ax = gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

# 添加图例 legend()
plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="cosine")
plot(X, S, color="red", linewidth=2.5, linestyle="-", label="sine")
legend(loc='upper left')
# 散点图
'''plt.scatter(Xb,yb)
plt.ylabel('value of house /1000 ($)')
plt.xlabel('number of rooms') '''

# 给特殊点加标注
# 这是特别有用的一个技能
t = 2 * np.pi / 3
# 作一条垂直于x轴的线段，由数学知识可知，横坐标一致的两个点就在垂直于坐标轴的直线上了。这两个点是起始点。
plot([t, t], [0, np.cos(t)], color='blue', linewidth=2.5, linestyle="--")
scatter([t, ], [np.cos(t), ], 50, color='blue')
annotate(r'$sin(frac{2pi}{3})=frac{sqrt{3}}{2}$',
         xy=(t, np.sin(t)), xycoords='data',
         xytext=(+10, +30), textcoords='offset points', fontsize=16,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plot([t, t], [0, np.sin(t)], color='red', linewidth=2.5, linestyle="--")
scatter([t, ], [np.sin(t), ], 50, color='red')
annotate(r'$cos(frac{2pi}{3})=-frac{1}{2}$',
         xy=(t, np.cos(t)), xycoords='data',
         xytext=(-90, -50), textcoords='offset points', fontsize=16,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.show()

# 子图展示，plt.subplot(***)


# 在同一张图片中（figure），做多个不同坐标系的图 plt.axes()
# create some data to use for the plot
dt = 0.001
t = np.arange(0.0, 10.0, dt)
r = np.exp(-t[:1000] / 0.05)  # impulse response
x = np.random.randn(len(t))
s = np.convolve(x, r)[:len(x)] * dt  # colored noise
# the main axes is subplot(111) by default
plt.plot(t, s)
plt.axis([0, 1, 1.1 * np.amin(s), 2 * np.amax(s)])
plt.xlabel('time (s)')
plt.ylabel('current (nA)')
plt.title('Gaussian colored noise')
# this is an inset axes over the main axes
a = plt.axes([.65, .6, .2, .2], axisbg='y')
n, bins, patches = plt.hist(s, 400, normed=1)
plt.title('Probability')
plt.xticks([])
plt.yticks([])
# this is another inset axes over the main axes
a = plt.axes([0.2, 0.6, .2, .2], axisbg='y')
plt.plot(t[:len(r)], r)
plt.title('Impulse response')
plt.xlim(0, 0.2)
plt.xticks([])
plt.yticks([])
plt.show()
