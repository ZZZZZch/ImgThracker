

# 概述

SURF 是一种兴趣点检测及描述子算法。可以说是对SIFT算法的加强版。是用来实现尺度不变性特征点的检测与匹配任务的，SURF算法首先利用Hessian矩阵确定候选点，然后进行NMS（非极大值抑制）初步确定特征点，再通过确定特征点主方向，综合构造SURF特征点描述算子从而进行图像特征抽取。



# SURF方法的具体实现



## Hessian矩阵简介

在数学中，**海森矩阵**（Hessian matrix 或 Hessian）是一个多变量实值函数的二阶偏导数组成的方块矩阵，描述了函数的局部曲率。假设有一实数函数![](https://wikimedia.org/api/rest_v1/media/math/render/svg/9ae30a831179af233ad961a841950332cad6146a)如果![f](https://wikimedia.org/api/rest_v1/media/math/render/svg/132e57acb643253e7810ee9702d9581f159a1c61)所有的二阶偏导数都存在，那么 ![f](https://wikimedia.org/api/rest_v1/media/math/render/svg/132e57acb643253e7810ee9702d9581f159a1c61) 的海森矩阵的第 ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/53fcc7b57da64979c370eb150eb5a61a625a08e8)-项即：

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/376f63ba327a32b89317226051808ea857992cad)

其中![](https://wikimedia.org/api/rest_v1/media/math/render/svg/7d750ee2fffa2be890b9626b672037f0104cfe7a)，即

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/389411defb8c9662a366a1d87c25c197c1c56dc4)



给定二阶导数连续的映射![](https://wikimedia.org/api/rest_v1/media/math/render/svg/a1c262f451d196f94214e2b8856b462c2d209306)，海森矩阵的行列式，可用于分辨![f](https://wikimedia.org/api/rest_v1/media/math/render/svg/132e57acb643253e7810ee9702d9581f159a1c61)的临界点是属于鞍点还是极值点。

对于![f](https://wikimedia.org/api/rest_v1/media/math/render/svg/132e57acb643253e7810ee9702d9581f159a1c61)的临界点![(x_{0},y_{0})](https://wikimedia.org/api/rest_v1/media/math/render/svg/29c296094af9a1c665425debeac5eaab99a37a04)一点，有![{\frac  {\partial f(x_{0},y_{0})}{\partial x}}={\frac  {\partial f(x_{0},y_{0})}{\partial y}}=0](https://wikimedia.org/api/rest_v1/media/math/render/svg/cff30491b15467e68444e86538651370bd571e0d)，然而凭一阶导数不能判断它是鞍点、局部极大点还是局部极小点。海森矩阵可能解答这个问题。

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/31f604c36c3f1e37efe8287d7848c797b09484fd)



- H > 0：若![{\frac  {\partial ^{2}f}{\partial x^{2}}}>0](https://wikimedia.org/api/rest_v1/media/math/render/svg/df873e9e76d5a3c14f76660e3bc9faa2d14e0aee)，则![(x_{0},y_{0})](https://wikimedia.org/api/rest_v1/media/math/render/svg/29c296094af9a1c665425debeac5eaab99a37a04)是局部极小点；若![{\frac  {\partial ^{2}f}{\partial x^{2}}}<0](https://wikimedia.org/api/rest_v1/media/math/render/svg/7b29b1c5e99c1d6836ca2246a77414ec32a973e8)，则![(x_{0},y_{0})](https://wikimedia.org/api/rest_v1/media/math/render/svg/29c296094af9a1c665425debeac5eaab99a37a04)是局部极大点。
- H < 0：![(x_{0},y_{0})](https://wikimedia.org/api/rest_v1/media/math/render/svg/29c296094af9a1c665425debeac5eaab99a37a04)是鞍点。
- H = 0：二阶导数无法判断该临界点的性质，得从更高阶的导数以泰勒公式考虑。



## SURF中Hessian矩阵的构建

我们假设一个函数f(x, y)，那么图像中像素点的Hessian矩阵我们表示为：

![](http://img.blog.csdn.net/20131127162720250)

从而每一个像素点都可以求出一个Hessian矩阵。Hessian矩阵判别式为：

![](http://img.blog.csdn.net/20131127162900312)

在SURF算法中，通常用图像像素I(x,y)取代函数值f(x,y)。然后选用二阶标准高斯函数作为滤波器。

使用高斯滤波的原因是我们要求我们的特征点具有尺度无关性，所以我们在进行Hessian矩阵构造之前做一次高斯滤波，用高斯核和函数图像在x处卷积来实现，这样我们就有：

![](http://img.blog.csdn.net/20131127184804390)

其中 G(t)为高斯核，g(t)为高斯函数，t 为高斯方差：

![](http://img.blog.csdn.net/20131127184739281)

这样通过特定核间的卷积计算二阶偏导数，又能计算出H矩阵的三个矩阵元素Lxx, Lxy, Lyy，从而计算出H矩阵公式如下：

![](http://img.blog.csdn.net/20131127184650203)

通过这种方法可以为图像中每个像素计算出其H矩阵的决定值，并用这个值来判别特征点。

这里SURF提出用近似的值代替L(x,t)，即可以将高斯二阶梯度模板用盒函数来近似：

![](http://img.blog.csdn.net/20151028181456547?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)



比如说y方向上的模板Fig1(a)与Fig1(c)。Fig1(a)即用高斯平滑后在y方向上求二阶导数的模板。为了加快运算用了近似处理，其处理结果如Fig1(b)所 示，这样就简化了很多。关键是可以使用积分图来进行运算！



## 积分图

SURF是对积分图像进行操作，从而实现了加速，采用盒子滤波器计算每个像素点的Hessian矩阵行列式时，只需要几次加减法运算，而且运算量与盒子滤波器大小无关，所以能够快速的构成出SURF的尺度金字塔。

积分图像中每个像元的值，是原图像上对应位置的左上角所有元素之和

![img](http://img.blog.csdn.net/20160825155743997)

如上图：我们要得到绿色矩形区域内的矩形像素之和S，只需要在积分图上，运算S=D-B-C+A就可以得到。



## 尺度空间

图像的尺度空间是这幅图像在不同解析度下的表示。上面讲的这么多只是得到了一张近似Hessian行列式图，但是在金字塔图像中分为很多层，每一层叫做一个octave，每一个octave中又有几张尺度不同的。

尺度空间通常通过高斯金字塔来实施，图像需要被重复高斯平滑，然后经过不断子采样，一层一层直到塔顶，如sift方法。而SUFR通过盒函数和积分图像，我们就不需要像SIFT方法那样，每组之间需要采样，而且还需要对当前组应用同上层组相同的滤波器，而SURF方法不需要进行采样操作，直接应用不同大小的滤波器就可以了。

![](http://img.blog.csdn.net/20131127194541250)

在SURF中，图片的大小是一直不变的，不同的octave层得到的待检测图片是改变高斯模糊尺寸大小得到的。左边是传统方式建立一个如图所示的金字塔结构，图像的寸是变化的，并且运算会反复使用高斯函数对子层进行平滑处理，上图右边说明SURF算法使原始图像保持不变而只改变滤波器大小。SURF采用这种方法节省了降采样过程，其处理速度自然也就提上去了。卷积滤波器大小变化图：

![](http://img.blog.csdn.net/20151102153457072)

在SURF特征检测算法中,尺度空间金字塔的最底层由9x9的盒子滤波输出得到,对应二阶高斯滤波σ=1.2。为保证盒子滤波的结构不变,后续滤波器的大小最少要有6个像素值步长的变化.

每4个模板为一阶(Octave)。第1阶中,相邻的模板尺寸相差6个像素,第2阶中相差12个像素,第3阶中相差24个像素,以此类推。每一阶的第一个模板尺寸是上一阶的第二个模板的尺寸。因为兴趣点的数量在尺度的方向上退化很快,所以一般情况下取4个Octave就足够了

![](http://img.blog.csdn.net/20151102143538320)



## 非极大值抑制

非极大值抑制在说目标检测的那篇提到了，这里直接说如何使用。将经过hessian矩阵处理过的每个像素点与其三维领域的26个点进行大小比较，如果它是这26个点中的最大值或者最小值，则保留下来，当做初步的特征点。

检测过程中使用与该尺度层图像解析度相对应大小的滤波器进行检测，以3×3的滤波器为例，该尺度层图像中9个像素点之一。如下图中检测特征点与自身尺度层中其余8个点和在其之上及之下的两个尺度层9个点进行比较，共26个点，图中标记‘x’的像素点的特征值若大于周围像素则可确定该点为该区域的特征点。

注意是三维领域哦！

![](http://img.blog.csdn.net/20131127210424156)

然后，采用3维线性插值法得到亚像素级的特征点，同时也去掉那些值小于一定阈值的点，增加极值使检测到的特征点数量减少，最终只有几个特征最强点会被检测出来。



##特征点方向特征的确定

为使兴趣点描述算子具有旋转不变的性能,首先要赋予每一个兴趣点方向特征。

我们在以某个兴趣点为圆心,以6S(S为该兴趣点对应的尺度)为半径的圆形邻域里,用尺寸为4S的Haar小波模板对图像进行处理,求x、y两个方向的Haar小波响应。

Haar小波的模板如下图所示,其中左侧模板计算x方向的响应,右侧模板计算y方向的响应,黑色表示一1,白色表示+1。

![](http://img.blog.csdn.net/20151102191407971)



用Haar小波滤波器对圆形邻域进行处理后,就得到了该邻域内每个点所对应的x、y方向的响应,然后用以兴趣点为中心的高斯函数(σ=2s)对这些响应进行加权。用一个圆心角为PI/3扇形以兴趣点为中心环绕一周,计算该扇形处于每个角度时，

它所包括的图像点的Haar小波响应之和。由于每一点都有x、y两个方向的响应,因此扇形区域中所有点的响应之和构成一个矢量。把扇形区域环绕一周所形成的矢量都记录下来,取长度最大的矢量,其方向即为该兴趣点所对应的方向。

![](http://img.blog.csdn.net/20151102190801811)

![](http://img.blog.csdn.net/20131127210454218)





## 构造SURF特征描述算子

在SURF中，也是在特征点周围取一个正方形框，框的边长为20s(s是所检测到该特征点所在的尺度)。该框带方向，方向当然就是第4步检测出来的主方向了。

然后把该框分为16个子区域，每个子区域统计25个像素的水平方向和垂直方向的haar小波特征，这里的x(水平)和y(垂直)方向都是相对主方向而言的。

该haar小波特征为x(水平)方向值之和，水平方向绝对值之和，垂直方向之和，垂直方向绝对值之和。该过程的示意图如下所示：

![](http://img.blog.csdn.net/20131127211200250)





## OPENCV实现

###Opencv-Python实现

https://github.com/ZZZZZch/ImgThracker



### 实际效果图

选用邹博微博中的图片，右下角为效果截图。

![](https://wx1.sinaimg.cn/mw690/8974ecb3ly1fmoxobg6kij20hj0dv48y.jpg)





# 参考资料

- [【图像分析】SURF特征提取分析](http://blog.csdn.net/tezhongjunxue/article/details/17040779)
- [SURF算法详解及同SIFT算法的比较](http://blog.csdn.net/tostq/article/details/49472709)
- [SURF特征点监测](http://www.cnblogs.com/walccott/p/4956857.html)
- 《Speeded-Up Robust Features》Herbert Bay