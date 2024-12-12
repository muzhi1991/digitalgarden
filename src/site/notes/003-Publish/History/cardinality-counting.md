---
{"dg-publish":true,"permalink":"/003-Publish/History/cardinality-counting/","title":"大数据算法：基数统计","tags":["技术","大数据","算法"],"created":"2024-12-11T18:09:35.729+08:00","updated":"2024-12-11T18:09:35.730+08:00"}
---



[TOC]

## 什么是基数

基数(cardinality，也译作势)，维基百科中的解释是： **cardinality** of a [set](https://en.wikipedia.org/wiki/Set_(mathematics)) is a measure of the "number of [elements](https://en.wikipedia.org/wiki/Element_(mathematics)) of the set"。我们可以理解为一个集合（这里的集合允许存在重复元素）中不重复元素的个数。例如集合{1,2,3,1,2}，它有5个元素，但它的基数/Distinct数为3。

## 应用举例

1. UV统计：统计日、月的独立用户访问数。
2. 电商应用中，统计一段时间内查看某个商品的独立用户数，分析产品的受众。
3. 搜索引擎中，统计一段时间内用户搜索的unique query数量，分析搜索query特征。

## 问题总结

1. 输入数据量大：数据不可一次性加入到内存中。
2. 不具有可累加性：任务不可分割成更小粒度，如天/小时的粒度，然后把特征直接合并计算出月级数据。
3. 对于某些任务，如上面的2、3，需要对key先GroupBy再统计。因此其中有很多稀疏数据。

因此，我们希望算法具有如下特性：

* 内存占用少
* 数据结构易于合并

## 概率算法

实际上，在**大数据**场景中**准确**计算基数十分困难，因此在不追求绝对准确的情况下，使用概率算法算是一个不错的解决方案。常用的算法包括：

* Linear Counting
* LogLog Counting
* HyperLogLog Counting
* HyperLogLog++

## 基础：Hash函数与Bitmap

### HashSet

一个最基本的想法是，使用HashSet，对任何数据进行Hash求值，并放入Hash Set中。一个Hash函数是 $H(M)=h$ （其中M为密文，h为定长的hash值）需要满足：

* 单向性
* 抗冲突性
* 映射分布均匀性和差分分布均匀性

这种方式可以十分**精确**的记录集合的基数，对于128bit的Hash算法，需要的空间是128*N bit （N为基数）。在大数据场景下，基数巨大，无法在单机内存中存储所有的值。

### Bitmap算法

在Bitmap算法中，我们不存储完整的hash值，而是用bit数组的某一位表示某一数据，从而一个bit数组可以表示海量数据。用0表示某一元素不在集合中，用1表示某一元素在集合中，如`0100000011000000`可以用来表示集合{1,8,9}。问题的关键成为，如何选择合适的Hash函数，将目标M映射成bitmap中的一位？我只需要选择Hash函数满足**抗冲突性**。实际算法选择时，以32位的MD5算法为例，需要$2^{128}$bit的空间，这显然是不可行的，因此，我们只取其结果的前8位（32bit），那么整个Hash函数的映射空间就有$16^8$（40多亿，尽量减少冲突）个值，取一个长度为$16^8$的bitmap（大约536MB）。每一位对应Hash函数映射空间中的一个值，初始值全为0。每当有新访问产生，对该访客标识进行Hash，并映射到bitmap中的某一位上，若该位置为0，则置1；若为1，则不作处理。最后统计整个bitmap中1的个数即为基数。

> 如果要精确统计，不使用Hash来映射Bitmap，可以维护一张目标M与index之间的映射表，并增量更新

优点：

* 算法简单直观
* 较为准确，理论上选择足够大的映射空间，减少冲突即可

缺点

* 所需空间依旧巨大，bitmap的长度与实际的基数无关，而是与**基数的上限**有关，即空间复杂度$O(N_{max})$

为了进一步减少存储所需要的空间，我们使用基于概率的基数统计算法。

### BloomFilter

使用BF实现一个简单的概率统计方法。布隆过滤器可以k个Hash函数来判断一个元素是否在集合中。用误判率$P_{fp}$表示判断的准确性。

![](http://fodi.limuzhi.us.kg/images/IMG-41385b7626c8cb19.webp)

误判率估算公式：$$P_{fp} \approx (1-e^{-kn/m})^k $$（集合大小n，bitmap位数m，Hash函数个数k）

$$m=\frac{n\ln P_{fp}}{(\ln 2)^2}$$

$$k=\frac{m}{n}\ln 2$$

```scala
import breeze.util.BloomFilter

val bf = BloomFilter.optimallySized[Int](5, 0.01)
val arr = Array(1, 3, 4, 5, 1, 2, 6, 3, 1)
var cnt = 0
arr.foreach { t =>
  bf.contains(t) match {
    case false => cnt += 1; bf.+=(t)
    case _ =>
  }
}
println(arr.distinct.length) // 6
println(cnt) // 6
```

特点

* 需要记录当前的基数值，新的数据加入时先判断是否在BF中，
* 虽然多个BF可以合并，但是无法计算出新的基数值。


## Linear Counting

### 简介

LC算法是较为简单的概率算法，**计算过程与Bitmap方法类似，实验完成后统计bitmap中0的个数即可**。

核心计算公式如下：

$$\hat n=−m\log\dfrac{u}{m}$$（这里的m为bitmap的大小；u为0的个数；n̂为n的一个估计，且为最大似然估计）

算法过程如下

- 选择一个哈希函数h，其结果服从**均匀分布**
- 开一个长度为m的bitmap，均初始化为0(m设为多大后面有讨论)
- 数据流每来一个元素，计算其哈希值并对m取模，然后将该位置为1
- 查询时，设bitmap中还有u个bit为0，则不同元素的总数近似为$ -m\log\dfrac{u}{m}$

![](http://fodi.limuzhi.us.kg/images/IMG-ee19ea265655aabc.webp)

### 基本思想

直观上，随着Hash不断的映射，使得bitmap中为空桶不断减少（即u值不断减少），对于固定长度m的bitmap，其满足与某种关系，使得我们求解的目标基数值满足随着u的减少，单调增加。

我们可以从概率上推导出上述公式，由于选择的Hash函数满足均匀分布，也就是说集合A中的**每一个元素映射到bitmap中的每一位都是等可能的**。设$C_j$为“经过$n$次元素Hash后，bitmap上第$j$位为0的概率”：

$$C_j=(1-\frac{1}{m})^n$$

~~把整个试验过程看做伯努利过程，即每一bit是一次实验且每个bit间相互独立~~，由于bitmap上每一位都是独立的，所以$u=C_1+C_2+...C_n$的期望为（独立同分布的期望之和）

$$E(u)=\sum_{j=1}^{m}P(C_j)=m(1- \frac{1}{m})^n=m[(1+\frac{1}{-m})^{-m}]^{-\frac{n}{m}}$$

当$m$和$n$都趋于无穷时有

$$E(u)=me^{-\frac{n}{m}}$$

即

$$n=-m\ln\frac{E(u)}{m}$$

因此，可以推导

* 因为bitmap上每一位的值服从参数相同0-1分布，因此u服从二项分布
* 当n很大时，可以用正态分布逼近二项分布，因此可以认为当n和m趋于无穷大时u服从正态分布
* 正态分布$f(x)=\frac{1}{\sigma \sqrt{2\pi}}e^{- \frac{(x-\mu )^2}{2\sigma
{ #2}
}}$的期望的最大似然估计是样本均值
* 我们观测到的值u，**是$\mu$ （即$E(u)$ ）的极大似然估计**
* $\hat{n}=-m\log\frac{u}{m}$为$n=-m\log\frac{E(u)}{m}$的一个极大似然估计。（因为函数$n=-m\log\frac{E(u)}{m}$可逆）

### 应用

#### Hash 函数选择

Hash函数的选择，在Bitmap方法中因为其过于追求Hash函数的抗冲突性，进而导致映射空间m过大。在LC算法中，我们只需要Hash函数具有**分布均匀性**即可。

#### 参数选择

选择合适的参数m，减少u为0情况（从上面的公式看错当u为0时候，值为无穷的，算法失效）

因此我们要选择足够大小的参数m。主要需要考虑的是bitmap长度`m`。**m主要由两个因素决定，基数大小以及容许的误差**。假设基数大约为n，允许的误差为ϵ，则m需要满足如下约束，

$m>\dfrac{ϵ^t−t−1}{(ϵt)^2}$, 其中 $t=\dfrac{n}{m}$ 

为了减少u为0情况

$$m>β(e^t−t−1)$$ 其中$$β=max(5,1/(ϵt)^2)$$

我们可以参考作者论文中给出的实验结果分布图

![](http://fodi.limuzhi.us.kg/images/IMG-7652b21cb4ad0535.webp)

#### 合并特性

LC非常方便于合并，合并方案与传统bitmap映射方法无异，都是通过按位或的方式，合并后重新使用上述公式计算即可。

#### 空间复杂度

可以看出精度要求越高，则bitmap的长度越大。**可以看出对于N是几万以内这张元素数量较少是很有优势的**。但是，随着m和n的增大，m大约为n的十分之一。。因此，**LC所需要的空间只有传统的bitmap直接映射方法的1/10**，但是从渐进复杂性的角度看，空间复杂度仍为$$O(N_{max})$$。

### 误差分析

$$StdError(\dfrac{\hat n}{n})=\dfrac{\sqrt{m}(e^t−t−1)^{1/2}}{n}$$，其中$t=\dfrac{n}{m}$ 

### 总结

该算法改进了Bitmap所需空间，但是需要的空间依然是线性增加的。在**基数较少的情况下表现不错**，一般不会单独使用，往往与下面的算法结合使用。

## LogLog Counting

使用LC算法，对于n（基数）很大情况下，依然需要极大的空间（相比与Bitmap已经是他的十分之一了）。对于上万/亿基数这种情况，我需要更小的空间复杂度的算法。假设基数的上限为1亿，原始bitmap方法需要12.5M内存，而LogLog Counting（以下简称LLC）只需不到1K内存（640字节）就可以在标准误差不超过4%的精度下对基数进行估计，效果可谓十分惊人。

### 简介

LLC算法不再使用Bitmap数据结构，我们依旧需要对对象进行Hash求值，但是，只需关心Hash比特串h中**第一个1出现的位置ρ(h)**（例如h为`098950fc`，则ρ(h)=5），并记录所有ρ(h)中的最大值$\rho_{max}$。

核心计算公式如下

$$\hat n=2^{\rho_{max}}$$（$\rho_{max}$是所有元素中首个“1”的位置最大的那个元素的“1”的位置，$\hat n$是n的一个粗糙估计）

算法过程如下

- 选择一个固定长度m的哈希函数h，其结果服从**均匀分布**，且**碰撞几率极小**。
- 分配一个变量$\rho_{max}$，来存储最大值，大小是：$p=\log_2m$ 即p bit
- 数据流每来一个元素，计算其哈希值并算出第一个1出现的位置$\rho$，判断是否有$\rho>\rho_{max}$，然后更新$\rho_{max}$。
- 查询时，计算$\hat n=2^{\rho_{max}}$，这里$\hat n$是n的一个**粗糙估计**。

### 基本思想

设a为待估集合中的一个元素，h=H(a)，这里把h表示为<u>长度为L的比特串</u>（如h为`098950fc`，写为`00001001100010010101000011111100`），将这L个比特串从左至右依次编号为1、2、……、L。因为Hash函数是均匀分布的，所以这L个比特服从如下分布且相互独立

$$ P(x=k) = \lbrace_{0.5(k=1)}^{0.5(k=0)}  $$

我们可以把上述寻找比特串中第一个1的过程看作一个投硬币试验：当硬币为反面时，记为0；当硬币为正面时，记为1，试验停止，记录投掷次数。设n次试验中，最大投掷数为k。

现在考虑如下两个事件：

* 事件A：进行n次试验，每次投掷次数都不大于k —> $$P(X \leq k)=(1-\frac {1}{2^k})^n$$
* 事件B：进行n次试验，至少有一次投掷次数大于等于k —>  $$P(X \geq k)=1-(1-\frac {1}{2^{k-1}})^n$$

注意到，

* 假设一：如果$n \gg 2^k$时，$P(A) \to 0$，
* 假设二：如果$n \ll 2^k$时，$P(B) \to 0$，

转换为自然语言描述就是，当n远远大于$2^k$时，每次试验投掷次数都不大于k的概率几乎为0，当n远远小于$2^k$时，至少有一次试验投掷次数大于等于k的概率也为0。而这些均与我们观察到的试验结果不符，即**存在至少一次试验的投掷次数等于k，且不存在比k更多的投掷次数（即最大投掷那一次）**。

假设一、二均无法满足，所以唯一合理的推断是$n \approx 2^k$。

### 应用

#### 平均分桶

实际应用时，上述方法由于偶然性而存在较大误差。因此，LLC采用了**分桶平均**的思想来消减误差。具体来说，就是将哈希空间平均分成m份，每份称之为一个桶（bucket）。对于每一个元素，其哈希值的<u>前k比特作为桶编号</u>，其中$2^k=m$，而后L-k个比特作为真正用于基数估计的比特串。桶编号相同的元素被分配到同一个桶，在进行基数估计时，首先计算每个桶内元素最大的第一个“1”的位置，设为`M[i]`，然后对这m个值取**算数平均**后再进行估计，即：

$$\hat n =2^{\dfrac{1}{m} \sum_{i=1}^{m} M[i]}$$

#### 偏差修正

通过数学分析可以知道这并不是基数n的无偏估计。因此需要修正成无偏估计

$$\hat n =α_m*2^{\dfrac{1}{m} \sum_{i=1}^{m} M[i]}$$

其中$α_m$为修正量，它是一个关于m分桶数的公式，计算参考公式，推导过程（略），实际使用过程中会根据m预先算好修正量。

需要注意的是**其无偏性是渐近的，只有当n远远大于m时，其估计值才近似无偏**。

#### Hash 函数选择

1. H的结果具有很好的均匀性，也就是说无论原始集合元素的值分布如何，其哈希结果的值几乎服从均匀分布。
2. H的碰撞几乎可以忽略不计。也就是说我们认为对于不同的原始值，其哈希结果相同的概率非常小以至于可以忽略不计。
3. H的哈希结果是固定长度的。（原论文使用32bit Bash）

#### 参数选择

因为算法精度与分桶的数量有关，所以主要考虑的是**分桶数m**，而这个m主要取决于误差。

这里不证明推导过程，如果要将误差控制在ϵ之内，则：

$m>(\dfrac{1.30}{ϵ})^2$

#### 合并特性

合并时取**相同桶编号数值最大者**为合并后此桶的数值即可。

#### 空间复杂度

空间复杂度与**Hash长度L**和**分桶数m**有关。

* 设基数空间n，$2^L=n$ —> $L=\log_2n$ 
* $\rho_{max}\leq L$ —> 每个桶需要 $p=\log_2L$ bit—>每个桶需要 $\log_2{\log_2n}$ bit大小
* 桶数m —> m*p —> 一共需要 $m\log_2{\log_2n}$

可以看到，LLC算法需要的空间仅仅是**基数空间n的两次log的大小**。这也是loglogCounting算法的命名来源。

假设m为1024，H的值为32bit，每个桶需要5bit空间存储，一共需要5×1024 = 5120 bit = 640字节。此时误差为$\dfrac{1.30}{\sqrt{1024}}$=0.040625，也就是约为4%。

### 误差分析

公式与参数m选择相同，这里不证明，**渐近标准误差**为：

$StdError(\dfrac{\hat n} {n})≈\dfrac{1.30}{\sqrt m}$

### 小结

LLC在**基数大的情况下占用空间极少**。但是，就是当n不是特别大时，其估计误差过大，因此HLL算法在次基础上进一步优化。

## HyperLogLog Counting

在了解LC和LLC之后，我们可以进一步学习实用的算法了。HLLC优化了LLC对离群值敏感的问题，并结合了LC，解决LLC算法在基数较少时误差较大的问题。

### 优化

#### 使用调和平均代替几何平均数

调和平均数：$H=\dfrac{n}{\sum_{i=1}^{n}\dfrac{1}{x_i}}$

$\hat n =α_m*\dfrac{m^2}{ \sum_{j=1}^{m} 2^{-M[j]}}$  其中，重新修正后  $α_m=(m\int_0^∞{(\log_2{(\dfrac{2+u}{1+u})})^mdu})^{-1}$

根据论文中的分析结论，与LLC一样HLLC是渐近无偏估计。**渐近标准差**为

$StdError(\dfrac{\hat n} {n})≈\dfrac{1.04}{\sqrt m}$

#### 分段偏差修正

HLLC中，为了解决LLC在基数较小时偏差大的问题，在小基数时，选择LC估计。设E为估计值：

* 当$E \leq \dfrac{5}{2}m$时，使用LC进行估计。（Small range correction）。注意：**此时每一个桶相当于LC中的一个bit，如果桶非空则为1，否则为0**。
* 当$\dfrac{5}{2}m \leq E \leq \dfrac{1}{30}2^{32}$是，使用上面给出的HLLC公式进行估计。
* 当$E \geq \dfrac{1}{30}2^{32}$时，估计公式如为$\hat n=−2^{32}\log(1−\dfrac{E}{2^{32}})$。（Large range corrections）

## HyperLogLog++

在HLLC的理论基础上，Google对其进行了**工程上的优化**，解决了实际运用的一些问题。我们把HLLC中不使用分段修正的原始算法记作HLLC_orign。

### 优化

####使用64位Hash代替32位

* 使用64bit的hash函数（L=64），仅仅增加1bit（2^5—>2^6）,
* 桶数目m选择在$2^4$–$2^{16}$（$p\in[4，16]$），桶大小为$\log(L+1-p)\approx 6$
* 减少在基数巨大的情况下hash冲突问题，可以处理100Billion的基数的情况。
* 同时避免了HLLC中分段偏差修正的Large range corrections。（2^64太大）

#### 优化基数小时偏差抖动问题

由于HLLC使用渐进偏差进行估计，在基数较小的情况下，HLLC在实践中表现为**偏差偏大**。不考虑使用LC的分段修正，在n为0时，具有约0.7m的固定误差。

我们以m=16384（p=14）为例，偏差随基数数目变化的实验结果如下图所示。

<img src="http://fodi.limuzhi.us.kg/images/IMG-522b7be008e76562.webp" width="400px" />

得出一下结论与修正过程如下：

* 在基数n<5m时，偏差比较明显
* 我们可以使用预先计算的误差对原始结果进行纠正，为此Google进行了大量实验，计算了$p\in[4，16]$时，分别取了小于5m的200个点计算了他们的原始估计值`rawEstimateData`和偏差`biasData`。（值参考http://goo.gl/iU8Ig）
* 在实际运用时，我们计算出原始估计E，并使用k邻近算法从rawEstimateData中选择k个相近的点（k=6）。找出对于的偏差求出EstimateBias。以此作为对计算出修正值：$E' = E-EstimateBias$

在HLLC中为了修正这个问题我们比较LC，HLLC ，HLLC_nobias的平均误差（p=14）：

<img src="http://fodi.limuzhi.us.kg/images/IMG-5e69114e2f314372.webp" width="400px" />

可以发现如下结论

* 在60000以内，HLLC_nobias结果优于HLLC的原始估计
* 对于小数据集，在p=14，n<11500时，LC算法整体上还是优于修正后的HLLC_nobias

**综合这些结论**，我们结合LC，HLLC，HLLC_nobias，来优化我们的算法

* `n < Threshold`：使用LC算法，其中Threshold是一个经验值（值参考http://goo.gl/iU8Ig）
* 在 `Threshold < n < 5m`：使用修正算法HLLC_nobias
* 在`n > 5m`：使用HLLC算法

对比，LC+HLLC的算法和LC+HLLC_nobias+HLLC修正偏差效果如下 ，可以发现**新加入的HLLC_nobias优化了突变值**，使偏差变得平缓：

<img src="http://fodi.limuzhi.us.kg/images/IMG-bc23872e5b4eaa06.webp" width="400px" />

#### 使用稀疏数组（空间优化）

HLLC（64bit Hash）算法中使用了6m的**固定空间**存储数据，但是在基数较少的情况下（n<<m），大部分桶的值为0，这是一个典型的稀疏数组。因此，

* 在稀疏表示的大小`size(list) < 6m`时，使用$(idx, \sigma(w))$ 这种稀疏pair表示，其中idx为桶号， ρ(w)为该桶值。大小`size(list)=(p+6)*x `(x为非0个数)，实际存储使用Integer从高Bit位开始存储。

  ![](http://fodi.limuzhi.us.kg/images/IMG-a470101fc0792c54.webp)

* 在内存中排序存储所有pairs，为了实现**高效插入**数据，我们维护一个零时的集合。插入时直接放入tmp_set中，当tmp_set的大小大于25%的size(list)时，会对tmp_set排序并**批量merge到list中**，合并相同的idx，取最大值，凭顺序插入新idx（这些操作一次遍历即可完成）。

* 在稀疏表示的大小`size(list) >= 6m`时，稀疏表示会转换成原始的表示方法。

通过稀疏表示法，我们通过一点计算开销实现了大量空间的节约。

#### 稀疏数组中使用动态精度（准确性优化）

在对精度要求更高的场合，我们可以利用稀疏数组节约的空间提高在基数较小的情况下的精度p，本质是**临时提高分桶数m**来实现。（实际算法中我们可以设置动态的范围p'）,称此算法为`HLL_sparse1`。

* 临时增加精度p—>p'，`size(list')=(p'+6)*x `，此时的临时分桶数是m'=2^p'。并按照稀疏数组逻辑存储$(idx', \sigma'(w))$。
* 随着数据不断加入，size(list')==6m。需要**对p'降级到p**。此时，需要更新$(idx', \sigma'(w))$为$(idx, \sigma(w))$。
  * 由于p<p'，从$idx'$中选出前p个最大的桶，组成$idx$
  * 更新桶中的值：当新增`p...p'`的这几位都是0时， $\sigma (w) = \sigma(w′)+(p′ −p)$。当`p...p'`这几位有1时，直接算出$\sigma (w) $

使用动态精度后(p'=25)，对比HLLC_nobias 效果如图，可以看到在使用LC算法的阶段（基数小）的时候精度大大增加:

<img src="http://fodi.limuzhi.us.kg/images/IMG-51bb0e1be51dc01e.webp" width="400px" />

> 备注：在Google提供的算法代码中，默认在稀疏数组表示中使用高精度p'，**降级为p时自动切换为正常表示方法**。 若设置的p'过大，会导致过早切换。参考代码EncodeHash/DecodeHash

#### 压缩稀疏数组（空间优化）

由于在稀疏数组表示时我们可以使用更高的精度p'，所以进一步压缩稀疏数组的存储，可以尽可能的提高效果。主要利用这两个特性

1. 之前我们直接使用Integer来存储pair $(idx, \sigma(w))$。在大部分编程语言中整型占用32bit，但是实际上size(pair)大小为(p+6)/(p'+6)。其余的空间浪费了。
2. 稀疏表示的list中，pair是顺序存储的，我们可以利用这一点优化空间。

对应的优化点是

1. **方案一**：使用变长编码的integer，对**较小的值的整数占用较少的空间bit**。称此算法为`HLL_sparse2`。
2. **方案二**：$\sigma(w)$中不存储绝对值，而利用有序性，**存储差值**。对于$a_1,a_2,a_3,...$ 存储为$a_1,a_2 − a_1,a_3 − a_2,…$。这样存储的值又小了，理论上需要的空间更小。称此算法为`HLL_sparse3`

此外，还有一个比较隐蔽的优化点（详细描述参考论文 5.3.3 Encoding Hash Values以及代码EncodeHash/DecodeHash）：在使用稀疏数组的高精度p'表示时，都只会用LC算法（此时基数显然很小），是用不到后面的6bit的$\sigma'(w)$值，我们只关心，桶出现在稀疏数组中，表示他为非空的。这6bit的作用是方便我们在转换为原始表示方法的时候恢复桶的值。

**方案三**：在某些情况下，我们可以不存储这个$\sigma'(w)$值。通过前面**对p'降级到p的过程**的分析，我们知道，只有当`p...p'`的这几位**都是0**时，才会用到6bit的$\sigma'(w)$值。因此我们有很大的概率（$1-\dfrac{1}{2^{p'-p}}$）可以不存储这个值，而是是要增加1bit来表示这种存储格式即可：0表示省略了6bit，1表示存储了6bit。称此算法为`HLL++`，伪代码入下：

```shell
if ⟨x63−p,...,x64−p′⟩ = 0 then
	return ⟨x63,...,x64−p′⟩ || ⟨ρ(⟨x63−p′,...,x0⟩)⟩ || ⟨1⟩
else
	return ⟨x63,...,x64−p′⟩ || ⟨0⟩
end if
```

> Google论文中提供了一个实践的建议：选择p'=25，则存储6bit的情况下，使用32bit的int是合适的选择

上述几种算法在稀疏数组表示时可以存储的pair数量（p'=25）对比如下

<img src="http://fodi.limuzhi.us.kg/images/IMG-42b961aa5b4a438f.webp" width="400px" />

### 序列化相关

* 利用列式存储的Dictionary Encoding特性优化存储值：把每个桶作为列存储在文件中，其值的Distinct空间是$2^p ·(64−p′ −1)+2p ·(2^{p′−p} −1) $。减少了字典空间。
* 如果用数组存储，利用Kyro序列化存数组。

### 总结

HLL++算法最终的效果对比原始HLL实现（p=14，p'=25）：

<img src="http://fodi.limuzhi.us.kg/images/IMG-0acba67e50443adb.webp" width="400px" />

参考文章

* [解读Cardinality Estimation算法](http://blog.codinglabs.org/articles/algorithms-for-cardinality-estimation-part-i.html)
* [如何科学的计数？](https://bindog.github.io/blog/2015/02/14/cardinality-counting/)
* [Google的论文](https://research.google.com/pubs/pub40671.html)：完整算法伪代码实现
* [Google论文的中文解读](http://www.cnblogs.com/fxjwind/p/3755300.html)
* [LC论文](http://dblab.kaist.ac.kr/Publication/pdf/ACM90_TODS_v15n2.pdf)
* [LLC论文](http://algo.inria.fr/flajolet/Publications/DuFl03-LNCS.pdf)

实现

* [Spark源码中HLL++的实现](https://github.com/apache/spark/blob/1270e71753f40c353fb726a0a3d373d181aedb40/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/aggregate/HyperLogLogPlusPlus.scala)
* [Spark关于改进HLL++算法的JIRA]( https://issues.apache.org/jira/browse/SPARK-16484)：open状态，作者提供了一种实现
* [在业务中应用HyperLogLog的实例-Spark](https://databricks.com/blog/2015/10/13/interactive-audience-analytics-with-apache-spark-and-hyperloglog.html)
* [Go语言中HLL/HLL++算法的实现&&实验比较](https://github.com/clarkduvall/hyperloglog)
