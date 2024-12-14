---
{"dg-publish":true,"dg-path":"History/Android构架系列之二-MVP&&Clean理解与实践之问题解答与总结.md","permalink":"/History/Android构架系列之二-MVP&&Clean理解与实践之问题解答与总结/","title":"Android构架系列之二--MVP&&Clean理解与实践之问题解答与总结","tags":["技术","Android","主框架"],"created":"2016-06-11 15:41:52","updated":"2016-06-11 15:41:52"}
---


Android框架系列之二持续了两个月，本文是这个系列的一个小结，解答开篇提出的一些问题的思考与结论。


## MVP相关
### Clean与MVP的关系是什么？
区别：

* 从概念上说：Clean是三层构架（一种系统构架）的一种具体表现形式，是一种设计原则。MVP则是一种设计模式，用来连接M与V的方式。
* 从使用范围来说：Clean使用于整个系统，其业务层（Model）十分复杂。MVP倾向于**薄Model的表现层**构架（客户端，展示层）。
* 从抽象层次上说：Clean更抽象一些，MVP更加具体。

联系：Clean构架是在MVP上使用了三层构架！

### MVP中V层接口的设计原则是什么？
首先，要明确接口设计的目的：

* 复用
* 测试

对于MVP中V层接口，它的目的具体而言是：**方便View的复用，方便Presentor的mock测试**。总而言之，方便View的行为切换。
因此，很明显View接口的设计必须：

1. 如果View要被复用，思考View对外暴露的操作是否方便被不同的Presenter使用。PS：理解什么是View的复用，参考其他问题。
2. 原子操作 VS 组合操作？如果是是Passive View的模式则暴露的接口尽量是原子操作（如显示list的一个item），组合逻辑放入Presentor中。否则，还是按照功能暴露接口（如显示一个List的所有item）。**推荐后者。**
3. 。。。。。参考 MVP中P需要主动去V中区数据吗？还是V主动把数据给P（PV？SoC?)

###  MVP中V与P的对应关系？1：1，n：1，1：n？？
* 独立的View（不是一个组合，如ViewGroup）：原则上View：Presnter==1：1
* 对于组合View（ViewGroup），如果ViewGroup内部控件比较简单，则View持有一个Presenter即可，如果内部是复杂的多个自定义控件，而且可以分块（分块原则是：每块之间关联尽量少，最好没有关联！），则View可以持有多个Presenter。
* 不建议View：Presenter是n：1，既一个Presenter持有多个View接口。这种情况，可以重构View为一个暴露接口，如果View很复杂，可以参考上面一条。

### MVP中V的所有操作都必须经过P吗？即使不涉及M的修改。
* 一般而言，不是！View内部可以很复杂，但是对外接口要尽量简单。
* 对于Passive View的MVP而言，对外接口可能是原子化的，View内部的逻辑这尽量简单
* 对于Soc的MVP，即使有一些修改Model的操作都可以不经过Presenter。

### MVP中P需要主动去V中区数据吗？还是V主动把数据给P（PV？SoC?)
这个问题本质上是View接口的设计问题，MVP中P需要主动去V中区数据会导致View接口中方法数过多，V主动把数据给P，导致Presenter的方法的参数过多。建议

* 尽量是通过参数传递给Presenter，Presenter的逻辑比较简单
* 但是一写View的通用状态（整体状态）可以暴露在View中作为接口

	> 在Google demo中，Fragment的View接口有isActive()表示是否UI可见。这个是有必要的，因为model返回后不用它来判断是否需要显示结果。
	
* PassiveView中可能获取状态的接口比较多一些，待讨论。

### 其他问题
1. Presenter需要设计接口吗？
	核心：View是否要被复用，如**一种常见的需求：相同的界面，加载不同的数据。**
	* 一般情况View不会被重用，Presenter接口没有必要，一个Presenter一般就用在一个地方，Presenter也不用mock来测试，直接测试Presenter本身就好。
	* 除非遇到这种情况：**Presenter的行为需要切换**，如有默认行为和自定义行为，或者相同的界面，加载不同的数据。
	
	> 但是Google demo中为Presenter设置了接口，和View集中在Contract类中，除了比较清晰以外，这个例子没有发现什么优势。
	
2. View怎么复用？
	View可以绑定不同的Presenter。因为View与Presenter耦合（双向绑定），如果需要解耦Presenter，**如果View需要被复用，Presenter必须使用接口，那么View可以被不同的Presenter使用**。但是，**一般情况下，我们没有View被复用的需要，更需要的是View可被测试（Mock）。**
	
3. Presenter怎么复用？
	Presenter可以用绑定不同的View。因为MVP中View的接口是强制的，所以Presenter与View是解耦的。**可以切换不同的View，这个对于测试Presenter时，View的Mock是必须的**。至于其他的需求这没有什么了。


## Clean相关

### Clean中Presentation与Domain，Domain与Repository之间接口设计是必要的吗？
这些接口的目的是Domain层（具体是UseCase）可以被复用，且可以被测试。
因此，最简单的方式是UseCase必须持有Presentation接口与Repository接口。UseCase自身未必需要是接口。思考：UseCase自身是接口作用是什么？

但是，参考Google Demo：

* UseCase没有持有Presentation，以给它返回值，而是通过命令模式，传递在Domain层定义好的实体(xxxResponse)，实现解耦。（解耦的另一种思路，同样Clean的RxJava版本也是）
* 令人意外的是UseCase持有Repository，没有通过接口：
	* 为了测试Domain，Repository内部有接口可以实现Mock。
	* 对于复用，猜测这样的原因是：把Domain和Repository看做一个整体的Model，我只要Model整体可以被复用即可。

###  可否把Clean构架理解成一种AsyncTask？
可以，相当于把AsyncTask分层了，doInBackground放入了Model(Domain+Repository)。但是不仅仅是AsyncTask，Clean的核心是Model的可测试性，UseCase可以测试。AsyncTask也是耦合很严重的。而且不推荐使用AsyncTask。

### Domain中的UseCase是什么？它重吗？
这个很难理解，Google认为对于移动应用Model很轻

### 一个UseCase应该做多少事情？一件？一组相关的？划分逻辑是什么？
一件事，一个execute，执行一个任务，一个request一个reponse。
必须是业务逻辑。

我之前理解UseCase也叫Interactor，因此，一个用户操作触发的事情就是一个UseCase。实际中这不合适（Google Demo也不是这样干的，而是划分了更小，尽量复用UseCase来组合）


### 多个UseCase如何组合
Goolge的Demo中是在Presentation的Presentor嵌套调用。
需要优化：

* Rxjava流式调用
* 新的UseCase组合其他多个UseCase

### 后台线程的事件与UseCase如何通讯？
由于Clean构架是MVP的改进，也存在同样的问题。这个问题是可解决的，使用EventBus通知Presention去调用Domain即可（为什么，参考MVP一文）

### Domain的UseCase，需要默认在非UI线程中运行吗？
建议是：最好在非UI中运行，但是对外可配置。

### MVP中Presenter是否能设计为单例？Domain不能设计为单例？Repository必须设计为单例？
* 理解什么是Presenter，就不会有这个问题了，除非View是单例。。

	> 如果有单例需求，思考是不是Presenter用错地方了，那是Controller，Helper那些干的事情。
* Domain -- 非单例，通过注入由外部构建加入到Presenter

	> Domain层缓存问题
* Repository -- 单例


###  *内存缓存放在哪一层？Repository还是Domain还是都需要？前者缓存原始数据（未处理计算），后者缓存处理后的数据。如果一个domain需要多个Repository然后计算，结果数据需要缓存吗？*

这个问题存疑，现在先解答一部分

* Repository的原始数据缓存在某些情况（为了性能，很多情况都必要）下是必要的。
* Domain层缓存在复杂计算的情况下也是需要的，但是设计是一个问题，我暂时使用的是一个UseCase引用了一个管理缓存的UseCase（管理缓存UseCase内部有个全局静态的缓存池）

### Domain的UseCase是否需要缓存设计？
我们应当设计薄Model的应用，**如果需要Domain缓存，应当思考你的Model是不是有点厚了**

###  Presenter是否需要缓存的设计？还是仅仅是数据引用？

没有缓存设计，仅仅是数据引用，这个引用放在View中也没有问题。（如Adapter中）

### Clean中Presentation层与Domain层的UseCase职责分别是什么？
还是理解Presentation的Presentor是展示器，负责展示相关的操作，而UseCase则是业务操作。一个例子：分页加载数据。

* UseCase中出数据加载，去重合并，根据传入的页码返回数据
* Presentor负责记录当前展示的页码和显示到了哪一条（一般是个id）

### *一些Presentation层的Adapter，如何优雅的与Presenter结合*
暂时的方法是，Adapter传入了Presentor的引用。
另一种思路，没有实践：
ViewHolder---View
Adapter--- Presentor
在Adapter中调用UseCase

### Data层的理解，什么才是data，图片是吗？
Data层属于Model，因此，Data是业务数据，不是没有业务含义的图片，文件。这些文件图片可以在工具类中直接使用即可。

### DBFlow可以启用他的缓存实现原始数据的缓存吗？
理论上可以，没有实践。

## 其他

### *Model层，如何设计*
指的是MVP中的Model，或者Clean中的Domain+Data
这个问题需要专门的专题讨论，autoValue等思想是否可以借鉴。

### *Cache如何设计合适，如何与原始数据同步？是否有最佳实践？*
这个问题很大，初步的探索类似于Google中HashMap+dirty处理的方式，是否有完善框架还没有调研。

## 总结
写这个专题真的很吃力，一方面自己的能力有限，实践也不多，对于设计 的驾驭能力显然还不够。其次，自己对于这方面的理解也比较肤浅，好的构建的也有限（甚至不能定义什么是足够好的构架）。这是一个要经常回头思考的问题。
这个系列都没有给出一个构架的实例，有这么几个原因：首先，构架必须是服务于业务的，不同业务选择的构架迥然不同，（例如Model层问题）。其次，自己的代码没有Google官方的示例优雅，尤其是参入了一些曲解之后可能会误导他人，从学习的角度还是入手原始资料较好，即使是开始一个新的项目，也建议以此为蓝本作修改来适应具体的需求。
后续，几个需要补充的问题，Model层设计和Cache设计会有专题来讲，希望在有深入的理解之后再继续讨论这个问题。
