---
{"dg-publish":true,"permalink":"/003-Publish/History/Android构架系列之二-常见框架分析/","title":"Android构架系列之二--常见框架分析(一）","tags":["技术","Android","主框架"],"created":"2024-12-11T18:09:35.734+08:00","updated":"2024-12-11T18:09:35.734+08:00"}
---



在搭建新项目的过程中，调研了一些常用的开发框架，恰好近日Google公布了[一系列框架的梳理](https://github.com/googlesamples/android-architecture)（未完成），对Android开发者给出了一些指导，颇有裨益。总结了如下几种模式：

1. 纯MVP开发
2. MVP+Loader
3. MVP+Databinding
4. MVP+ContentProvider
5. MVP+Dagger
6. MVP+Clean构架

总体而言，基本涵盖了Android中的流行的开发构架方式。这些开发框架是以MVP为基础，再结合其他工具，或者再次细分某些层，实现更好的复用。这里首先简介一下MVP，然后将上述6个框架按照我的理解（可能不完备）进行归类总结。

## MVP分析
MVP是上述六个框架的基础，对于MVP的认识，最好结合MVC、MVVM这两个框架进行学习，一个基本的理解参考阮一峰先生的[这篇文章](http://www.ruanyifeng.com/blog/2015/02/mvcmvp_mvvm.html)。
我总结几点：

* M层，即`model`，负责数据部分，如网络请求、数据库，获取数据。
* V层，即`view`，负责数据展示，Android的View或者Fragemnt
* P层，即`presenter`,核心，负责从M获取数据，然后控制V更新。
* **核心：M与V不直接交互数据，必须通过P**
* 具体而言：V实现某个UIController接口，暴露自己实现的UI功能，如显示Progress等，P持有V的接口。V同时也会持有P，他们相互依赖（最好V/P都是通过接口依赖）。

	> 流程是：V收到用户操作-->调用P的某方法-->P去M获取数据-->获取成功-->P调用V暴露的接口去更新界面
	> 注意：有些框架可能把Activity看着`Presenter`，这可能导致Activity中有过多的代码，建议还是抽出一个Presenter类，Activity仅仅看做一个View层实体。

## 构架分析

### MVP+Loader/MVP+ContentProvider

这两个构架都是在P层与数据M层之间进行优化，主要**优化了P层从M层获取数据的方式**，使用Android原生（3.0开始）提供的`Loader`异步加载机制，从数据库/网络等地方加载数据。`CotentProvider`思想类似，使用了`CotentProvider`方式，分离数据部分，实现解耦。

### MVP+Databinding

这个框架可以看着是MVVM框架了。Databinding是利用Android提供的一个库实现了数据绑定的功能，即某个`Entity`内容可以直接反应到`View`中，同样`View`一旦修改，`Entity`也会变换。**本质上，这种方式简化了MVP中P层与V层的关系**，P层处理完的数据，放入某个实体中就可以显示在界面上。当然也有其缺陷。

### MVP+Dagger
这就比较好理解了，将MVP中对象的构造使用Dagger2代替，是一种强制性解耦的措施，具体可以参加[之前的文章](http://limuzhi.com/2016/03/06/Android%E6%B3%A8%E5%85%A5%E6%A1%86%E6%9E%B6Dagger2%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/)。

### MVP+Clean构架
这里主要讲解一下Clean构架，什么是Clean构架，这在网上有很多介绍的文章，其实就是一种基于MVP再次分层的构架，历史比较悠久了（2012年？）。下面详细介绍。

#### Clean学习

基本的学习看Clean构架的文章：

* [原始文章](http://blog.8thlight.com/uncle-bob/2012/08/13/the-clean-architecture.html)
：比较抽象，难以理解，不建议首先学习！
* Clean构架详细的解析文章，[原文](http://fernandocejas.com/2014/09/03/architecting-android-the-clean-way/)，[中文翻译](http://zhuanlan.zhihu.com/p/20001838)：可以直接学习这一篇文章，GitHub有详细的[代码和讲解](https://github.com/android10/Android-CleanArchitecture)，注意作者迭代了两个版本,第一版是没有依赖注入，RxJava那些框架的，相对比较好理解。
* [国人讲解的Clean框架](http://blog.chengdazhi.com/index.php/101)：建议优先学习这篇文章，构架在在一个工程Module中，代码干净，比较好理解，但是可能对Clean的理解并不透彻。

#### Clean基本思想
总结一下Clean的基本思想：
首先，我们应该**从MVP出发理解Clean构架**。
Clean构架中，把整个项目分成3个模块，`presentation`（表现层），`domain`（领域层），`data`（数据层）。

* 表现层中，就已经包含了完整的MVP构架，但是与传统的MVP不同的是：
	* 这里的M，更加倾向于ViewModel，即仅仅用来存放展示数据的`Model`，与数据库等无关！『从数据源获取数据』的功能被剥离出去，形成了`data`层。
	* 这里的P，是不完整的P，它的核心功能，『调用M从数据源（真正的M）获取数据&&处理数据』的逻辑被剥离了出去！形成了`domain`层。
	* 这里的V，是完整意义上的`View`
* 领域层中，即以前MVP中P剥离处理的调用M和数据处理的逻辑。在Clean种，一个`domain`中有n个`usecase`，一个`usecase`相对独立，处理单独的业务逻辑，其设计模式参考**『命令模式』**。
* 数据层中，是真正的从数据源获取数据，`domain`层会调用它，获取数据，注意：**domain层并不关心数据的来源（网络/本地文件/数据缓存）**，至于内存cache的处理，是放在`domain`还是`data`，需要再研究。（server端看法：放在`domain`中，因为数据层更倾向于io操作）

#### Clean的数据流
presentation（V）--> presentation（P）--> domain(usecase) -->data-->domain(usecase,数据回调)--> presentation(P，结果回调) -->presentation（V,P调用V的UIController接口）

#### 几个注意点与疑惑
* 多个model实体，即`presentation`、`domain`、`data`需要三层独立的Model，思考：**为什么，如何优化？**
* `domain`层是纯java层，**理想状态是纯java module**，与android无关。如果多module项目，这需要考虑，使用纯`java module`后，`presentation`无法直接访问`data`（跨层访问），`domain`也无法访问`presentation`和`data`（**只能通过放在domain层的接口来访问**）。虽然也好处是强制解耦，开发人员也不会误操作，但是限制很大。这也直接导致了第一个问题。
* 太多的接口，总结一下在尽量接口设计且`domain`是纯java层时，需要这些接口
	* `presentation`内部，VP之间一组（两个，ui，presenter）
	* `presentation`与`domain`之间一组（两个，调用`usecase`与回调，必须放在`domain`中）
	* `domain`与`data`之间一组（两个，调用repository与回调，必须放在domain中）
	* `data`内部，调用网络异步（一组）、数据库异步（一组）
	
	> 如何解决：1.通用接口设计，而不是更具业务不同（如各种callback）2.能不使用接口的地方直接调用，但会丧失Clean的优势（在纯domain，java层时有些接口不可避免） 3.**用Rxjava不仅能减少接口，还能避免深层次的回调逻辑**
* 关于`domain`的理解，一个`usecase`的粒度是多大？只做一件事，然后通过usecase组合来实现新的`usecase`来完成业务？还是在一个`usecase`中写n个逻辑与回调，来实现（内存缓存数据处理大大方便，这也涉及内存缓存数据存放的问题）。尝试从『命令模式』角度思考。
* Clean的一大优势是，可以强制所有的`usecase`必须在非UI线程完成，只有`presentation`在UI线程，大大的减少了阻塞UI线程的可能性，即`domain`层在非UI完成，那么data层的接口是否还需要异步接口，还是直接使用同步接口？

## 总结
上面介绍了Google推荐了几种构架，主要说明了MVP与Clean构架。基本上MVP是现在项目的标配，虽然Clean清晰，可维护性高，可复用性强，但是是否使用Clean，则看具体情况，也可能对其适当的简化后使用。
这里Clean构架的[demo](https://github.com/android10/Android-CleanArchitecture)是一个比较高级的样例，综合了Rxjava，Dagger2，OkHttp/Retrofit。混合使用这些技术学习曲线还是比较陡峭的，在实际的构架中，综合团队人员水平，往往用接口代替Rxjava，不使用Dagger2。
其次，Clean构架可能显得臃肿，**尤其实际中业务逻辑复用的程度并不高**，所以，我们需要简化Clean构架以符合我们的业务需求。
