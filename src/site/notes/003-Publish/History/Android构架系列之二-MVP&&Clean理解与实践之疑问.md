---
{"dg-publish":true,"dg-path":"History/Android构架系列之二-MVP&&Clean理解与实践之疑问.md","permalink":"/History/Android构架系列之二-MVP&&Clean理解与实践之疑问/","title":"Android构架系列之二--MVP&&Clean理解与实践之疑问","tags":["技术","Android","主框架"],"created":"2024-12-11T18:09:35.723+08:00","updated":"2024-12-11T18:09:35.723+08:00"}
---


好几周没有更新博客了，不是忘记或者偷懒，而是想写一个比较复杂的话题，MVP与Clean构架的理解，既然是理解，不深入实践，自然不敢动笔，最近会多整理一些，补一补功课。

最近研读了Google发布的[一系列框架的梳理](https://github.com/googlesamples/android-architecture)中的MVP中的源代码，又学习了一些文章，同时自己也在工作中使用了基于MVP的Clean构架开发了浏览器的文件下载模块。这篇文章主要分析遇到的问题和一些感悟，同时记录一些疑问。
学习MVP的主要内容是

* Google的源码（todo-mvp分支）
* 一篇讲解的文章：[Android官方MVP架构示例项目解析](http://mp.weixin.qq.com/s?__biz=MzA3ODg4MDk0Ng==&mid=403539764&idx=1&sn=d30d89e6848a8e13d4da0f5639100e5f&scene=23&srcid=0414ejxUkZ3mPYQaYU3PJYTd#rd)
* 一篇不错的MVP的总结文章包括Clean构架：[Android MVP 详解（上）](http://www.jianshu.com/p/9a6845b26856)，[下](http://www.jianshu.com/p/0590f530c617)
* [Clean构架的GitHub的Issue](https://github.com/android10/Android-CleanArchitecture/issues?q=sort%3Acomments-desc)中大量提问很有价值
* 其他很多MVP介绍的文章，不一一列举


## 疑问

在使用Clean构架中遇到的几个疑惑，先记录一些，会不断的补充新的问题。尽量按照我遇到的顺序：

### MVP相关
* Clean与MVP的关系是什么？
* MVP中V层接口的设计原则是什么？
* MVP中V与P的对应关系？1：1，n：1，1：n？？
* MVP中V的所有操作都必须经过P吗？即使不涉及M的修改。
* MVP中P需要主动去V中区数据吗？还是V主动把数据给P（PV？SoC?)

### Clean相关
* Clean中Presentation与Domain，Domain与Repository之间接口设计是必要的吗？
* 可否把Clean构架理解成一种AsyncTask？
* Domain中的UseCase是什么？它重吗？
* 一个UseCase应该做多少事情？一件？一组相关的？划分逻辑是什么？
* 多个UseCase如何组合
* 后台线程的事件与UseCase如何通讯？异步通知
* Domain的UseCase，需要默认在非UI线程中运行吗？
* MVP中Presenter是否能设计为单例？Domain不能设计为单例？Repository必须设计为单例？
* 内存缓存放在哪一层？Repository还是Domain还是都需要？前者缓存原始数据（未处理计算），后者缓存处理后的数据。如果一个domain需要多个Repository然后计算，结果数据需要缓存吗？
* Domain的UseCase是否需要缓存设计？
* Presenter是否需要缓存的设计？还是仅仅是数据引用？
* Clean中Presentation层与domain层的UseCase职责分别是什么？
* 一些Presentation层的Adapter，如何优雅的与Presenter结合
* Data层的理解，什么才是data，图片是吗？
* Cache如何设计合适，如何与原始数据同步？是否有最佳实践？
* DBFlow可以启用他的缓存实现原始数据的缓存吗？

## 写作计划
之后的一系列文章包括：

* [MVP&&Clean理解与实践之MVP](2016/05/02/Android构架系列之二-MVP&&Clean理解与实践之MVP/)(done)
* [MVP&&Clean理解与实践之Clean](2016/05/08/Android构架系列之二-MVP&&Clean理解与实践之Clean/)（done）
* [MVP&&Clean理解与实践之实例分析](2016/05/15/Android构架系列之二-MVP&&Clean理解与实践之实例分析/) (done)
* [MVP&&Clean理解与实践之问题解答与总结](2016/06/11/Android构架系列之二-MVP&&Clean理解与实践之问题解答与总结/) (done)






