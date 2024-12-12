---
{"dg-publish":true,"dg-path":"History/learn-actor.md","permalink":"/History/learn-actor/","title":"初识Actor - 从Scala库开始","tags":["技术","编程模型","Scala"],"created":"2024-12-11T18:09:35.722+08:00","updated":"2024-12-11T18:09:35.722+08:00"}
---



## 旧版本Scala中的Actor库(scala.actor)

### 教程
* 官方
  * [Scala Actors: A Short Tutorial](https://www.scala-lang.org/old/node/242)
    * 一个简单的ping、pong的例子，使用react receive实现，并对比了一下，[参考](https://github.com/muzhi1991/spark/blob/branch-0.5/examples/src/main/scala/test/actor/ActorTest.scala)
    * 实现了一个高级的iterator的例子，[参考](https://github.com/muzhi1991/spark/blob/branch-0.5/examples/src/main/scala/test/actor/ActorIterator.scala)
  * [THE SCALA ACTORS API](https://docs.scala-lang.org/zh-cn/overviews/core/actors.html)
    * 作者是库的设计者，讲解了api功能与设计
  * Actor的简单使用 
    * 推荐：https://www.cnblogs.com/gnivor/p/4268689.html
    * https://blog.csdn.net/yuan_xw/article/details/49663099

### 重点

#### Actor设计
- Reactor
  - ReplyReactor
    - Actor

* Reactor:提供**事件模型**能力&&基本的收发功能（包括事件存储分发&&状态控制&&调度&&异常处理）
  * sendBuffer：其他地方对 `reactor send xxx` 的时候，会放入的buffer
  * mailbox：读取数据，会先从buffer中读到mailbox里，再被处理
* ReplayReactor：记录了sender，对他发送消息可以收到回复。并提供回复功能`reply`&&发送并回复功能`!?`、`!!`
* Actor 
  * 提供了receive这个同步方法
  * link和unlink方法, [参考用法](https://github.com/muzhi1991/spark/blob/branch-0.5/examples/src/main/scala/test/actor/ActorLink.scala)
  * trapExit，用法参考上面的例子

#### 常用操作
* 接收数据  
  * react
  * receive
  * reactWithin
  * receiveWithin
* 发送
  * send 或者 ! 异步发送，无返回
  * !? 同步发送 返回类型是Option[Any]
  * !! 异步发送 返回Feature
  * reply == sender ! msg (sender返回消息的发送者）
* 控制
  * loop
  * exit
  * continue
[参考](https://github.com/muzhi1991/spark/blob/branch-0.5/examples/src/main/scala/test/actor/ActorControl.scala)

#### Schedule设计
scheduler用于执行一个Reactor实例
学习源码了解Actor中schedule结构
IScheduler接口
  - DelegatingScheduler -- 委托类
    - Scheduler 一个单例object
    - DaemonScheduler
  - ForkJoinScheduler（默认） -- 使用ForkJoinPool执行任务 了解ForkJoin模型底层原理
  - ResizableThreadPoolScheduler/ExecutorScheduler/SingleThreadedScheduler 其他方式执行任务：各种线程池
  - SchedulerAdapter -- 自定义Scheduler时使用，使用上面的单例Scheduler

ManagedBlocker原理：优化forkJoinPool中的io操作--某个线程阻塞时，启用新线程（Actor中在阻塞调用之前总会使用他通知ForkJoinPool）

调度对象：
ForkJoinTask接口
 - RecursiveTask
 - RecursiveAction
   - ReactorTask（actor控制的核心实现 -- Control异常的处理）
     - ReplayReactorTask
       - ActorTask
         - Reaction


#### Channel

Channel的作用:
* 提供『固定类型』的数据传输（对比Actor|Reactor是Any类型）

Channel设计：
* channel内部其实封装了一个Actor（名字是receiver），他本质上就是实现了一个数据类型的转换（InputChannel&&OutputChannel接口中定义的方法）
* 因此，在功能上相同(Actor实现了InputChannel&&OutputChannel(父类Reactor中))

Channel的共享方法:
* 直接传递Channel的引用
* 在消息中传递Channel

[实例代码](https://github.com/muzhi1991/spark/blob/branch-0.5/examples/src/main/scala/test/actor/ActorChannel.scala)

#### Future的使用
* Actor的 `!!`操作返回future
* 直接使用Feature处理异步问题
* [参考代码](https://github.com/muzhi1991/spark/blob/branch-0.5/examples/src/main/scala/test/actor/ActorFuture.scala)


#### Remote
惯用法：RemoteActor.classLoader= getClass().getClassLoader()
* 上面的classloader是RemoteActor的一个成员变量
* 理解[java的Classloader](https://en.wikipedia.org/wiki/Java_Classloader)
使用方法：
* [Client](https://github.com/muzhi1991/spark/blob/branch-0.5/examples/src/main/scala/test/actor/ActorRemoteClient.scala)
* [Server](https://github.com/muzhi1991/spark/blob/branch-0.5/examples/src/main/scala/test/actor/ActorRemoteServer.scala)
### 待研究
actor用于并发的问题（lock free）
https://www.chrisstucchio.com/blog/2013/actors_vs_futures.html
https://www.chrisstucchio.com/blog/2013/actors_vs_futures.html

区分：block vs suspend==receive vs react

如何表示空的偏函数
https://stackoverflow.com/questions/7188933/empty-partial-function-in-scala

  



