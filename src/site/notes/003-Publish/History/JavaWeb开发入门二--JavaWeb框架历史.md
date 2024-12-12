---
{"dg-publish":true,"permalink":"/003-Publish/History/JavaWeb开发入门二--JavaWeb框架历史/","title":"JavaWeb开发入门二--JavaWeb框架历史","tags":["技术","JavaWeb","后端"],"created":"2024-12-11T18:09:35.733+08:00","updated":"2024-12-11T18:09:35.733+08:00"}
---



本章先务虚地介绍一下JavaWeb技术的发展历史，由此引出当前使用的一些技术的来龙去脉。比如Spring技术就是在JavaWeb发展的过程中形成的。由此，我们可以更好的体会这些技术要解决的问题和意义。参考资料如下：

* [Web MVC简介](http://jinnianshilongnian.iteye.com/blog/1593441)
* [Java帝国之Java bean上](http://mp.weixin.qq.com/s?__biz=MzAxOTc0NzExNg==&mid=2665513115&idx=1&sn=da30cf3d3f163d478748fcdf721b6414#rd) 
* [Java帝国之Java bean下](http://mp.weixin.qq.com/s?__biz=MzAxOTc0NzExNg==&mid=2665513118&idx=1&sn=487fefb8fa7efd59de6f37043eb21799#rd)

## JavaWeb技术的发展历史

### JavaWeb框架的发展历史

JavaWeb框架的的发展基本遵循从简答到复杂的过程，复杂的框架带来的好处是清晰的业务逻辑、明确的分层、更可扩展可维护的代码。基本遵循这么一个过程：**从简单的Servlet&&JSP技术（无分层），到Model1模型（两层），到Model2模型（MVC三层）**。最后到SpringMVC的**服务到工作者模型**（更细的分层，基于MVC）。

#### 第零步：纯Servlet编程（无框架时代）

最先有的Servlet技术，正如我们在简单的Servlet例子中看到的，单纯地处理request，调用业务逻辑，输出html的数据流。所有过程都在sevice函数中（或者doGEt等），以字符串形式拼接并输出HTML语句也十分低效。这是最原始的方式，也是所有的技术基础，当我们不使用框架时，可以使用基本的设计模式知识实现简单的框架（如命令模式）。

优势：

* 理解简单，没有框架知识
* 高性能

劣势：

* 直接output html语句十分低效
* view的代码与业务逻辑代码都在service一个函数中，无法复用。
* 没有框架导致基本的功能都要手动实现，如路由分发

#### 第一步：JSP编程（无框架时代）

为了改进Servlet技术的问题：直接用字符串输出view层的代码（HTML代码的拼接）太低效了。为此人们**基于Servlet开发了JSP技术**。JSP技术的目的在于使用`*.jsp`文件代替`*.html`，在jsp中不仅有HTML代码，还有java代码。最后由一个处理工具将*.jsp**转换成**一个Servlet类（这样就回到了上面一步，Servlet中有输出HTML的语句，**这里面的HTML的输入语句自动生成**）。该技术的实质是对Servlet技术的优化，解决了上面提到的第一个问题。

![纯JSP技术]()

优势：

* 不需要写一堆output方法了
* jsp标签库加入了很多便捷方法
* 理解相对简单，没有框架（jsp的使用方法要学习）
* 复杂度不算高，性能可以

劣势：

* **没有解决view层与业务逻辑混合的问题，只是现在所有代码都到了jsp文件中了**
* 这种混合引入了复杂性，jsp还是比较恶心的。一堆html中插着java代码还有一堆符号，头晕。

#### 第二步：Model1模型（2层模型框架）

为了解决上述问题，人们**基于JSP技术，进一步改进**。为了把复杂的业务逻辑分离出去，开发了Model模型。JSP文件负责View层和调用业务逻辑，独立出去的业务逻辑形成JavaBean实体作为Model层。**JavaBean实体是一个满足一定规则的Java类**（如必须要属性的get set方法，复杂的EJB即企业级JavaBean规则更多）。

![Model1模型](http://sishuok.com/forum/upload/2012/7/1/d3b5b2ec88706fff7ef4b98d110837d1__9.JPG)

优势：

* 保留了之前JSP开发的基本优势
* 解决了业务逻辑混合的问题，JavaBean负责数据处理与数据持久化（数据库操作）操作，JSP文件相对清晰。

劣势：

* JSP文件除了有**View层代码还有很多判断逻辑代码**，主要是对Model的操作（大量if else的java代码）。View依然不可复用。
* 除了简单的JavaBean规范。其他规范发展了越来越复杂，尤其是EJB。
* JSP混合的那些java代码和符号一如既往的恶心。维护jsp文件依旧心累。

#### 第三步：Model2模型（经典的MVC三层模型）

为了追求干净的View，工程师把Model1模型中混合在View中的控制逻辑单独分离出来，成为Controller，形成了著名的MVC三层框架。至此表示View的JSP中只有纯的视图代码。

![MVC三层架构](http://sishuok.com/forum/upload/2012/7/1/a6b7b2ca293e610a7a2b32e47a16d718__10.JPG)

优势：

* 分层清晰，View与Model彻底分开，Controller负责两者的结合。
* View现在是单一职责，只负责UI和填入展示的值，**没有了表示业务逻辑的Java代码**。前端可以单独开发，只要约定好需要的数值即可。

劣势：

* 不是直接请求JSP了，而是通过Servlet，**Controller可能变复杂，里面有大量的路由逻辑**，但是通过合适的设计可以分离出公用部分来解决。目的是只写业务相关的部分即可。
* 还有一些不灵活的地方，比如视图部分只能用JSP技术吗？视图的渲染只能用ServletAPI吗？这部分可以通过设计框架优化。
* Model层部分（JavaBean）可能含义复杂的业务逻辑，**还需要分层。**比如从整个系统角度看分为三层：表示层、业务层、持久层，就是所谓的三层架构。表示层常用架构就是MVC（上面的除了JavaBean的部分），业务层（Model）常用架构模式分为事务脚本模式和[**领域模型模式**](http://www.infoq.com/cn/articles/cjq-ddd)等，持久层架构模式有入口、数据映射器（ORM）等。**（需要demo）**，[参考](https://segmentfault.com/q/1010000000334139)。
  * 服务层：提供具体业务功能模块，通过WebService调用，内部聚合了多个领域对象（UserService.java）
  * 领域层：大量业务实体对象。功能是组织数据，包含业务逻辑。内部对象直接可能互相应用（需要设计）
  * 持久层：数据库访问操作（UserDao.java）—》数据库中对象是（User.java）

#### 解决方案：服务到工作者模型（SpringMVC的Model2方案）

Model2模型是一个十分清晰的方案，业界有许多框架对其进行了封装实现，提供了工具方法，方便开发，如Struts，SpringMVC等。这些框架解决了（或者部分解决了）上面遇到的问题。

![SpringMVC 服务到工作者模型](http://sishuok.com/forum/upload/2012/7/1/e7fae17e52bb3664d0d3f4ea8db7ae55__15.JPG)

在服务到工作者模型中，我们将Model2中的Controller进行了细化处理：

* 前端控制器：负责路由逻辑即页面的跳转。委托给应用控制器，更具URL选择页面控制器— DispatcherServlet
* 应用控制器：选择具体视图技术（视图的管理）和具体的功能处理（页面控制器），一般是**策略模式**。— Handler Mapping（处理不同Controller）&&View Resolver（依据ModelAndView选择不同View处理器）
* 页面控制器：具体的Controller，功能代码，往往一个请求对应一个Controller，处理参数，调用业务逻辑 — XXController，返回ModelAndView

SpringMVC中具体的分析，可以参考[这篇文章](http://jinnianshilongnian.iteye.com/blog/1594806)。其核心是把业务控制挪到了XXController中，通过写`ModelAndView handleRequest(request, response)` 方法实现。它其他控制逻辑隐藏到了框架中，但是依然有一些问题：

* 虽然不容易出错，但是增加了很多潜规则，学习成本高。
* 业务对象的分解还是需要程序员自己实现的，如上面所说的三层架构。

### JavaBean的发展历史

可以说伴随这JavaWeb框架的发展，他们的具体实现也在发生变化，最显著的就是框架中的**对象实体**的变化：**从普通的Java对象到JavaBean，再到复杂的EJB，最后返璞归真到POJO**，这一系列变化很有启示意义。下面按照时间逻辑来介绍：

1. JavaBean：

   * 定义：具有一个**简单规范的普通Java类**。如User类具有get/set方法，具有AddListener方法。
   * 应用：早期应用在Java桌面程序，支持一些View通过可视化工具设置属性，因此定义了这个规范。如复合JavaBean规范的View可以通过一个界面设置字体大小，颜色等。现在这个术语出现就理解成一些有固定方法的Java对象即可（直接理解成POJO也可以😀）

2. EJB:

   * 定义：Enterprise Java bean，具有复杂规范的一个Java类，**往往需要实现一个复杂接口（对这个类带来了额外的复杂度），然后可以被EJB容器载入**。提供很多高级功能，如分布式管理，消息队列等。

   * 应用：具有负责功能的企业级应用，现在也有很多传统公司使用。

   * 问题：使用复杂麻烦。库框架耦合严重（通过继承耦合）

     ```java
     public TestSession implements SessionBean{
       public void ejbActive(){
         // 为什么要写这些
       }
       public void ejbPassivate(){
         // 为什么要写这些
       }
       public void ejbRemove(){
         // 为什么要写这些
       }
       public void hello(){
         // 有用的代码，业务逻辑
       }
     } 
     ```

     ​

3. POJO

   * 定义：Plain Old Java Object，对于EJB的复杂人们提出了一个相对应的概念（就是针对EJB的）。这中Java类只有普通的Java方法，**不需要继承一个接口才能实现某种功能**。但是又可以**通过依赖注入的方式方便地实现EJB才有的高级功能**。
   * 应用：现在最流行的Web开发方式。使用Spring框架注入POJO，可以方便的实现各种高级功能。

## 总结

JavaWeb的开发方式一路演化，由简单变得复杂，最后形式不断简化，形成现在主流的开发方式。JavaWeb开发的模式与框架基本形成稳定的态势：使用SpringMVC框架来与前端交互（分离M与V），使用Spring的依赖注入来组合与解耦不同后台组件，数据库则是使用ORM来实现便捷的开发。当然，也有一些新的变换，比如SpringMVC中使用注解代替XML，解决大量配置文件的问题。更进一步，SpringBoot来使Web框架开箱即用，提供了一系列类库与约定的配置。还有更为高级的SpringCloud（类似dubbo，但是更通用全面）提供了[后端完整的架构](http://www.infoq.com/cn/articles/basis-frameworkto-implement-micro-service)（包括Model层）包括服务发现与治理的功能。Web开发还在不断发展进化，下面会介绍Spring家族的框架，包括Spring，SpringMVC，SpringBoot的内容，以及Dubbo相关的分布式服务的开发基础，形成一个较为全面的开发模型。



注：此文可能是全沾系列近期的最后一篇文章，本人从事的业务方向调整。这些内容是兴趣所至，而且离普通用户更近，是消费产品不可缺少的一部分，后续有机会会继续写一下。现在的知识水平停留在JavaWeb框架的基本了解（包括Spring，SpringMVC的使用和阅读了部分SpringMVC源码，对SpringBoot是入门（demo）水平，对Dubbo和SpringBoot科普性的学习）。这里特别做个备注，算是技术债了。

