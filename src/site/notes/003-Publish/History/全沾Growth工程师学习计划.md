---
{"dg-publish":true,"dg-path":"History/全沾Growth工程师学习计划.md","permalink":"/History/全沾Growth工程师学习计划/","title":"全沾Growth工程师学习计划","tags":["技术","Guide"],"created":"2016-07-03 14:14:44","updated":"2016-07-03 14:14:44"}
---



关于最近的学习计划，3个月的时间(7、8、9三个月)做一名全沾工程师。全沾是指对各个方面的技术的框架有几个基本了解，Growth是指如果需要深入可以快速成长切入。

## 技术清单

* 前端：
  * 移动端
    * Android/IOS App开发（以后可以RN开发）
  * web开发
    * HTML
    * CSS
    * JavaScript
* 后端
  * NodeJS
  * Redis MySql
  * JSP  Spring MVC+Spring+Hibernate 
* 运维
  * Nginix
  * 虚拟机，容灾
  * 性能优化 分库分表
* 设计UX
  * 设计工具Sketch
  * 设计基础原则学习
* 运营
  * 数据分析 
  * 广告 推广 SEO 流量 

## 学习方式

以一个博客系统为实践相关知识。先使用一种最简单的技术栈实现，然后用其他技术栈实现，形成对比。然后分析不同方案，思考如何大并发等问题，挑选不同技术形成"最优"方案。

## 学习计划
安排12周的时间（大概75天，每天6--8小时）进行学习，重点偏向技术，UX和运营有个基本的认识，没有时间可以延后。

使用思维导图描绘学习路径

### 第一周
总体目标：html/css/javascript完成，nodejs入门，博客启动，初步打通前后端。

- [x] 周一 javascript对象模型，原型模式
- [x] 周二 前后端框架初探，Growth全栈增长工程师指南138
- [x] 周三 html/css 3WSchool教程看完 
- [x] 周四 CSS相关，javascript语法
- [x] 周五 javascript语法 http://javascript.ruanyifeng.com/
- [x] 周末 javascript语法 http://javascript.ruanyifeng.com/


第二周：js完结周，node启动周

-[ ] 周一 [javascript语法完成]( http://www.liaoxuefeng.com/wiki/001434446689867b27157e896e74d51a89c25cc8b43bdb3000)
-[ ] 周二 js相关工具库学习，jquery underscore 
-[ ] 周三 node入门环境搭建
-[ ] 周四 node相关库学习 全局变量 http fs
-[ ] 周五 node相关库学习 stream
-[ ] 周末 总结html css js


第三周：node完结周，项目启动

-[ ] 周一 node框架学习，全栈指南阅读
-[ ] 周二 全栈指南阅读，测试驱动开发。。。（截至到上线章节之前）
-[ ] 周三 调研项目，网上搜索nodejs学习工程，找了一堆项目。
-[ ] 周四 已经无心学习，准备飞往哈尔滨
-[ ] 周五 不在。。。
-[ ] 周末 不在。。。

修整周，感觉nodejs学习了太皮毛了！

-[ ] 周一 刚回来，啥都没干！思考server开发不应该有redis之类的吗？于是搜了一圈，发现学的太皮毛，无法实践啊。
-[ ] 周二 学习[《Node.js 包教不包会》](https://github.com/alsotang/node-lessons)，18篇教程！这篇教程比较实战（express），后面的nodeclude是一个比较好的例子
-[ ] 周三 服务端爬虫（使用js的异步特性），eventproxy asyn库控制并发
-[ ] 周四 node中的测试框架mocha should supertest（还有浏览器前端测试）istanbul 测试覆盖率与测试驱动开发的思想。
-[ ] 周五 其他，线上部署heroku，cookie，session，redis，github的持续集成travis，数据mongodb与mongoose
-[ ] 周末 Promise，**理解中间件**，Connect与Express学习， js api与相关的库总结


第四周：

-[ ] 周一 初步了解redis，zookeeper，dubbo，soa，rpc，thrift，mq，nigix，lvs，docker等一些列概念与关系。形成大局观。
-[ ] 周二 nodeclude项目源码 环境部署，出入看了一下数据库，monggodb nosql文档型数据库，与hbase对比（列存储与行存储），与mysql对比等。
-[ ] 周三 项目模块分析，使用哪些中间件
-[ ] 周四 nodeclude项目源码 登录部分学习，oath2.0,token,sessionId,jwt以及安全问题
-[ ] 周五 nodeclude项目源码 ，以后要总结这个项目
-[ ] 周末 总结nodejs基础知识与下一步学习计划

第五周：

-[ ] 周一
-[ ] 周二
-[ ] 周三
-[ ] 周四
-[ ] 周五 
-[ ] 周末 总结nodejs进阶与express


webpack，react，bower，gulp等前端工具

前段mvc框架

第六周 完整的撸了一遍nodeclub代码，写了nodeclub项目分析的文章

第七周 docker实践，学习了docker（看完了[docker从入门到实践](https://www.gitbook.com/book/yeasy/docker_practice)），[docker-compose 部署nodejs](https://github.com/b00giZm/docker-compose-nodejs-examples)，参考其部署了nginx+负载均衡n个node+redis+mongodb的例子，在该环境下跑起来了nodeclub。

第八周 学习Java Web框架，Servlet+JSP+Spring +SpringMVC

* [极客学院的Java Web简单教程](http://wiki.jikexueyuan.com/project/java-web/) 比较浅，但是有个大概的概念
* [跟开涛学SpringMVC](http://jinnianshilongnian.iteye.com/blog/1593441) 前三章，比较好，很深入，涉及源码分析，讲了服务端构架，但是只讲了springmvc
* 理了Java中Servlet部分的结构（不含tomcat）
* 调研了学习的路径，
  * 先学Servlet，然后学Spring 的IOC，再学SpringMVC
  * 了解一下JSP的原理和简单用法，实际中可能不用它开发。
  * Spring AOP，数据库框架Hibenate？
  * 项目学习--一个SpringMVC项目--[官方showcase](https://github.com/spring-projects/spring-mvc-showcase) 
  * 项目学习--一个SpringBoot项目--[springside4项目](https://github.com/springside/springside4)
* 区分了一些概念
  * WebService（基于http的服务） VS RPC（远程调用，更广的概念）
  * [Restful vs SOAP](https://www.ibm.com/developerworks/cn/webservices/0907_rest_soap/) 思想的区别
  * thrift 理解，还不深刻，与[http+json的区别](http://stackoverflow.com/questions/9732381/why-thrift-why-not-http-rpcjsongzip)


第九周  Spring SpringMVC 数据库

* Spring 还要学一下[这个教程](http://wiki.jikexueyuan.com/project/spring/)，[IBM的教程](http://www.ibm.com/developerworks/cn/java/web/spring.html)
* [Hibernate](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=http%3A%2F%2Fread.pudn.com%2Fdownloads96%2Febook%2F392604%2FHibernate.pdf)
* maven
