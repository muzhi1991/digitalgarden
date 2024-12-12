---
{"dg-publish":true,"dg-path":"History/JavaWeb后台开发知识目录与结构.md","permalink":"/History/JavaWeb后台开发知识目录与结构/","title":"JavaWeb后台开发知识目录与结构","tags":["技术","Guide"],"created":"2024-12-11T18:09:35.729+08:00","updated":"2024-12-11T18:09:35.729+08:00"}
---



开始Java后台系列，前一阵学习了基于Nodejs的后台开发技术，对后台开发有了一个基本的认识，从这一篇开始学习Java后台开发，由于Java语言是我的工作语言，语法本身不做介绍。只谈一谈和后端开发相关的内容，同时与Node章节对比。

|       | Java Web中的技术                     | Node对应                        |
| ----- | :------------------------------- | ----------------------------- |
| 基础语言  | Java                             | Javascript                    |
| Web框架 | Spring + Spring MVC              | Express                       |
| 底层技术  | Servlet                          | Node.js                       |
| 数据库框架 | Hibernate  mybatis            | mongo  others             |
| 核心技术  | Spring IOC框架                     | Express中间件                    |
| 模板工具  | JSP  velocity freemaker | jade ejs  handlebars |
| 构建工具  | maven                            | npm && package.json           |
| 运行环境  | tomcat容器                         | v8引擎                          |
| 特点    | 复杂强大                             | 简单灵活                          |

不同点：

* 基本的架构不同，设计思想不同。JavaWeb中每个request对应一个线程，阻塞处理请求再返回(同步模型)，在node中是只有一个主线程，然后基于事件异步处理请求。当然，在新的SpringWeb框架中也支持类似的方式编程。
* 一般情况下JavaWeb技术用来开发应用服务器，虽然自身也可以作为简单的http服务器，但是http服务器会由其他高性能组件负责（apache或者nginx）。但是node自带高性能的http服务器。这个生成环境的话，要具体情况具体分析。
* Java的Web开发相对而言是比较复杂，在node中，设置路由然后挂上需要的中间件就基本完成了，但是，在JavaWeb涉及到Spring与Spring MVC，引入很多注入的xml配置文件和对底层Servlet的封装（如Spring中的ApplicationContext与SpringMVC中的DispatchServlet），同时引入了大量的其他概念（如依赖注入，注解，过滤器，控制器，Handler函数等等），甚至，**数据绑定，数据类型转换，数据验证**，都有很多细节框架设计。这些东西在Node中（包括express）往往需要自己处理实现。同时，spring框架中还有大量的~~潜规则~~（主要是xml与handler函数中）。



## 知识结构

### Servlet与JSP基础

* web开发基础：servlet与jsp的功能与职责
* 目录结构分析：
  * web-inf文件夹
  * web.xml
    * 路由配置
    * Servlet配置与默认Servlet
* Servlet详解
  * 编写基本功能的Servlet
  * 转发与重定向
  * Servlet API
    * filter
    * listener
  * Servlet源码浅析
  * Servlet运行模型
* JSP介绍：本质是Servlet
  * 用什么，不用什么
  * [jsp 基本语法](http://www.runoob.com/jsp/jsp-syntax.html)
    * 引入Java代码块
    * JSP声明
    * JSP表达式 模板填值
    * JSP指令 page include tablib
    * JSP行为
  * JSTL && EL
    * jstl：JSP标签集合，通过tablib引入
    * EL：全名为Expression Language，就是为了替代<%= %>脚本表达式。



### JavaWeb框架演进与历史

* JavaBean的历史：从EBJ到POJO
* Web MVC框架的历史
  * 请求响应模型（请求驱动）
  * MVC分离—JSP与Servlet
  * Model1&&Model2模型
  * 服务到工作者
* 新的趋势：
  * 异步开发模型--事件驱动编程 reactor
  * Java世界的事件驱动
  * SpringMVC中的应用

### Spring与依赖注入基础

* Spring家族
* 理解依赖注入与容器
* Spring的基本使用
  * 依赖注入
    * bean的理解
      * 支持两个对象模型
      * 生命周期
      * bean的后置处理器BeanPostProcessor
      * bean的继承--parent属性
    * 容器
      * BeanFactory
      * ApplicationContext
    * 依赖注入的方式
      * 构造函数注入
      * setter方法注入
      * 混合使用两种方式
      * 集合的注入
    * 自动装配
    * 依赖注入的形式
      * XML注入 
      * Annotation注入 
        * Spring方式
        * JSR330
      * 纯Java配置文件注入（无需任何xml）
    * 其他
      * 模块化：import其他配置
        * xml中
        * java配置文件中
      * 两大核心总结：bean的生成方式 && bean的依赖生成
      * 最佳实践
  * AOP
    * 概念理解
      * aspect
      * joint point
      * pointcut
      * advice
    * 使用的方式
      * xml方式配置
      * annotation方式
  * 数据库
    * JDBC
      * 理解DataSource
    * 事务
    * 存储过程
  * 其他
    * 静态资源处理
* Spring源码浅析(当做注入工具使用，没有全盘分析源码)
  * BeanWrapper
  * PropertyEditor && PropertyEditorRegistry

### SpringMVC框架基础

* 环境搭建
  * idea配置，目录结构理解
  * 简单使用maven
  * xml文件配置
    * applicationcontext.xml
    * servlet-context.xml
    * 约定
      * mvc:annotation-driven
      * component-scan
* 基本使用
  * 路由设置
  * 处理请求（Controller）
    * 拦截器HandlerInterceptor
      * 应用
    * Handler函数
      * 参数绑定，理解命令对象
      * 参数类型转换
        * PropertyEditor 方式
        * ConversionService方式
        * FormattingConversionServic格式化显示数据
      * 参数验证（与错误处理配合）
        * validator接口的实现
        * validator的使用
          * 直接编码、注入使用
          * **JSR-303验证框架**-Hibernate-validator库，直接使用注解验证，并指定错误信息。
      * **model构建与传递**
      * 数据转换错误处理：带有 `BindingResult errors`参数
        * `Errors`接口
        * 配置错误信息，MessageSource接口（ApplicationContext实现）
    * 会话状态保持
      * session
      * cookies
  * 返回
    * 模板视图与渲染
    * json
      * @RequestBody
      * @ResponseBody
      * HttpMessageConverter
      * ContentNegotiatingViewResolver
  * 异常处理（Controller）
    * @ExceptionHandler
  * 安全
* 源码分析（重点）
  * DispatchServlet
    * 分析DispatchServlet的层级
    * 分析DispatchServlet
      * HandlerMapping：映射到对应的Handler函数
      * HandlerAdapter：构造Handler需要的参数，并调用函数，统一返回ModelAndView。
      * 调用相应的ViewResolver或者异常处理。
  * Controller接口相关类分析
    * WebContentGenerator层级分析
    * Controller层级分析
      * AbstractController分析
    * 以MultiActionController为实例分析
      * 如何处理请求handleRequestInternal
      * 异常
      * 数据转换绑定ServletRequestDataBinder
      * 数据校验Validator

补充：与数据库结合



## 小结

可以看出JavaWeb对比Node.js十分复杂，在Node中，大部分功能通过添加中间件（特指node中的中间件）处理。Node中一些**数据绑定、类型转换、验证、错误信息的返回**也没有类似框架，自己编写代码处理，相对还是轻量级的。
