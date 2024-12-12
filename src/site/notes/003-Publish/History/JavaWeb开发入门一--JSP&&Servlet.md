---
{"dg-publish":true,"dg-path":"History/JavaWeb开发入门一--JSP&&Servlet.md","permalink":"/History/JavaWeb开发入门一--JSP&&Servlet/","title":"JavaWeb开发入门一--JSP&&Servlet","tags":["技术","JavaWeb","后端"],"created":"2024-12-11T18:09:35.725+08:00","updated":"2024-12-11T18:09:35.726+08:00"}
---




这一节主要介绍最基本的JavaWeb开发技术，流行的JavaWeb开发框架的底层都是Servlet与与JSP，因此了解他们是接下来学习的基础。参考资料如下：

* [极客学院的第一个章节，十分基本](http://wiki.jikexueyuan.com/project/java-web/00-02.html)
* [一个台湾的资料，比较偏向实践](http://openhome.cc/Gossip/ServletJSP/index.html)


## WEB开发基础

### Servlet与JSP的功能与职责

在JavaWeb的开发中，我们往往使用Servlet与JSP来实现应用服务器，他们是SpringMVC等其他Web技术的底层基础技术。其中Servlet是核心，**JSP文件最终也会编译成Servlet**。我们可以在JSP中写Java逻辑代码，调用数据库，绑定数据。但是实际项目开发时，应当职责清晰。**Servlet偏向于实现Controller逻辑（Model层），而JSP仅仅用于展示页面数据的View层**。由于JSP功能强大，过于复杂，往往导致代码逻辑混乱，在很多项目中要么只使用JSP的部分功能（如只有EL表达式展示数据），或者干脆代替JSP，选择其他的模板引擎，如Velocity等。

### 认识HTTP协议

任何Web开发技术都无法避免学习Http协议，把握以下几点。

* 应用层协议，基于Socket，TCP，完成可靠通讯
* 由Header+Body组成
* 注意Header中的一些字段，
  * 状态码
  * Method
  * Content-Length
  * Content-Type
  * Cookies

## Web应用目录结构

### 常见目录结构

新建的java web工程目录如下

![](http://fodi.limuzhi.us.kg/images/IMG-e7a8a8348a1bae3a.webp)

* src 源码目录，存放java代码
* web web工程目录，该目录下除了web-inf以为都是可以被浏览器访问的，既`www.xxx.com`方法了index.jsp文件，**建议存放<u>静态</u>资源文件（如css js html）**。
  * WEB-INF（必须） ：重点理解，该目录下不可被浏览器直接访问，**建议存放所有jsp文件与配置文件**
    * web.xml（必须）
    * 其他配置文件，如spring的servlet-context.xml
    * jsp文件
  * index.jsp
  * resource 可以存放静态资源文件

> 目录结构多种多样，但是基本要素相同。还有一种常见的是
>
> * src 
>   * main
>     * java
>     * resource **该目录下的xml会被放到classpath下**，经常放一些配置文件
>     * webapp
>       * WEB-INF  jsp，web.xml，spring的servlet-context.xml等
>       * resource 可以存放静态资源文件
>   * test
>     * java
>     * resource

### web.xml分析

web.xml是web工程的入口配置文件，一般分析一个web项目由它开始。

* header分析，指定了命名空间，重点关注，**默认的命名空间**（xmlns="http://java.sun.com/xml/ns/javaee"的内容），所有的直接应用，都是该命名空间的一个属性，如**`<web-app`**，`<context-param>`,`<servlet>`等。

* **设置路由**：

  *  `<filter-mapping>`标签给filter设置路由，一般都是
  *  `<servlet-mapping>`标签给servlet设置路由

  ```xml
      <filter>
  		<filter-name>csrfFilter</filter-name>
  		<filter-class>org.springframework.web.filter.DelegatingFilterProxy</filter-class>
  		<async-supported>true</async-supported>
  	</filter>
  	<filter-mapping>
  		<filter-name>csrfFilter</filter-name>
  		<url-pattern>/*</url-pattern>
  	</filter-mapping>
      
  	<!-- Processes application requests -->
  	<servlet>
  		<servlet-name>appServlet</servlet-name>
  		<servlet-class>org.springframework.web.servlet.DispatcherServlet</servlet-class>
  		<init-param>
  			<param-name>contextConfigLocation</param-name>
  			<param-value>/WEB-INF/spring/appServlet/servlet-context.xml</param-value>
  		</init-param>
  		<load-on-startup>1</load-on-startup>
  		<async-supported>true</async-supported>
  	</servlet>
  ```

* `<context-param>` 定义一些自定义的初始化参数，这些值最终会被Servlet读入，可以自定义使用。

* `<servlet>` 定义了一个实现了`Servlet`接口的实例，可以定义多个

* `<filter>` 过滤器，定义了一个实现了`Filter`接口的实例，请求在进入servlet之前，会经过他，可以定义多个。

* `<listener>` servlet提供的一个事件通知回调机制，实现了`ServletContextListener`接口，会在启动和销毁时通知

* `<session-config>`

> * 关于 url-pattern 规则
>
>   * 路径映射：以`/`开头且以`/*`结尾---`/static/*`
>   * 扩展映射：以`*.`开头--- `*.js`
>   * *defualt servlet映射**：特指 `/`
>   * 详细映射：详细路径---`/static/a.js`
>   * 不能混合，如 `/static/*.js` 是**错误的！**
>   * [容易混淆的几个设置](http://stackoverflow.com/questions/4140448/difference-between-and-in-servlet-mapping-url-pattern)：`/*`与`/`
>
> * 关于ServletContext，下面源码时会分析，但是要记住，这个Context是一个webapp只有一个，所有Servlet共享的
>
> * 默认Servlet：一个webapp有一个默认的servlet（servlet-name是default），可以通过`default`来引用，常常用来处理静态资源。
>
>   ```xml
>   <servlet-mapping>  
>       <servlet-name>default</servlet-name>  
>       <url-pattern>/static/*</url-pattern>  
>   </servlet-mapping> 
>   ```
>
>   ​

## Servlet详解

### 编写基本功能的Servlet

我们**可以直接继承`Servlet`接口**实现`service(ServletRequest var1, ServletResponse var2)`方法。所有去请求都会走这里，但是为了更加方便，**推荐使用`HttpServlet`**,对http请求进行了分类（get post put del等 ）。

* maven包引入：**注意scope是provided**。因为这里仅仅引用了api，代码在tomcat容器中。

  ```xml
          <dependency>
              <groupId>org.apache.tomcat</groupId>
              <artifactId>tomcat-jsp-api</artifactId>
              <version>8.0.33</version>
              <scope>provided</scope>
          </dependency>
          <dependency>
              <groupId>org.apache.tomcat</groupId>
              <artifactId>tomcat-servlet-api</artifactId>
              <version>8.0.33</version>
              <scope>provided</scope>
          </dependency>
  ```

* 编写Servlet

  ```java
  public class HelloServlet extends HttpServlet {

      protected void processRequest(HttpServletRequest request, HttpServletResponse response)
              throws ServletException, IOException {
          response.setContentType("text/html;charset=UTF-8");
          try (PrintWriter out = response.getWriter()) {
              out.println("<!DOCTYPE html>");
              out.println("<html>");
              out.println("<head>");
              out.println("<title>Servlet HelloServlet</title>");            
              out.println("</head>");
              out.println("<body>");
              out.println("<h1>Servlet HelloServlet at " + request.getContextPath() + "</h1>");
              out.println("</body>");
              out.println("</html>");
          }
      }

      @Override
      protected void doGet(HttpServletRequest request, HttpServletResponse response)
              throws ServletException, IOException {
          processRequest(request, response);
      }

      @Override
      protected void doPost(HttpServletRequest request, HttpServletResponse response)
              throws ServletException, IOException {
          processRequest(request, response);
      }
  }
  ```

* web.xml中引用此Servlet 

  ```xml
  <servlet>
          <servlet-name>HelloServlet</servlet-name>
          <servlet-class>com.test.servlet.HelloServlet</servlet-class>
  </servlet>
  <servlet-mapping>
          <servlet-name>HelloServlet</servlet-name>
          <url-pattern>/HelloServlet</url-pattern>
  </servlet-mapping>
  ```

### 转发与重定向

* 转发：forward，只是web应用内部的跳转（Servlet间？），对用户而言不可知。
* 重定向：redirect，直观来看，浏览器会重新跳转到一个新的url，重新路由并且加载页面。

```java
// 在doGet中，转发给jsp页面的Servlet处理
    protected void processRequest(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        response.setContentType("text/html;charset=UTF-8");
       // 设置attr，可以在jsp中读取到这个值
        request.setAttribute("title", "Hello Servlet");
        request.setAttribute("content", "你好");
        RequestDispatcher rd = request.getRequestDispatcher("/WEB-INF/jsp/hello.jsp");
        rd.forward(request, response);
    }

// 在doGet中，重定向给jsp页面的Servlet处理
    protected void processRequest(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
      response.sendRedirect("HelloServlet")
}

```

jsp代码如下

```jsp
<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>${title}</title>
    </head>
    <body>
        <h1>${content}</h1>
    </body>
</html>
```



#### 几种常见场景

* 静态文件（html,js,css文件）处理
   * web.xml 设置default，调用内置DefaultServlet
* 代码返回html字符串
   1. response设置contenttype--"text/html;charset=UTF-8"
   2. response.getWriter().println输出html字符串
* json返回
   1. response设置contenttype--"application/json;charset=UTF-8"
   2. response.getWriter().println输出JSON格式的字符串

* servlet调用jsp（模板返回）
   1. response设置contenttype--"text/html;charset=UTF-8"
   2. 设置模板变量，request.setAttribute("title", "Hello Servlet");
   3. 返回 request.getRequestDispatcher("/WEB-INF/jsp/hello.jsp").forward(request,response);

* request处理
   * 获取baseurl  request.getRequestURI()
   * 获取query参数和**post的form值**  request.getParameterMap() getParameter


### Servlet API

* Filter：过滤器是一个程序，它先于与之相关的servlet或JSP页面运行在服务器上。可以过滤完之后交给下一个处理，也可以直接返回信息。应用在用户登录判断等场景。一个应用可以有一个或者多个filter在web.xml中配置
  * dofilter函数的写法：
    * 修改request
    * chain.doFilter(req,res);
    * 修改response
* Listener：基于观察者模式，事件回调机制。常见的Listener如下
  * EventListener
    * ServletContextAttributeListener
    * ServletContextListener：应用在Spring的context加载，参考`org.springframework.web.context.ContextLoaderListener`。
    * HttpSessionListener
    * HttpSessionAttributeListener
    * HttpRequestListener
    * HttpRequestAttributeListener

### Servlet源码浅析

#### 接口分析

* Servlet接口
  * `init(ServletConfig config)`：读入web.xml的Servlet配置
  * `service(ServletRequest req, ServletResponse res)`：主要的服务方法
  * `destroy()`
  * `ServletConfig getServletConfig()`


* `ServletConfig`--接口，从**`init()`传入**的参数，可以**获得解析好的web.xml中关于这个servelet自身的`<init-param>`参数**
    * **getServletContext()** 获取所有servlet共享的Context
    * getInitParameter,getInitParameterNames
    * getServletName
* ServletRequest
* ServletResponse
* ServletException
#### 继承关系

Servlet--GenericServlet--HttpServlet
![](http://hi.csdn.net/attachment/201008/26/0_1282804027HHcv.gif)

* GenericServlet 装饰模式，**对ServletConfig装饰了**，子类可以调用ServletConfig接口的所有方法。
* HttpServlet，安照http的header分出，doGet doPost等方法
#### 重要的对象

*   ServletConfig 对象，通过Servlet中的getServletConfig获取，或者直接使用GenericServlet子类的方法。
    *   getServletContext，获取ServletContext
    *   getInitParameter getInitParameterNames：获取servelet自身的`<init-param>`参数。
    *   getServletName
*   ServletContext对象代表当前web工程的上下文，所有Servlet共享，ServletContext通过ServletConfig获得对象。
    * getInitParameter(name) 获取web.xml中配置的 **`<context-param>`**属性值
    * **全局共享数据** 
      * setAttribute 
      * getsAttribute
      * getAttributeNames
      * removeAttribute
    * 资源
      * getResource()路径是虚拟路径,web目录下的文件，即使"/"开头,代表web工程 部署好以后的配置文件的地址.ex:"/WEB-INF/classes/jdbc.properties"
      * getResourceAsStream() 获取文件，返回Inputstream。**最常用，一般用来读配置文件形成Properties**。 props = new Properties(); props.load(in);
      * getRealPath 得到资源文件磁盘上的绝对路径，在web项目中**不能直接使用相对路径访问文件,如 new File**。必须使用绝对路径。
    * servlet 转发
      * getRequestDispatcher
*   ServletRequest及其子类
*   ServletResponse及其子类
### Servlet运行模型

* **servlet容器一般是单进程，servelt的service()方法是一个线程**，必须等service()方法返回后，线程才能销毁。[参考文章](http://www.programcreek.com/2013/04/what-is-servlet-container/)
* 因此存在**线程安全问题**，与node的巨大区别。
* 我们一般在service()方法中进行阻塞调用，这也是Servlet设计的初衷（单进程多线程，对比于GCI）。但是Spring3.0以后支持异步响应（可以参考[这篇](http://www.infoq.com/cn/news/2013/11/use-asynchronous-servlet-improve)）
* 那么，一般应用中有几个Servlet呢？尽量使用一个Servlet。Spring既是这种用法。
* 当我们不使用框架，直接使用Servlet时，参考[这篇文章](http://www.ibm.com/developerworks/cn/education/java/j-intserv/j-intserv.html)（翻译的挺糟糕），使用action模式（命令模式）优雅的划分请求，Spring也是类似的思想。

### 值得注意的细节

* 地址的写法：地址给谁用的。如果是给服务器用的，以`/`代表当前web应用; 如果是给客户端（浏览器）用的`/`代表网站。


* `request.getRequestDispatcher("/my.jsp").forward(request, response);` 转发，服务器内部使用
* `this.getServletContext().getRealPath("/my.jsp");` 获取资源，服务器内部使用。
    * `response.sendRedirect("/JavaWeb/my.jsp");` 重定向，客户端使用。JavaWeb是contextPath，包含在网址里，必须加上。
    * `<form action="/JavaWeb/my.jsp">` html中，客户端使用


## JSP简介

### 理解JSP

JSP（JavaServer Pages）是以`.jsp`为后缀名的一种脚本语言。使Java代码和特定的预定义动作可以嵌入到静态页面中。**它经过编译后成为Java的Servlet类**。通俗的理解，我们编写的.jsp文件（包含html，css，js，java逻辑代码）会在服务端动态处理（执行了jsp中的java代码）成一个由Servlet，最终该Servlet会out.println输出的静态html语句。

### JSP技术的取舍

JSP技术是特定历史时期的产物，用来方便我们编写动态网页，简化Servlet的编写（现有Servlet技术，所有html语句在java中print太复杂，后来才发展的JSP）。但是由于JSP功能丰富，可以直接调用任何java代码，如数据库操作等，会带来一个严重的问题，即**容易将业务逻辑代码（Model层，大部分是java代码）混杂在View层中（html，css，js）**，导致高度的耦合。因此现在网页开发框架中，依据Model2模型，往往**使用模板技术代替JSP**，模板仅仅有填值的功能，使用model层传过来的值渲染界面。

当然，我们也可以使用JSP实现模板功能，并且使用扩展的标签（JSTL）来取代只有Java才能实现的逻辑判断（如if while）。这里谈一下JSP使用的最佳实践：

* 不要使用：
  * **在JSP文件中含有任何Java代码。**
  * 不使用JSP中的Scriptlet（如`<%..%>`,`<%=..%>`）
* 使用：
  * 使用EL表达式，填值。
  * 使用JSTL来做逻辑判断
  * 可以使用JSP原生的一些指令，如`<%@ taglib ... %>`` <%@ page ... %>`
  * 适当的使用一些JSP的行为，jsp:useBean jsp:setProperty jsp:getProperty

### 基本语法

- ~~引入Java代码块~~：` <% 代码片段 %>`
  - if
    - while、for
- ~~JSP声明~~：可以声明一个或多个变量、方法，供后面的Java代码使用。 `<%! declaration; [ declaration; ]+ ... %>`
- ~~JSP表达式~~ 模板填值：包含的脚本语言表达式，先被转化成String，然后插入到表达式出现的地方`<%= 表达式 %>`
- jsp指令：` <%@ directive attribute="value" %>` 
  -  page 定义页面的依赖属性，比如脚本语言、error页面、缓存需求等等
  -  include 包含其他文件
  -  tablib **引入标签库的定义，可以是自定义标签**
- jsp行为
  - jsp:useBean 寻找和初始化一个JavaBean组件
  - jsp:setProperty 设置JavaBean组件的值 对应setXxx
  - jsp:getProperty 获取JavaBean组件的值 对应getXxx
  - jsp:include 用于在当前页面中包含静态或动态资源
  - 其他
- 隐含对象：request response out session pageContext等

### EL

用来取代JSP的表达式功能（`<%=..%>`），可以用来进行属性、请求参数、cookies的获取，以及一些简单运算判断。

* 语法参考：http://www.runoob.com/jsp/jsp-expression-language.html
* 最常用：${expr}
* 操作符：
  * . 访问一个Bean属性或者一个映射条目
  * [] 访问一个数组或者链表的元素
  * `+` `-`  `*` `/(div)`
  * ==(eq) !=(ne) <(lt) >(gt) <=(le) >=(ge)
  * empty 判空
* 调用自定义标签中的函数 `${ns:func(param1, param2, ...)}`
* 内置对象
  * param、paramValues：访问参数值。如`${param.order}`，或者`${param["order"]}`。
  * pageContext
  * cookie
  * 。。。。

### JSTL

最基础的功能是提供逻辑判断的标签，可以彻底取代jsp文件中的java代码。当然它还提供了通过标签加入各种功能。但是不建议滥用。

* 具体语法：http://www.runoob.com/jsp/jsp-jstl.html
* 导入：如导入core库，` <%@ taglib prefix="c"uri="http://java.sun.com/jsp/jstl/core" %>`
* 5大标签
  * core 核心表情库，提供条件判断，属性存取，url处理，错误处理等。
    * [`<c:if>`](http://www.runoob.com/jsp/jstl-core-if-tag.html)
    * [`<c:when>`](http://www.runoob.com/jsp/jstl-core-choose-tag.html)
    * [`<c:forEach>`](http://www.runoob.com/jsp/jstl-core-foreach-tag.html)
  * fmt 格式化 日期格式，国际化功能等
  * sql 与关系型数据库（Oracle，MySQL，SQL Server等等）进行交互的标签
  * xml 创建和操作XML文档的标签
  * functions jstl函数，JSTL包含一系列标准函数，大部分是通用的字符串处理函数。
    * [fn:substring()](http://www.runoob.com/jsp/jstl-function-substring.html)
    * [fn:contains()](http://www.runoob.com/jsp/jstl-function-contains.html)
    * [fn:length()](http://www.runoob.com/jsp/jstl-function-length.html)
    * [fn:split()](http://www.runoob.com/jsp/jstl-function-split.html)

