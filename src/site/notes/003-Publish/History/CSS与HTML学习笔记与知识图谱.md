---
{"dg-publish":true,"permalink":"/003-Publish/History/CSS与HTML学习笔记与知识图谱/","title":"CSS与HTML学习笔记与知识图谱","tags":["技术","全栈","html","css","web开发"],"created":"2024-12-11T18:09:35.713+08:00","updated":"2024-12-11T18:09:35.714+08:00"}
---



## 概述

* css 与html的职责
  * html负责内容
  * css负责**布局**和修饰（背景等）

>  所以不推荐用html布局（如表格布局）和设置颜色背景等

* 学习内容
  * w3cschool教程
  * [css](http://zh.learnlayout.com/toc.html)

## 知识图谱

[下载mindnode图谱](/files/HTML与CSS知识图谱.mindnode)

## 浏览器常识
* URL 编码
  * scheme://host.domain:port/path/filename  www是host，默认主机名
  * URL 只能使用 ASCII 字符集来通过因特网进行发送。
  * 由于 URL 常常会包含 ASCII 集合之外的字符，URL 必须转换为有效的 ASCII 格式。
  * URL 编码使用 "%" 其后跟随两位的十六进制数来替换非 ASCII 字符。
  * URL 不能包含空格。URL 编码通常使用 + 来替换空格。

* cookies的主要作用：server用来传送数据，保持状态的，不是client用的。

* 登录问题

  * 登录：认证方式
    可以对比一下三种认证方式的区别，这也是很基础的事情了 

    1. basic auth 
       没有安全性可言，完全依赖 https，并且每次请求都需要传递密码，是最不安全的 

    2. session 
       一般需要 basic auth 或其它 auth 方式先进行身份认证后才能建立，和 basic auth 一样没有什么安全性可言，需要 https 保障，只不过避免了每次传递密码（忽略服务器端状态这点特性） 
       **服务端更具sessionId缓存信息**

    3. oauth （jwt的原理？）
       部分是为了解决 basic auth 的安全问题而设计的，但是要复杂很多，即使没有https也能保证基本的安全，数据包被监听后不能防止信息泄露，但可以防止信息伪造，包括重放攻击。**用于无状态的restful API设计？原来session的缓存放在客户端本地！**

    4. jwt
       http://www.jianshu.com/p/576dbf44b2ae

    http://www.cnblogs.com/xiekeli/p/5607107.html

    待研究http://blog.rainy.im/2015/06/10/react-jwt-pretty-good-practice/




## HTML

* 参考思维导图学习


* [html头部meta的作用](http://www.w3school.com.cn/html/html_head.asp)
  * 设置中午，支持编码 `<meta charset="UTF-8">`
  * 一些描述
    * `<meta name="keywords" content="HTML, CSS, XML" />`
    * `<meta name="description" content="Free Web tutorials on HTML, CSS, XML" />`
  * style，内置css
  * link引入css ` <link rel="stylesheet" type="text/css" href="./style.css" />`
  * js 引入 `<script src="//code.jquery.com/jquery-1.11.3.min.js"></script>`
* HTML 实体
  &entity_name;或者&#entity_number;

### HTML5:
* video autdio location
* drag
* canvas svg
* localstorage/sessionstorage vs cookies
* app cache
* work thread
* [server send event vs websocket]( http://stackoverflow.com/questions/5195452/websockets-vs-server-sent-events-eventsource)
* 一些表单form的属性，帮助写表单

  ​



## CSS

参考这个[css](http://zh.learnlayout.com/toc.html)

* CSS 对大小写不敏感,但是class 和 id 名称对大小写是敏感的
* 根据 CSS，子元素从父元素继承属性（body里面的元素自动使用body的样式）。但可能有兼容问题。此外，背景属性都不能继承。
* 引入css的三种方式和优先级
* CSS选择器
  * 区分几种情况：空格，逗号，连着写
    *  `div #id` `p .div` `.classone .classtwo`
    *  `div#id`  `p.div`  `.classone.classtwo`
    *  `#id div` `.div p`
    *  `div,#id` `p,.div`
    *  `#id,div` `.div,p`
  * 一个标签有多个class `<p class="important warning">hello</p>`

  * 上下文选择器（派生选择器）：依据元素在其位置的上下文关系来定义样式,**层次间隔可以是无限的**。必须用`空格`分割，例子： 
    * `li strong{}` //li中的strong标签
  * id选择器
    * `#myid{}  ` 
    * `#sidebar p {}` //上下文选择器+id选择器
    * `div#sidebar{}` //**div且id为xx的**,区分`div #sidebar{}` div内部有#sidebar的
  * 类选择器，类名的第一个字符不能使用数字
    * `.className`
    * `.fancy td{}` //上下文选择器+类选择器
    * `td.fancy{}`//**td且是facncy类**,区分`td .fancy{}`
  * 属性选择器：用于选择那些没有id也没有class的标签
    * `[title]{}` 有title的元素
    * `[title=hello]{}`有title值仅等于hello的元素
    * `[title~=hello]{}`有title值等于hello的元素，可以有空格分割如`<div title="hello world"><div>`
    * `[title|=hello]{}`有title值等于hello的元素，可以有-分割如`<div title="hello-world"><div>`
    * 与其他选择器结合
  * 类选择器还是 ID 选择器？
    * 只能在文档中使用一次
    * 不能使用 ID 词列表,既一个元素不能有多个id
    * ID 能包含更多含义，指定业务含义
  * **伪类**
    * `selector : pseudo-class {property: value}`
    * `selector.class : pseudo-class {property: value}`
    * `a:visited {color: #00FF00}`
    * `p:first-child {font-weight: bold;}` **选择p作为第一个child时的那个p，不是选择p的第一个child**
    * `p > i:first-child {font-weight:bold;} `选择作为p的第一个直接子元素的i。**从做到右来理解表达式!**
    * :lang 伪类使你有能力为不同的语言定义特殊的规则
  * 伪元素
    * `selector:pseudo-element {property:value;}`
    * `selector.class:pseudo-element {property:value;}`
    * 与伪类区分，它园中的是冒号后面的
    * p:first-line{color:#ff0000;} 选择p的第一行文字
    * :first-letter
    * h1:before{content:url(logo.gif);} 在h1前插入内容，类似的:after
  * 子元素选择器：不是选择任意的后代元素，而是只选择某个元素的子元素。
    *  `h1 > strong`
  * 相邻兄弟选择器：可选择紧接在另一元素**后的元素**，且二者有相同父元素。
    * `h1 + p {margin-top:50px;}`：选择紧接在 h1 元素后出现的段落p。

    ​


* CSS样式 一些惯例用法
  * 设置body的字体与字号,em单位的使用
  ```html
  body {font-size:100%;}
  h1 {font-size:3.75em;}
  h2 {font-size:2.5em;}
  p {font-size:0.875em;}
  ```
  * 链接相关的选择器
    * a:link - 普通的、未被访问的链接
    * a:visited - 用户已访问的链接
    * a:hover - 鼠标指针位于链接的上方
    * a:active - 链接被点击的时刻

* CSS框模型：padding,margin，content是什么
  * 元素组成:内容，内边距（padding），边框(border)，外边距(padding)
  * 与android不同的是**padding和边框不算在内容区域width/height中**(IE5,6是这样的）
  * 背景应用在在边框内（content+padding），下面红色虚线
    ![](http://www.w3school.com.cn/i/ct_css_boxmodel_example.gif)
  * 外边距可以是负值，而且在很多情况下都要使用负值的外边距。
  * 由于 border-style 的默认值是 none，如果希望边框出现，就必须声明一个边框样式。
  * 一个奇怪的性质：外边距合并，仅仅在垂直方向

* [CSS布局与定位](http://www.w3school.com.cn/css/css_positioning.asp)
  * 一切皆为框：块框(display:block，如p)，行内框(display:inline，如span)，无名框，行框（如，文本的一行）
    * 一行形成的水平框称为行框（Line Box），行框的高度总是足以容纳它包含的所有行内框，设置行高可以增加这个框的高度。
      * 行内框会导致一些css属性失效，如高度
  * 三种定位方式：
    * 普通流（默认）：position:static，不会重叠
    * 绝对定位：原来的占位不保留，改变结构。定位后可能重叠。
      * position:absolute 相对于**最近的已定位祖先元素**（不是static的父亲，一直到body）
      * position:fix 相对于整个视窗，效果：滚动页面定位不动。
    * 相对定位：position:relative就是static加上偏移值，原来的占位还保留，不改变结构。偏移之后可能重叠
    * 浮动：最常用的手段，左右浮动，文字框重叠，文字不会重叠，文字环绕
      * float
      * clear
      * 一个demo http://www.w3school.com.cn/tiy/t.asp?f=csse_float6

* CSS3新特性

  * flex布局：http://www.ruanyifeng.com/blog/2015/07/flex-grammar.html
    * Flex布局以后，子元素的`float`、`clear`和`vertical-align`属性将失效。

* 坑：
  * css中忘了加分号，导致属性不起作用

