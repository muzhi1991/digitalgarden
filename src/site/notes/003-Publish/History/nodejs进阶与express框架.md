---
{"dg-publish":true,"dg-path":"History/nodejs进阶与express框架.md","permalink":"/History/nodejs进阶与express框架/","title":"Node.js进阶与Express框架及知识图谱","tags":["技术","全栈","javascript","nodejs","后台开发"],"created":"2024-12-11T18:09:35.735+08:00","updated":"2024-12-11T18:09:35.736+08:00"}
---



本篇介绍后台开发的常用技术与概念，以Node.js的Express框架为例，Express是Node.js上最流行的开发框架，十分轻量，提供了一套方便开发的工具方法。当前使用最多的是Express4.0版本。

> Express太轻，叫他框架都有点不合适。个人认为框架的特点是我们编写的代码被框架调用，而Express只是提供了一套便捷的API与相关中间件。

学习资料：

* [Express教程](http://javascript.ruanyifeng.com/nodejs/express.html)
* [nodejs实践，包教不包会](https://github.com/alsotang/node-lessons)
* [官网](http://expressjs.com/zh-cn/)：安全最佳实践和性能最佳实践两篇文章不错


知识图谱：[nodejs进阶与Express知识图谱](/files/nodejs进阶与express知识图谱.mindnode)，尽量在其他技术上也可以**通用**。


## 基本使用

### 生成项目--express-generator

该工具会自动生成Express工程目录，安装mvc的结构新建一些目录

```javascript
 npm install express-generator -g
 express myapp
 
 .
├── app.js // 主文件，设置中间件间，被www调用
├── bin
│   └── www // 启动脚本，建立了一个server，监听了端口，不用这个启动就自己写在app.js中，更清晰。
├── package.json
├── public // 静态资源
│   ├── images
│   ├── javascripts
│   └── stylesheets
│       └── style.css
├── routes // 路由，设置了一堆get post方法的路由
│   ├── index.js
│   └── users.js
└── views // html或者模板文件，这里用的jade
    ├── error.jade
    ├── index.jade
    └── layout.jade
```

### 启动服务

```javascript
var express = require('express');
var app = express();
// 一个内置的中间件，处理静态资源
app.use(express.static(__dirname + '/public'));
// 这里设置其他中间件。。
// 这里设置路由。。
// 监听端口
app.listen(8080);
```

### 路由设置

本质也是中间件，安装路径分发请求

```javascript
app.get('/', function (req, res) {
  res.send('Hello World!');
});
```

### 响应请求

这里的req和res和Nodejs原生的**相同**，只是在我们添加了一些中间件之后，req中会多一些变量，帮助我们处理请求。可以在不涉及 Express 的情况下调用 `req.pipe()`、`req.on('data', callback)` 和要执行的其他任何函数，如`res.end()`等。

Express添加了一些便捷方法

* [请求req](http://expressjs.com/zh-cn/api.html#req)中：
  * `req.params`：路由中定义的参数
  * `req.query`：问号后面参数
  * `req.body`：
  * `req.cookies`
* [返回res](http://expressjs.com/zh-cn/api.html#res)中：
  * `res.status(200)`设置header中的状态码的便捷方法。
  * `res.send("xxx");`返回json，text结果，比较常用，自动设置Content-Type，内部包含end。[参考](http://expressjs.com/zh-cn/api.html#res.send)
  * `res.render();`:渲染视图，返回html，内部包含end。
  * `res.sendfile()`:依据文件扩展名设置Content-Type。如直接返回网页。内部包含end。

### 一般中间件

中间件会被链式调用，处理请求

```javascript
// 定义中间件
var myLogger = function (req, res, next) {
  console.log('LOGGED');
  next(); // 如果不调用，表示结束，不会启动下个中间件
};
// next 带参数表示错误
function uselessMiddleware(req, res, next) {
  next('出错了！');
}
function uselessMiddleware(req, res, next) {
  res.end("ending...");
}
// 使用中间件
app.use(someMiddleware);
app.use('/path', someMiddleware);
// 其他
app.all("*",someMiddleware);// 所有请求都会调用
app.get("*",someMiddleware);// get请求会调用，Express还提供post、put、delete方法
// 除了绝对匹配以外，Express允许模式匹配，参加文档

// 使用第三方库
var bodyParser = require('body-parser'); 
app.use(bodyParser.json({limit: '1mb'}));
```

### 错误处理

在其他 `app.use()` 和路由调用之后，**最后定义**错误处理中间件

```javascript
// 与其他中间件不同，它有四个参数，第一个是error
app.use(function(err, req, res, next) {
  console.error(err.stack);
  res.status(500).send('Something broke!');
});
// 可以定义多个错误处理中间件
```

### 模板引擎的使用

- `views`：模板文件所在目录。例如：`app.set('views', './views')`
- `view engine`：要使用的模板引擎。例如：`app.set('view engine', 'jade')`

```javascript
// 默认这个目录下面
// app.set('views', './views')
// 设置这个之后，下面的render不用在.jade的后缀了。同理，也可以设置html
app.set('view engine', 'jade');

app.get('/', function (req, res) {
  res.render('index', { title: 'Hey', message: 'Hello there!'});
});
```

> jade是一种格式的模板引擎，注意是服务端渲染使用，它的语法与html不同，感觉不如ejs，还是保留原始的html比较好。

## 核心概念

### 中间件

这个概念在后台十分广泛，广义上的中间件是任何在系统中通用的组件，比如在分布式系统中的消息队列，缓存系统，配置管理系统，soa治理系统，都可以称之为中间件。这里的中间件也有通用的含义，如对http的结果的预处理（body解析，路由）。

nodejs中的中间件是使用的connect库，深入学习https://github.com/alsotang/node-lessons/tree/master/lesson18

### 路由

通俗的说就是对请求地址的分发，按照地址路由给某个方法来处理。因此需要一些正则表达式的知识。

多个函数处理一个路由

```
app.get('/example/b', function (req, res, next) {
  console.log('the response will be sent by the next function ...');
  next();
}, function (req, res) {
  res.send('Hello from B!');
});
```

链式处理的便捷方法 app.route('/book')

```javascript
app.route('/book')
  .get(function(req, res) { // 对book的get方法处理
    res.send('Get a random book');
  })
  .post(function(req, res) { // 对book的post方法处理
    res.send('Add a book');
  })
  .put(function(req, res) { // 对book的put方法处理
    res.send('Update the book');
  });
```

Express的Router方法可以模块化的处理路由

```javascript
var express = require('express');
var router = express.Router();
// middleware that is specific to this router
router.use(function timeLog(req, res, next) {
  console.log('Time: ', Date.now());
  next();
});
// define the home page route
router.get('/', function(req, res) {
  res.send('Birds home page');
});
// define the about route
router.get('/about', function(req, res) {
  res.send('About birds');
});
module.exports = router;

//////////////////////其他文件///////////////////////
var birds = require('./birds');
app.use('/birds', birds); // 对/birds/xxx的路由
```

### 模板引擎与render

渲染一个常用的概念，主要用来对前端界面（html）的填值处理。渲染分为**服务端渲染**和**客户端渲染**。

* 服务端：有服务器处理模板文件，生成填值后的html文件，返回给客户端。优点是客户端一次请求就是结果，渲染工作在服务端，客户端性能较好。缺点也很明显，这些值只能一次更新的，**如果要刷新，必须刷新整个页面**。
* 客户端：客户端请求的就是含义模板信息的html文件，使用ajax请求接口返回动态信息（类似app）。然后调用模块引擎库（客户端js工具方法）填值。有点是可以不刷新页面更新值，缺点是必须发起多此请求，需要做好过渡效果处理。
* 两种结合：**只有载入时更新的值用服务端渲染，动态更新的值用客户端渲染。**

模板引擎是渲染填值的工具，如jade主要用于服务端，ejs两端通用，可以**深入学习ejs的使用方法，和它的一些概念如layout模板**等。

### 异步控制方法与回调地狱

* promise
* 事件订阅与发布--eventproxy
* 异步流程控制--async：要求用户传入待执行的函数列表，记为funlist。流程控制库的任务是让这些函数 **顺序执行** 。

## 实践

下面的总结有些乱，可以参考知识图谱梳理。

### 登录

#### 授权登录oAuth2.0（第三方登录）

* 理解什么是：
  * oAuth流程，为什么是两步？http://www.tianmaying.com/tutorial/oAuth-login
  * 各种oauth授权方式汇总：http://www.ruanyifeng.com/blog/2014/05/oauth_2_0.html
* 理解oauth中的token：见jwt

#### [jwt](http://blog.rainy.im/2015/06/10/react-jwt-pretty-good-practice/)

* [理解token](https://www.zhihu.com/question/20274730)
  * token本身只是某些信息的base64+rsa的验签（不防查看，防止修改）
  * 为了防止token泄露，传输需要加密（https）
* 区别token和sessionid，token包含信息（可查看，不可修改，思考应该包括什么，不应该包括什么，回放攻击），session就是会话id，信息在内存中，因此占内存。使用范围也不同，**一个无状态（只负责授权）一个有状态（授权和关联会话信息）。**
  - http://security.stackexchange.com/questions/81756/session-authentication-vs-token-authentication

#### 思考

* 扩展：这是一个有意思的问题，是web与app跨界的地方，以前web（browser）是server为主的构架（可能是因为js不好用，browser只负责展示），所以用sessionId这种东西。现在App或者h5（RESTful），用更多主动权与业务逻辑，所有token这种客户端为主的技术广泛应用。xxx为主是谁维护状态，谁写业务逻辑。
* 防止xxx泄露：指的是防止传输过程泄露，客户端泄露无能为力。
  * 防止token泄露，参考oauth2.0的第二次请求（https传输）
  * 防止sessionId泄露，使用https

### MemCache Redis

memcache、redis为什么是必须的
由于应用程序实例作为单独的进程运行，因此它们不会**共享同一内存空间**。也就是说，对象位于应用程序每个实例的本地。因此，无法在应用程序代码中保存状态。然而，可以使用 Redis 之类的内存中数据存储器来存储与会话相关的数据和状态。此警告适用于几乎所有形式的水平扩展（无论是多个进程的集群还是多个物理服务器的集群）
对于负载均衡功能，可能必须确保与特定会话标识关联的请求连接到产生请求的进程。这称为会话亲缘关系或者粘性会话，可通过以上的建议来做到这一点：将 Redis 之类的数据存储器用于会话数据（取决于您的应用程序）。

### 数据库

mongodb hbase mysql 常见的数据库和应用场景
mongodb：不支持表 join，事务，schema-less(应用场景：log）
mongodb 和 mysql 要我选的话，无关紧要的应用我会选择 mongodb，就当个简单的存 json 数据的数据库来用；如果是线上应用，肯定还是会选择 mysql。毕竟 sql 比较成熟，而且各种常用场景的最佳实践都有先例了。

* 数据库存储方式：[行存储与列存储](https://www.ibm.com/developerworks/community/blogs/IBMi/entry/database?lang=en)
* 数据库分片：sharding

### 多进程与进程管理

nodejs是单线程的，一个进程中只有一个主线程，因此无法发挥多核CPU的优势。解决方法是**启动多个nodejs实例**，即一个程序启动多个相同的进程。这就需要一个中介来**分发请求**和**管理进程的生命周期**。常用的管理方式有

* 最原始：使用nodejs的cluster模块写代码自己管理
* 第三方工具
  * PM2
  * [StrongLoop Process Manager](http://expressjs.com/zh-cn/advanced/pm.html#sl) 供全面的运行时和部署解决方案的唯一工具？

多进程带来的问题，如何进程间通讯，共享内存变量（如session），这就引入了redis或者memcache这些分布式缓存。

> 一个典型的场景：登录请求第一次被分配到A实例，缓存了一些值（如sessionId和对应的值，如登录成功），第二次请求被分配到B实例，由于它没有缓存的值，sessionId找不到，又被判定为未登录

### 测试

* 开发测试--测试驱动开发
  * 服务端js mocha，should，istanbul（测试覆盖率）
  * 客户端js mocha，chai，phantomjs
* 性能测试 benchmark  
  * benchmark.js 库

### 部署

* 线上部署，一些pass云平台部署应用
* 持续集成travis—代码提交后自动运行测试和构建，并生成报告

### [cookie与session](https://github.com/alsotang/node-lessons/tree/master/lesson16)

HTTP 是一个无状态协议，所以客户端每次发出请求时，Server下一次请求无法得知上一次请求所包含的状态数据。两种方案解决这个问题。（重点：**Server需要这些信息，来逻辑处理**）

* cookies 在请求header中暂存这些信息，传给Client后，**原封不动**的回传给Server。（以前一直认为cookie是客户端用的）达到记录状态的目的
  * 优势：方便，逻辑简单
  * 问题：1.值被篡改。2.关键数据来回传送不安全。3.非流量，占网速。4.回放攻击问题。
* session cookie中只传输一个ID，类似于临时用户的概念，代表了与之相关的一组信息。
  * 优势：相对安全，不用来回传关键数据，流量少，无回放攻击
  * 问题：sessionID泄露，还是会泄露信息。

> 这两种技术都是在服务端解决http无状态的问题的方案，且主要用在浏览器中（有cookies实现），其思想可以通用，
>
> 另外一种是Token授权，主要是客户端的授权。这个技术往往用在由客户端维持状态，服务端仅仅提供无状态的服务的场景。（客户端凭token获取服务）

* sessionId最重要特征：不可修改性！，最担心的问题session泄露（劫持）
* signedCookies：对cookie进行加盐算hash，那么cookies可查看但是防止修改。

### 安全

* 常见的攻击：
  * 回放攻击
  * [csrf攻击](http://www.cnblogs.com/hyddd/archive/2009/04/09/1432744.html)：跨站请求伪造（Cross-site request forgery）
  * [XSS](http://blog.csdn.net/ghsau/article/details/17027893)：跨站脚本（Cross-site scripting），在网页输入中嵌入脚本
    * dom based
    * stored xss：发帖存储相关代码
  * 中间人攻击：实现会话劫持
* 攻击的基本思想
  * 利用cookies：
    * 非法获取：xss可以读取cookies，获取用户信息
    * 模拟某个请求浏览器会自动带上cookies（csrf）
  * 在外部（恶意）网站模拟被攻击网站
    * 虽然浏览器禁止跨域访问，但是还是可以通过其他方法访问
      * `<img/>`的src
      * `<scropt>`的src
* 预防
  * 保护好cookies
    * 使用http-only
  * 防止非法代码注入xss
    * 对输入校验
    * 统一对输入encode，防止嵌入`</script>`
  * 防止csrf
    * 对请求来源进行验证—下发表单时加入hide input包含csrf token，确保以后提交的表单是自己下发的，不是伪造的。
  * 中间人攻击：https
