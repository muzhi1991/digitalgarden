---
{"dg-publish":true,"dg-path":"History/nodejs&&express实践-nodeclub项目分析.md","permalink":"/History/nodejs&&express实践-nodeclub项目分析/","title":"Node.js&&Express实践-nodeclub项目分析","tags":["技术","全栈","javascript","nodejs","实践","后台开发"],"created":"2024-12-11T18:09:35.742+08:00","updated":"2024-12-11T18:09:35.743+08:00"}
---




简单分析一个真实的项目-[nodeclub](https://github.com/cnodejs/nodeclub)，这个项目是node社区的源码，可以看做是一个用Node.js(Express框架)实现的社区论坛的模板。建议参考下面的**技术栈目录**和[上一篇文章中的知识图谱](/files/nodejs进阶与express知识图谱.mindnode)学习。

包含的功能：

1. 登录&&注册&&验证
   1. 通过第三方github信息注册
   2. 直接注册
2. 账户系统
3. 发帖，评论
   1. 图片上传
   2. markdown
4. 站内搜索
5. 日志、性能、监控
6. 安全
7. 其他功能
   1. rss

使用的技术栈：

1. Node.js&&express
2. 代码设计结构--mvc
   1. m
      1. 数据持久化
         1. 数据库方案：mongodb文档数据库
         2. orm框架：`mongoose`模块
         3. cookies：`cookie-parser`模块
      2. 内存数据存储
         1. session&&redis—`express-session`,`connect-redis`模块
   2. v
      1. bootstrap框架
      2. 渲染方式&&模板引擎[`ejs-mate`模块](http://yijiebuyi.com/blog/08cf14e904325c19814465689453b3aa.html)
      3. app.locals res.locals传值
      4. loader加载css，js，less
      5. 混合使用css与less配置样式
   3. c
      1. 路由方案—`express.Router();`
      2. 代码流程控制，eventproxy，解决回调地狱。
3. 工具lib
   1. html解析中间件 body-parse
   2. 解决js回调问题 — `eventproxy`模块
   3. **http请求模拟**，`superagent`模块：测试api，代理，爬虫等
   4. 上传文件 `busboy`
   5. [uuid生成工具](https://github.com/broofa/node-uuid)—uuid的作用与场景
   6. [nodejs的后端字符串验证器-validator](http://blog.csdn.net/zzwwjjdj1/article/details/52042194)
   7. url解析库—nodejs自带的
   8. [method-override](http://blog.csdn.net/boyzhoulin/article/details/40146197)
   9. moment模块：处理时间相关操作。
4. 登录解决方案，token技术
   1.  `passport`  node的登录认证中间件，支持用户名密码登录或者oauth登录
   2.  `passport-github` passport的github插件，实现github的登录策略
5. 安全
   1. 全站https
   2. 加解密
      1. bcrypt模块
      2. 数据库不存明文，只存加盐后的值或者token
      3. cookies加盐签名,sessionId加盐签名
   3. csrf问题：`csurf`模块
   4. 输入校验 
   5. [helmet模块](https://segmentfault.com/a/1190000003860400)
6. 性能优化
   1. 上线前开启production模式
   2. 使用gzip
   3. js，css文件压缩—Loader模块
   4. 视图缓存（production会启用）：app.set('view cache', true);
   5. api接口限流—`limit.js`自定义的中间件
7. 监控
   1. 日志系统`log4js`库
   2. 输出请求参数，ip，耗时
   3. 错误日志输入
   4. 第三方性能监控工具：oneapm
8. 其他
   1. 支持api json接口的跨域访问—`cors`模块,
   2. 反向代理（nginx）背后 —`app.enable('trust proxy');`


## 项目分析

### 目录结构

- main主目录下
  - app.js 项目入口文件
  - xxroute.js 路由，web页面的路由和api路由
  - config.js 项目用到的配置，如各种密钥，地址，访问频次等
  - .jshintrc jshint工具配置
  - package.json 
  - Makefile 定义一些常用任务，如make test，make build生成压缩js
- common：一些工具方法的封装
  - cache.js：缓存封装，redis.js的再封装
  - logger.js：日志log4js的封装
  - redis.js：ioredis模块的封装，形成redis的client
  - mail.js：发送邮件功能的封装
  - tools.js 其他工具方法封装，格式化时间、密码存储比较等
  - message.js 发送消息的功能，操作数据库
  - at.js 提供at的功能，发送at消息，操作数据库
  - store.js 保存文件到本地、七牛
  - render_helper.js：一些在html中使用的工具方法的封装
- middlewares：中间件，处理一些请求会用到的功能。
  - auth.js 权限校验，含有登录判断，管理员判断的中间件
  - error_page.js 错误页的统一界面
  - limit.js 频率限制的中间件，如api请求次数等
  - proxy.js 在站内提供了一个代理，访问localhost/agent?url="www.google-analytics.com"即可。
  - requst_log.js 打印所有的http请求的日志，如完成时间
  - mongoose_log.js dubug环境下打印mongoose操作日志。。不是中间件，就一个配置
  - render.js  debug环境下偷偷替换（装饰）res的了render函数的中间件
- controllers：控制器，所有的路由都会传递给controller。由它去操作model（通过proxy）调用工具方法处理数据，最后render view输出。因此controller比较重。按功能模块分为不同js，每个js中暴露多个控制器。
  - site.js: 网站主页相关的的请求控制器，如主页
  - user.js：账户相关的请求控制器，如个人主页，用户设置。
  - topic.js 话题相关的，如新建帖子，删除帖子，置顶等。
  - message.js 消息相关，如用户个人的所有消息。
  - 其他：。。。
- views：界面html，ejs模板写的，目录下按功能模块分包，略了
- proxy：**model层的操作封装**，可能涉及跨多个model的查找。如根据用户id查找帖子。按功能模块分为不同js。
  - index.js 其他所有proxy的集合，都放到它的`exports`变量中
  - user.js 用户相关操作，如新建用户，查找用户
  - topic.js 帖子相关操作，如查找某个帖子的相关信息，如发布的用户，内如（要查多个model）。
  - 其他。。
- models：**纯的model实体定义，数据库实体，类似POJO**，不含操作。
  - index.js 其他所有model的集合，都放到它的`exports`变量中
  - base_model.js 给所有model添加方法。
  - user.js 用户实体，如用户名，密码，token
  - topic.js 帖子的实体，如标题，时间，浏览数目，作者。
  - 其他。。
- api：RestfulAPI，由api的router直接调用这里。
- test

### 调用流程

![nodeclub数据流动](http://fodi.limuzhi.us.kg/images/IMG-3c272c682a163cf5.webp)

注意一下，操作model这里有点乱，很多地方都操作了，如proxy，common中的一写工具，controllers下的控制器直接操作model。api目录下的RestfulAPI也有直接操作的。理想情况下：所有的model操作都在proxy中。controller api common中都引用proxy操作。问题是：比较麻烦，很多简单操作要多封装一层。




## View层界面相关技术

app.locals这个对象字面量中定义的键值对，是可以直接在模板中使用的，就和res.render时开发者传入的模板渲染参数一样

### 模板的值传递

* render参数，传入值，等价于res.locals赋值
* app.locals与res.locals

>  `locals`可能存在于`app`对象中即：[`app.locals`](http://itbilu.com/nodejs/npm/VJ5TlyRnl.html#app-properties-locals)；也可能存在于`res`对象中，即：[`res.locals`](http://itbilu.com/nodejs/npm/Vkp32gJpg.html#res-prop-locals)。两者都会将该对象传递至所渲染的页面中。不同的是，`app.locals`会在整个生命周期中起作用；而`res.locals`只会有当前请求中起作用。由于`app.locals`在当前应用所有的渲染模中访问，这样我们就可以在该对象中定义一些顶级/全局的数据，并在渲染模板中使用。

### ejs

* [ejs 标签](http://www.voidcn.com/blog/hzw05103020/article/p-2337629.html)(参考layout.html文件)

  *  <%...%> 块中安排JavaScript 代码

  *  <%=输出变量%>

  *  <%- VARIABLE_NAME %>：输出原始内容,不会被escape，应用在导入html，json对象

  *  <%-include filename %>加载其他页面模版

  *  自定义开闭符号(<% %> )

     > =号输出,就会被escape转义编码（在JavaScript中，escape(s)是一个全局函数，其对字符串s某些字符-------替换成了十六进制的转义序列。在escape(s)返回的新字符串中，除了ASCII字母、数字、标点符号*+-./@_外，所有字符都转义成%xx或%uxxxx（x是十六进制数）形式，其中，从%u0000到%u00ff的Unicode字符转义成%xx形式。）

* 支持母模板 ejs-mate：一个ejs的分支版本，支持母模板

  * 什么是母模板，[介绍](http://yijiebuyi.com/blog/08cf14e904325c19814465689453b3aa.html)：某网站中的所有网页的header（js，css）与footer一般是相同的，不同的部分是body中的内容。
  * layout.html中 公用的模板布局
    * head
      * 一些库的css,js加载，如bootstrap，jquery
      * layout中公用布局的css，js
      * meta元素配置：description等，比较重要的有[csrf存在meta中](https://cnodejs.org/topic/5533dd6e9138f09b629674fd)，[referrer设置](https://imququ.com/post/sth-about-switch-to-https-3.html)。
      * title，用js选择合适的title
      * 载入一下配置文件中的header
      * link：[icon](http://www.cnblogs.com/LoveJenny/archive/2012/05/22/2512683.html)和rss
    * footer
      * 赞助信息等
      * ga，网盟配置

  ```html
  <!DOCTYPE html>
  <html xmlns="http://www.w3.org/1999/xhtml">
  <head>
    <!-- style  包括公共库的css，公共布局layout用的css-->
    <%- Loader('/public/stylesheets/index.min.css')
    .css('/public/libs/bootstrap/css/bootstrap.css')
    .css('/public/stylesheets/common.css')
    .css('/public/stylesheets/style.less')
  	.........
    .done(assets, config.site_static_host, config.mini_assets)
    %>

    <!-- scripts 包括公共库的js，公共部分layout使用main.js，responsive.js-->
    <%- Loader('/public/index.min.js')
    .js('/public/libs/jquery-2.1.0.js')
    .js('/public/libs/lodash.compat.js')
    .js('/public/libs/jquery-ujs.js')
     ..........
    .js('/public/javascripts/main.js')
    .js('/public/javascripts/responsive.js')
    .done(assets, config.site_static_host, config.mini_assets)
    %>
    <body>
    <!--公共布局如标题栏，侧边栏--->
    <!--。。。。。。--->
      
    <div id='main'>
    <!--被替换部分--->
    <%- body %>
    </div>
      
     <!--公共布局如footer--->
     <!--。。。。。。--->
      
     <script>
     // 这个页面的一些脚本
     </script>
    </body>
    </html>
    
  ```

>  直接使用ejs的include支持模板复用：https://cnodejs.org/topic/50c1a0ed637ffa4155d05256

### bootstrap前端样式框架

- 什么是响应式布局：简单来说可以适应不同大小屏幕的布局
- 原理：
  - [viewport](http://www.cnblogs.com/2050/p/3877280.html)
  - CSS 媒体查询
  - [断点](https://developers.google.com/web/fundamentals/design-and-ui/responsive/fundamentals/how-to-choose-breakpoints?hl=zh-cn)
- [常见样式](https://developers.google.com/web/fundamentals/design-and-ui/responsive/patterns/?hl=zh-cn)
- 什么是bootstrap：一个**前端样式框架**。仅仅是方便布局和提供一些样式。可以理解为一个增强的CSS的框架（内部使用less），同时支持[自定义](http://v3.bootcss.com/customize/)。
  - 更方便布局：**12栅格系统**
  - 一些css样式：如一下按钮、表格等样式，通过class属性引用。
  - 一些js组件：如点击按钮下拉，弹窗等常用的操作。通过自定义的属性如`data-toggle`引用。
- [入门](http://www.cnblogs.com/zhili/p/BoostrapQuickStart.html)

### css与less样式表

[Less](http://www.bootcss.com/p/lesscss/)是CSS样式表的一个进化版本，支持变量的定义。需要注意的是less最终会转换成css文件:

* 客户端解析：在客户端解析less为css
* 服务端解析：请求的less文件，服务端解析此文件为css后返回给客户端。


## Controller层分析

controller主要由路由调用，由它去操作model（通过proxy）调用工具方法处理数据，**包含了大部分业务代码**，最后render view输出。因此controller比较重。按功能模块分为不同js，每个js中暴露多个控制器。

代码主要

* 通过eventproxy这个库来实现的各种异步逻辑。
* session的使用：如session.user中存储帐号信息
* cache(redis)的使用

## Model层分析

### [Mongodb](https://github.com/alsotang/node-lessons/tree/master/lesson15)

nodeclub使用Mongodb做数据库，存储用户和帖子信息。

* 文档型数据库：存储bson（json的超集，支持二进制数据）

  ```javascript
  var post = { // 一个文档，这里是一个json
    title: '呵呵的一天',
    author: 'alsotang',
    content: '今天网速很差',
    tags: ['呵呵', '网速', '差'], // 可以嵌套
  };
  ```

* 数据的层级是：数据库 -> collection -> document -> 字段。对应mysql的数据库（db） -> 表（table） -> 记录（record）-> 字段。

* 特点：

  * 表（collection）之间没有联系，不支持join
  * 不支持事务操作
  * **自动分片**：存储到多个实体（应该是按照key分片）
  * scheme-less：每个文档（record）格式可以不同，但是不建议这么干。这个特定可用在log系统中。
  * 支持索引，而且**支持复合索引，支持索引排序**

* 建议：关键数据还是mysql，mongodb适合存储非关键的数据。

* mongoose：odm框架（对象文档映射） ，对应 sql 中的 orm。

### Cookies存储

```javascript
// cookie-parser中间件
app.use(require('cookie-parser')(config.session_secret));
// write
var opts = {
    path: '/',
    maxAge: 1000 * 60 * 60 * 24 * 30,
    signed: true,
    httpOnly: true
  };
res.cookie(config.auth_cookie_name, auth_token, opts);
// clear
res.clearCookie(config.auth_cookie_name, { path: '/' });
// read
var auth_token = req.signedCookies[config.auth_cookie_name];
```

* cookie解析器,解析req.headers.cookie的内容,使用(value+secret)做签名（md5）,防止cookies篡改
* 可以直接从 req.cookies[] 或者 req.signedCookies[]中获取需要的值
* cookies中存储：
  * auth_token，是用户的登录凭证。
  * sessionId，session模块使用的id，代表会话对应的内存数据。
* 建议不要在cookies中存储过多信息，存一些对攻击者没有意义的值即可。

### session&&redis

* 本项目，redis的作用：
  * 存储session会话数据
    * user信息
  * 缓存部分数据，如热门帖子等。
* 实现：`express-session`,`connect-redis`模块结合使用，session自动存储到redis中。只要读写`req.session.xxx`即可。


## 登录模块分析

登录模块基本包含了应用的所有技术，是一个不错的分析点。

### 第三方登录与组成

第三方授权成功后分为三种情况：

* 非首次授权，直接登录论坛成功 
* 首次授权，跳转都新用户页面，让用户选择
  * 注册新用户
  * 与以前的帐号绑定

基本流程：

1. 通过oath2.0获得github的授权，获得accessToken

   ```javascript
   // oauth 中间件
   app.use(passport.initialize());
   // github oauth
   passport.serializeUser(function (user, done) {
     done(null, user);
   });
   passport.deserializeUser(function (user, done) {
     done(null, user);
   });
   // 注册了github的auth登录策略(passport-github)
   passport.use(new GitHubStrategy(config.GITHUB_OAUTH, githubStrategyMiddleware));
   // github登录的路由，入口，调用了github授权
   router.get('/auth/github', configMiddleware.github, passport.authenticate('github'));
   // oauth2.0的回调路径，见oAuth的原理，最终获得accessToken，回调github.callback方法(自定义方法)
   router.get('/auth/github/callback',
     passport.authenticate('github', { failureRedirect: '/signin' }), github.callback);
   ```

   ​

2. 使用accessToken访问github的帐号信息，获得email（必须要有email，否则提示错误）：这一步包含在 `passport.authenticate`中，`github.callback`回调中已经有这些值了。保存在req.user中。

3. 通过github查询是否该github帐号已经注册过，有则更新数据库信息（accessToken等）然后直接登录，没有跳转到新帐号创建界面。

   > 思考：
   >
   > * 为什么非首次可以直接登录成功？因为在这里已经拿到accessToken，说明用户已经授权，以前也登录过。
   > * 为什么首次授权成功后要跳转？这里的新用户界面**给用户选择是否绑定以前的帐号**。。如果没有这个需求可以直接创建账户然后登录成功。

   ```javascript
   exports.callback = function (req, res, next) {
     var profile = req.user;
     var email = profile.emails && profile.emails[0] && profile.emails[0].value;
     // 查询数据库
     User.findOne({githubId: profile.id}, function (err, user) {
       if (err) {
         return next(err);
       }
       // 非首次授权，已经是 cnode 用户时，更新他的资料
       if (user) {
         user.githubUsername = profile.username;
         user.githubId = profile.id;
         user.githubAccessToken = profile.accessToken;
         // user.loginname = profile.username;
         user.avatar = profile._json.avatar_url;
         user.email = email || user.email;
   	  // 保存数据到数据库
         user.save(function (err) {
           if (err) {
            // 错误处理 略。。。。
             return next(err);
           }
           // 登录成功！！！在res中生成auth_token，放入cookies，其中auth_token是user对象的_id,即数据库中的主id
           authMiddleWare.gen_session(user, res);
           // 回主页
           return res.redirect('/');
         });
       } else {
         // 首次授权，用户还未存在，则建立新用户，重定向网页
         req.session.profile = profile;
         return res.redirect('/auth/github/new');
       }
     });
   };

   function gen_session(user, res) {
     var auth_token = user._id + '$$'; // 以后可能会存储更多信息，用 $$ 来分隔
     var opts = {
       path: '/',
       maxAge: 1000 * 60 * 60 * 24 * 30,
       signed: true,
       httpOnly: true
     };
     res.cookie(config.auth_cookie_name, auth_token, opts); //cookie 有效期30天
   }
   ```

4. 新用户流程：添加数据库数据，生成sessionId返回给用户

   ```javascript
   router.get('/auth/github/new', github.new);
   // 新用户返回这个界面，其中有两个表单，1个是创建新用户，另一个是绑定以前的帐号。action都是调转到/auth/github/create，提交的表单有个isNew的标志，区分用户提交的哪个表单。
   exports.new = function (req, res, next) {
     res.render('sign/new_oauth', {actionPath: '/auth/github/create'});
   };
   ```

   表单界面布局代码:

```html
   <!--第一个表单：注意csrf isnew 两个hidden -->      
   <form id='signin_form' class='form-horizontal' action=<%= actionPath%> method='post'>
         <input type='hidden' name='_csrf' value='<%= csrf %>'/>
         <input type='hidden' name='isnew' value='1'/>

         <div class='control-group'>
           <label class='control-label'>通过 GitHub 帐号</label>

           <div class='controls'>
             <input type='submit' class='span-info' value="注册新账号">
           </div>
         </div>
   </form>
         
   <!--第二个表单 注意csrf 还有密码的处理-->
   <form id='signin_form' class='form-horizontal' action=<%= actionPath%> method='post'>
         <div class='control-group'>
           <label class='controls'>或者</label>
         </div>
         <div class='control-group'>
           <label class='control-label' for='name'>用户名</label>

           <div class='controls'>
             <input class='input-xlarge' id='name' name='name' size='30' type='text'/>
           </div>
         </div>
         <div class='control-group'>
           <label class='control-label' for='pass'>密码</label>
           <div class='controls'>
             <input class='input-xlarge' id='pass' name='pass' size='30' type='password'/>
           </div>
         </div>
         <input type='hidden' name='_csrf' value='<%= csrf%>'/>
         <div class='form-actions'>
           <input type='submit' class='span-primary' value='关联旧账号'/>
         </div>
   </form>
      
```

   创建帐号/绑定github核心代码

```javascript
   router.post('/auth/github/create', github.create);

   exports.create = function (req, res, next) {
     var profile = req.session.profile;

     var isnew = req.body.isnew;
     var loginname = validator.trim(req.body.name || '').toLowerCase();
     var password = validator.trim(req.body.pass || '');
     var ep = new eventproxy();
     ep.fail(next);

     if (!profile) {
       return res.redirect('/signin');
     }
     delete req.session.profile;

     var email = profile.emails && profile.emails[0] && profile.emails[0].value;
     if (isnew) { // 注册新账号
       // 创建新的数据库实体
       var user = new User({
         loginname: profile.username,
         pass: profile.accessToken, //github用户密码就是accessToken
         email: email,
         avatar: profile._json.avatar_url,
         githubId: profile.id, // githubId，一个重要标志符
         githubUsername: profile.username,
         githubAccessToken: profile.accessToken,//github的token
         active: true,
         accessToken: uuid.v4(),// uuid生成，这么名字不好
       });
       user.save(function (err) {
         if (err) {
           // 错误处理。。。如重复的email，loginname等
           return next(err);
         }
         // 登录成功！！！
         authMiddleWare.gen_session(user, res);
         res.redirect('/');
       });
     } else { // 关联老账号
       ep.on('login_error', function (login_error) {
         res.status(403);
         res.render('sign/signin', { error: '账号名或密码错误。' });
       });
       User.findOne({loginname: loginname},
         ep.done(function (user) {
           if (!user) {
             return ep.emit('login_error');
           }
         // 密码校验
           tools.bcompare(password, user.pass, ep.done(function (bool) {
             if (!bool) {
               return ep.emit('login_error');
             }
             // 更新github相关信息，绑定帐号
             user.githubUsername = profile.username;
             user.githubId = profile.id;
             // user.loginname = profile.username;
             user.avatar = profile._json.avatar_url;
             user.githubAccessToken = profile.accessToken;

             user.save(function (err) {
               if (err) {
                 return next(err);
               }
               // 登录成功！！！
               authMiddleWare.gen_session(user, res);
               res.redirect('/');
             });
           }));
         }));
     }
   };
```

5. 登录成功：登录成功后页面会重新跳转，发出新的http请求，在app.js中的中间件会更具cookies中的auth_token查询用户信息，并放入 `res.locals.current_user `和`req.session.user`中。前者用于界面渲染，后者用于session。

   ```javascript
   exports.authUser = function (req, res, next) {
     var ep = new eventproxy();
     ep.fail(next);

     // Ensure current_user always has defined.
     res.locals.current_user = null;

     ep.all('get_user', function (user) {
       if (!user) {
         return next();
       }
       user = res.locals.current_user = req.session.user = new UserModel(user);

       if (config.admins.hasOwnProperty(user.loginname)) {
         user.is_admin = true;
       }

       Message.getMessagesCount(user._id, ep.done(function (count) {
         user.messages_count = count;
         next();
       }));
     });
     // 第一步：查看session是否有值，有直接使用它
     if (req.session.user) {
       ep.emit('get_user', req.session.user);
     } else {
       // 第二步，如果没有session，则获取auth_token
       var auth_token = req.signedCookies[config.auth_cookie_name];
       // 没有auth_token，说明没有登录
       if (!auth_token) {
         return next();
       }
       // 有auth_token，没session，首次登录，查询数据库。
       var auth = auth_token.split('$$');
       var user_id = auth[0];
       UserProxy.getUserById(user_id, ep.done('get_user'));
     }
   };
   ```

   ​

6. 总结:
   1. 传输安全性：，**表单的post提交本身没有加密，安全是通过https保证的**。如果不使用https，建议[提交时用js加密处理](https://www.zhihu.com/question/20060155)。

   2. 数据库安全性：**数据库中存储的密码是加盐后的，没有存储明文（使用bcrypt模块）**

      > 简单介绍bcrypt的两方法
      >
      > * bcrypt.hash(source_string,salt_length,callback):第一个参数是密码明文，第二个参数是盐的长度，salt有时钟tick产生，然后会用salt计算一个hash值xxxx，然后拼接上saltvalue，最终生成`xxxxsaltvalue`。存入数据库。
      > * bcrypt.compare(sourceString,hash,callback),第一个参数是用户输入的密码，第二个是数据库存的`xxxxsaltvalue`，这个函数会取出xxxx和saltvalue，用saltvalue与sourceString计算的值比较xxxx，来验证密码输入是否正确。
      > * 思考：**为什么xxxxsaltvalue存在数据库中是安全的**，可以防止脱库？因为脱裤的原理是：计算常用的密码的md5形成字典，然后比较。但是这里**每个密码的salt都不一样**，没法生成字典，所以无法脱库。

   3. csrf：校验过程参考[这篇文章](https://cnodejs.org/topic/5533dd6e9138f09b629674fd)

   4. 字符串校验部分是用的validator库

   5. 这里存了**uuid**，但是没有用，[uuid生成是专门的库](https://github.com/broofa/node-uuid)

   6. **生成auth_token**，是用的数据库的`_id`保证唯一性，后续也方便查询数据库。过期时间是30天。思考：为什么要这样，为什么不直接把数据放入session中？？？

      > 这个是**token与session的区别！**，sessionId与token都存在cookies中，但是sessionId代表了内存中的一段数据，是服务端用来缓存会话的。token是登录凭证，表示登录成功过。
      >
      > * 如果只有session，没有token：用户一旦登录，信息直接存储在session中，可以使用。但是一旦session丢失（主要指**服务端重启，redis清空**。客户端清空cookies，SessionId没了token也没了），用户需要重新登录。
      > * 只有token，没有session：用户登录后token放入cookies。由于没有session，每次用户请求都带着token，**必须每一次都查询数据库**。
      > * 既有token也有session：服务端session丢失，只会用token重新查找数据库，恢复到session中。但是清空cookies依然会导致信息丢失。
      > * 当然，也可以用token作为sessionId，那么在cookies中只要存一个值就可以了。但是最好不要这样，职责清晰。

### 原生注册与登录

注册基本流程：

1. post提交表单：信息需要用户名密码，邮件，未加密（有https保证安全）直接post提交，**client未校验，全部由服务端校验**。

   ```html
   <form id='signup_form' class='form-horizontal' action='/signup' method='post'>
           <div class='control-group'>
             <label class='control-label' for='loginname'>用户名</label>

             <div class='controls'>
               <% if (typeof(loginname) !== 'undefined') { %>
               <input class='input-xlarge' id='loginname' name='loginname' size='30' type='text' value='<%= loginname %>'/>
               <% } else { %>
               <input class='input-xlarge' id='loginname' name='loginname' size='30' type='text' value=''/>
               <% } %>
             </div>
           </div>
           <div class='control-group'>
             <label class='control-label' for='pass'>密码</label>

             <div class='controls'>
               <input class='input-xlarge' id='pass' name='pass' size='30' type='password'/>
             </div>
           </div>
           <div class='control-group'>
             <label class='control-label' for='re_pass'>确认密码</label>

             <div class='controls'>
               <input class='input-xlarge' id='re_pass' name='re_pass' size='30' type='password'/>
             </div>
           </div>
           <div class='control-group'>
             <label class='control-label' for='email'>电子邮箱</label>

             <div class='controls'>
               <% if (typeof(email) !== 'undefined') { %>
               <input class='input-xlarge' id='email' name='email' size='30' type='text' value='<%= email %>'/>
               <% } else { %>
               <input class='input-xlarge' id='email' name='email' size='30' type='text'/>
               <% } %>
             </div>
           </div>
           <input type='hidden' name='_csrf' value='<%= csrf %>'/>

           <div class='form-actions'>
             <input type='submit' class='span-primary' value='注册'/>
             <a href="/auth/github">
               <span class="span-info">
                 通过 GitHub 登录
               </span>
             </a>
           </div>
    </form>
   ```

2. 处理post请求：注意路由中是post

   ```javascript
   exports.signup = function (req, res, next) {
     var loginname = validator.trim(req.body.loginname).toLowerCase();
     var email     = validator.trim(req.body.email).toLowerCase();
     var pass      = validator.trim(req.body.pass);
     var rePass    = validator.trim(req.body.re_pass);

     var ep = new eventproxy();
     ep.fail(next);
     ep.on('prop_err', function (msg) {
       res.status(422);
       res.render('sign/signup', {error: msg, loginname: loginname, email: email});
     });

     // 服务端验证信息的正确性
     if ([loginname, pass, rePass, email].some(function (item) { return item === ''; })) {
       ep.emit('prop_err', '信息不完整。');
       return;
     }
     if (loginname.length < 5) {
       ep.emit('prop_err', '用户名至少需要5个字符。');
       return;
     }
     if (!tools.validateId(loginname)) {
       return ep.emit('prop_err', '用户名不合法。');
     }
     if (!validator.isEmail(email)) {
       return ep.emit('prop_err', '邮箱不合法。');
     }
     if (pass !== rePass) {
       return ep.emit('prop_err', '两次密码输入不一致。');
     }
     // END 验证信息的正确性

     // mongo数据库查询or语句，是否用户名或者email占用
     User.getUsersByQuery({'$or': [
       {'loginname': loginname},
       {'email': email}
     ]}, {}, function (err, users) {
       if (err) {
         return next(err);
       }
       if (users.length > 0) {
         ep.emit('prop_err', '用户名或邮箱已被使用。');
         return;
       }
   	// 可以注册！！！！
       // 生成密码的hash值！不保存密码原文。
       tools.bhash(pass, ep.done(function (passhash) {
         // create gravatar
         var avatarUrl = User.makeGravatar(email);
         // 数据库操作
         User.newAndSave(loginname, loginname, passhash, email, avatarUrl, false, function (err) {
           if (err) {
             return next(err);
           }
           // 发送激活邮件
           mail.sendActiveMail(email, utility.md5(email + passhash + config.session_secret), loginname);
           res.render('sign/signup', {
             success: '欢迎加入 ' + config.name + '！我们已给您的注册邮箱发送了一封邮件，请点击里面的链接来激活您的帐号。'
           });
         });

       }));
     });
   };
   ```

   分析：

   * 首先对输入进行校验，
   * 查询是否已经注册，如果有，则提示用户
   * 如果没有，计算密码，hash。**只保存加盐的值，不保存原文。**
   * 发送用户激活邮件到邮箱，邮箱中包含一个连接其中有token和loginname。token=md5(passhash+serectkey+email)计算出（组成随意，能标识用户即可）。用户点击链接会调用服务器api，传回值，我们依据loginname查找用户，然后按照规则计算对比即可。
   * 注册完成，激活完成后，需要登录。

登录基本流程

1. 提交表单。。略，与上面类似。

2. 处理表单

   ```javascript
   exports.login = function (req, res, next) {
     // 输入校验
     var loginname = validator.trim(req.body.name).toLowerCase();
     var pass      = validator.trim(req.body.pass);
     var ep        = new eventproxy();

     ep.fail(next);

     if (!loginname || !pass) {
       res.status(422);
       return res.render('sign/signin', { error: '信息不完整。' });
     }

     // 判断是用户名还是邮箱登录
     var getUser;
     if (loginname.indexOf('@') !== -1) {
       getUser = User.getUserByMail;
     } else {
       getUser = User.getUserByLoginName;
     }

     ep.on('login_error', function (login_error) {
       res.status(403);
       res.render('sign/signin', { error: '用户名或密码错误' });
     });
    // 查询数据库
     getUser(loginname, function (err, user) {
       if (err) {
         return next(err);
       }
       if (!user) {
         return ep.emit('login_error');
       }
       var passhash = user.pass;
       // 密码对比，与注册相互呼应，参考第三方登录的总结一节
       tools.bcompare(pass, passhash, ep.done(function (bool) {
         if (!bool) {
           return ep.emit('login_error');
         }
         if (!user.active) {
           // 重新发送激活邮件
           mail.sendActiveMail(user.email, utility.md5(user.email + passhash + config.session_secret), user.loginname);
           res.status(403);
           return res.render('sign/signin', { error: '此帐号还没有被激活，激活链接已发送到 ' + user.email + ' 邮箱，请查收。' });
         }
         // 生成auth_token,与第三方登录类似
         authMiddleWare.gen_session(user, res);
         // 登录完成的跳转，通过referrer获得来源
         var refer = req.session._loginReferer || '/';
         for (var i = 0, len = notJump.length; i !== len; ++i) {
           if (refer.indexOf(notJump[i]) >= 0) {
             refer = '/';
             break;
           }
         }
         res.redirect(refer);
       }));
     });
   };
   ```

   分析：

   * 输入校验
   * 判断是用户名还是邮箱登录，设置相应的查找方法
   * 数据库查询，对比密码参考`第三方登录的总结`一节的`数据库安全`
   * 生成auth_token，参考第三方登录。
   * 登录完成的跳转redirect。referrer处理，**referrer中记载了跳转的源页面**，这里过滤了一些不用跳转的页面。关于[什么是referrer](https://imququ.com/post/referrer-policy.html)。`req.session._loginReferer`值在登录页面的请求中有赋值 `req.session._loginReferer =req.headers.referer;`
   * 登录完成，参考第三方登录过程。


## HTTP请求类型分析

按照开发的常用情况把所有http请求分为以下几种：

* 网站界面请求
  * 未登录的公用界面—以首页的帖子为例
  * 登录后的用户私有界面—以用户发表帖子为例
* Restful API：提供一下api，返回json数据，给其他人使用。

## 其他

### 文件上传

* busboy库

### 测试

* 入口：Makefile test目标 test-cov目标（覆盖率）
* mocha工具、istanbul工具
* 运行所有`*.test.js`文件
* benchmark??

### uuid

* 在其他系统应该挺重要的，这个项目没有用，只是用了auth_token代替

### 安全

* 使用一些公用的中间件如helmet防止攻击
* 全站https，很多接口内部没有再加密，如登录等。
* 密码等存储加盐hash，加密方法调用了bcrypt库
* cookies安全：不存有意义的数据，只要防止篡改（sign）即可。
* csrf攻击—[这篇文章](https://cnodejs.org/topic/5533dd6e9138f09b629674fd)，`csurf`模块
* xss—`validator`模块对输入校验

### [性能](http://expressjs.com/zh-cn/advanced/best-practice-performance.html)

* 上线前，开启production模式，会启用很多优化。
* 上线前，压缩多个js，css文件为一个min.js min.css文件。
  * 入口：Makefile build目标
  * 工具：Loader，国人开发的，国外也有很多类似的如grunt
  * 原理：可以参考布局文件（html）加载js/css的方法，上线后会优先加载压缩后的文件
* gzip：两种方式，在node中使用，或者用nginx处理
  * nodejs
    * 使用`compress`中间件**压缩**返回response。
    * `bodyparse`解析请求时会内部**解压**请求
  * nginx
* 视图缓存：`app.set('view cache', true);`
* api接口限流 ：在redis中计数
  * 登录用户，用户名限制api每天访问次数
  * 非登录用户，ip限制api每天访问次数

### 监控

* 日志系统，封装在`./common/logger`中
  * ` log4js`，模块输出日志到**文件**与**控制台**,不要直接使用`console.log`输出
  * `colors`模块，控制台输入着色。
* 请求日志输出：`./middlewares/request_log`，请求的输入，ip，返回状态，和**耗时**。
* 错误输出：分为debug和线上
  * debug— errorhandler模块，输出错误栈到客户端浏览器。
  * release— 是如此错误日志。
* oneapm工具？性能监控

### 处理跨域访问

对**暴露的API接口**（api/v1/xxx）使用cors模块处理跨域访问的问题：

* Access-Control-Allow-Origin 头设置，支持get和html的post请求
* OPITION请求处理，支持json的post请求

### 部署问题处理

* 代理背后（如nginx）：`app.enable('trust proxy');` nodejs会使用head中的x-forward字段填充ip字段,在nginx中有对应的配置。

## 一些缺乏的

* 没有消息订阅推送设计
* 没有后台管理系统，权限管理
* 文件上次部分没有仔细分析

## 总结

至此，Node.js部分学习完成，这是我第一次分析一个完整的后端项目，nodeclub是个很好的demo，
