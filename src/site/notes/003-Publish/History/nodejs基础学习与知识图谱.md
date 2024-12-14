---
{"dg-publish":true,"dg-path":"History/nodejs基础学习与知识图谱.md","permalink":"/History/nodejs基础学习与知识图谱/","title":"Node.js基础学习与知识图谱","tags":["技术","全栈","javascript","nodejs","后台开发"],"created":"2016-08-07 16:10:40","updated":"2016-08-07 16:10:40"}
---



Node.js是我第一次接触后端开发，有些思想需要提炼出共性，以扩展到不同的语言，如Java与Python，甚至Go语言。这里只是基础介绍，后续会有Node.js进阶与框架Express实践，之后会学习其他语言的开发技术与框架（初步计划是成熟的Java框架，出名的Python，最新的Go都看一下），提炼一些重要的特点。

[知识图谱](/files/nodejs基础知识图谱.mindnode)

## 安装

* 方法一：brew install nodejs
* 方法二（推荐）：nvm 管理多个node版本
* 直接安装
  * 权限问题https://docs.npmjs.com/getting-started/fixing-npm-permissions


## 特点

* 单线程+异步模型，这是node的优势
* 事件循环
* 理解setTimeout()和setInterval()函数的区别
* 推荐学习：http://www.ruanyifeng.com/blog/2014/10/event-loop.html

## 约定

* 如果某个函数需要回调函数作为参数，则回调函数是最后一个参数

* 回调函数本身的**第一个参数**，约定为上一步传入的**错误对象err**。如果没有发生错误则为null。只要判断回调函数的第一个参数，就知道有没有出错！这样还可以层层传递错误。

  ```javascript
  if(err) {
    // 除了放过No Permission错误意外，其他错误传给下一个回调函数
    if(!err.noPermission) {
      return next(err);
    }
  }
  ```

  > 传统的错误捕捉机制try…catch对于异步操作行不通，所以只能把错误交给回调函数处理

* 常用的全局变量

  - 全局对象
    - global，用来定义共享变量`global.x = 1`
    - process 进程控制与信息，下面介绍
    - console 控制台输出，生成环境不推荐使用
    - `__filename`：指向当前运行的脚本文件名。
    - `__dirname`：指向当前运行的脚本所在的目录。
  - 全局函数
    - **require()**：用于加载模块。
    - **setTimeout()**
    - **clearTimeout()**
    - **setInterval()**
    - **clearInterval()**
    - **Buffer()**：用于操作二进制数据。
  - 伪全局变量
    - module
    - module.exports
    - exports

## NPM

Node.js包管理工具，统一管理包的依赖与版本，配置文件是`package.json`。一般各种语言都会有[类似工具](http://harttle.com/2015/05/29/pkg-manager.html)，如python的pip，javascript前端的bower，java的maven（还有gradle，ant主要是编译，没有依赖管理）。

> npm 与bower区别：NPM主要运用于Node.js项目的内部依赖包管理，安装的模块位于项目根目录下的node_modules文件夹内。而Bower大部分情况下用于前端开发，对于CSS/JS/模板等内容进行依赖管理，依赖的下载目录结构可以自定义。npm设计之初就采用了的是嵌套的依赖关系树，这种方式显然对前端不友好；而Bower则采用扁平的依赖关系管理方式，使用上更符合前端开发的使用习惯。

* npm install xxx 

  * —save 参数表示将该模块写入`dependencies`属性
  * `-g`表示全局安装 ,默认安装在当前目录的node_modules下

* npm search

* npm run

  * 可以通过该命令运行一下脚本，如执行测试，`npm run`会创建一个Shell，执行指定的命令，并临时将`node_modules/.bin`加入PATH变量，这意味着本地模块可以直接运行。（**也可用自己写makefile实现，参考nodeclub**）

  * 以安装eslint为例

    ```json
    npm i eslint --save-dev
    // 在package.json添加直接调用eslint
    {
      "name": "Test Project",
      "devDependencies": {
        "eslint": "^1.10.3"
      },
      "scripts": {
        "lint": "eslint ."
      }
    }
    ```

    运行上面的命令以后，会产生两个结果。首先，ESLint被安装到当前目录的`node_modules`子目录；其次，`node_modules/.bin`目录会生成一个符号链接`node_modules/.bin/eslint`，指向ESLint模块的可执行脚本。

    `npm run lint`的时候，它会自动执行`./node_modules/.bin/eslint .`。

  * `start`和`test`属于特殊命令，可以省略`run`

  * scripts 中 pre post 钩子

  * [最佳实践](http://javascript.ruanyifeng.com/nodejs/npm.html#toc13)

    * npm-run-all
    * **live-server** 自动更新目录下内容？类似gulp？

* npm init 初始化一个node项目，生成`package.json`文件

* package.json 

  * `main`字段指定了加载该模块时的入门文件，默认是模块根目录下面的`index.js`。

  * `scripts`指定了运行脚本命令的npm命令行缩写，比如start指定了运行`npm run start`时，所要执行的命令。

  * `dependencies`字段指定了项目运行所依赖的模块

  * config字段用于向环境变量输出值`process.env.npm_package_config_port`

    ```json
    {
      "name" : "foo",
      "config" : { "port" : "8080" },
      "scripts" : { "start" : "node server.js" }
    }
    ```

* npm publish 发布模块到npm中，要先注册

* npm link 生成模块的符号连接，指向其他模块，实现其他目录下模块代码的变化可以立刻反应过来。


## 基本知识

### 模块Module

* Node.js中模块是同步加载的（CommonJS规范）。

  * 每个`.js`文件都是一个模块，它们内部各自使用的变量名和函数名都互不冲突
  * require加载模块

* 每个模块内部，都有一个`module`对象，代表当前模块。

  * `module.exports` 表示模块对外输出的值
  * `module.children` 返回一个数组，表示该模块要用到的其他模块。
  * `module.parent` 返回一个对象，表示调用该模块的模块。

* 加载模块

  * `require`方法读入并执行一个JavaScript文件，然后返回该模块的exports对象

  * 加载规则

    * 默认后缀名`.js`。如果找不到会尝试`.json`、`.node`（编译好的node模块） 即

      ```javascript
      var foo = require('foo');
      //  等同于
      var foo = require('foo.js');
      ```

    * 以`/`，`./`分别是绝对和相对路径加载，没有指定则是

      1. 系统核心模块
      2. 先搜索本地node_modules目录
      3. 父目录的node_modules目录
      4. 全局node_modules目录

    * 加载的目标是目录，自动查看该目录的`package.json`文件，然后加载`main`字段指定的入口文件。如果`package.json`文件没有`main`字段，或者根本就没有`package.json`文件，则会加载该目录下的`index.js`文件或`index.node`文件

    * require多次，只会运行一次，自动缓存结果。

* ​

* 输出模块，区分`exports`与`module.exports`，参考模块加载原理

  ```javascript
  function greet(name) {
      console.log('Hello, ' + name + '!');
  }

  function hello() {
      console.log('Hello, world!');
  }
  // 方法一(推荐)，输出对象，包括多个函数
  module.exports = {
      hello: hello,
      greet: greet
  };
  // 直接输入一个函数对象
  module.exports = function(){
    console.log('Hello, world!');
  }
  // 方法二
  exports.hello = hello;
  exports.greet = greet;

  // 错误写法！！，代码可以执行，但是模块并没有输出任何变量:
  //exports = {
  //    hello: hello,
  //    greet: greet
  //};
  ```

* 模块加载原理

  ```javascript
  // 1.构造module对象
  var module = {
      id: 'hello',
      exports: {} // 构造了一个空对象
  };
  // 加载函数，注意，参数exports，与module
  var load = function (exports, module) {
      // hello.js的文件内容
      ...
      // load函数返回:
      return module.exports;
  };
  // 开始鸡杂
  var exported = load(module.exports, module);
  ```

## 核心模块

内置在node中的代码，会优先加载。源码都在Node的lib子目录中。为了提高运行速度，它们安装时都会被编译成二进制文件。

- **http**：提供HTTP服务器功能。
- **path**：处理文件路径。
- **url**：解析URL。
- **fs**：与文件系统交互。
- **querystring**：解析URL的查询字符串。
- **child_process**：新建子进程。
- **util**：提供一系列实用小工具。
- **crypto**：提供加密和解密功能，基本上是对OpenSSL的包装。

### http

* 处理请求

  * 处理get请求

    ```javascript
    var http = require('http');

    http.createServer(function (request, response){
      response.writeHead(200, {'Content-Type': 'text/plain'});
      response.end('Hello World\n');
    }).listen(8080, '127.0.0.1');
    ```

  * 处理post请求

    ```javascript
    var http = require('http');

    http.createServer(function (req, res) {
      var content = "";

      req.on('data', function (chunk) {
        content += chunk;
      });

      req.on('end', function () {
        res.writeHead(200, {"Content-Type": "text/plain"});
        res.write("You've sent: " + content);
        res.end();
      });

    }).listen(8080);
    ```

  * 返回数据 [response](https://nodejs.org/api/http.html#http_class_http_serverresponse)

    * 手动返回
      * `res.writeHead(statusCode,"{"xxx":"content"}")`
      * `res.write("xxx")`
      * `res.end("xxx");` **必须有**，表示结束。参数可以不写。

* 发出请求

  * 发出get—`http.get(callback)`
  * 发出post/get — `http.request(options[, callback])`

* https：Node内置Https支持(但是建议Ngnix处理)

  ```javascript
  var https = require('https');
  var fs = require('fs');

  var options = {
    key: fs.readFileSync('key.pem'), // 私钥
    cert: fs.readFileSync('cert.pem') // 证书
    // ca: certificateAuthorityCertificate
  };

  var a = https.createServer(options, function (req, res) {
    res.writeHead(200);
    res.end("hello world\n");
  }).listen(8000);
  ```

### path && url

* `path.join()` 组合完整的文件路径，**处理不同操作系统路径表示不同的问题**

* `path.resolve()` 将相对路径转为绝对路径

  ```javascript
  var path = require('path');

  // 解析当前目录:
  var workDir = path.resolve('.'); // '/Users/michael'

  // 组合完整的文件路径:当前目录+'pub'+'index.html':
  var filePath = path.join(workDir, 'pub', 'index.html');
  ```

* 其他不常用

  * path.relative 方法接受两个参数，这两个参数都应该是绝对路径。该方法返回第二个路径想对于地一个路径的系那个相对路径

* `url.parse('http://www.xx.com')`解析网址，生成对象

  ```javascript
  var url = require('url');

  console.log(url.parse('http://user:pass@host.com:8080/path/to/file?query=string#hash'));
  // 生成对象
  Url {
    protocol: 'http:',
    slashes: true,
    auth: 'user:pass',
    host: 'host.com:8080',
    port: '8080',
    hostname: 'host.com',
    hash: '#hash',
    search: '?query=string',
    query: 'query=string',
    pathname: '/path/to/file',
    path: '/path/to/file?query=string',
    href: 'http://user:pass@host.com:8080/path/to/file?query=string#hash' }
  ```

### fs

* 读（异步） `fs.readFile`

  * 文本

    ```javascript
    // 用utf8格式读取文件
    fs.readFile('sample.txt', 'utf-8', function (err, data) {
        if (err) { // 失败处理
            console.log(err);
        } else {
            console.log(data);
        }
    });
    ```

  * bytes（二进制）:不传入文件编码，默认是返回Buffer对象

    ```javascript
    // 读取图片
    fs.readFile('sample.png', function (err, data) {
        if (err) {
            console.log(err);
        } else {
            console.log(data);
            console.log(data.length + ' bytes');
        }
    });
    ```

    * Buffer对象：一个包含零个或任意个字节的数组（注意**和Array不同**）

      ```javascript
      // Buffer -> String
      var text = data.toString('utf-8');
      console.log(text);
      // String -> Buffer
      var buf = new Buffer(text, 'utf-8');
      console.log(buf);
      ```

* 写（异步）`fs.writeFile`

  * 文本 第二个参数传入字符串，回调函数前面，还可以再加一个参数，表示写入字符串的编码（默认是`utf8`）。

  * bytes 第二个参数传入Buffer

    ```javascript
    var data = 'Hello, Node.js';
    fs.writeFile('output.txt', data, function (err) {
        if (err) {
            console.log(err);
        } else {
            console.log('ok.');
        }
    });
    ```

* 查看文件信息

  * `fs.exists()` 是否存在 `fs.exists('/path/to/file', function (exists) { });`

  * `fs.stat()`，返回stat对象

    ```javascript
    fs.stat('sample.txt', function (err, stat) {
        if (err) {
            console.log(err);
        } else {
            // 是否是文件:
            console.log('isFile: ' + stat.isFile());
            // 是否是目录:
            console.log('isDirectory: ' + stat.isDirectory());
            if (stat.isFile()) {
                // 文件大小:
                console.log('size: ' + stat.size);
                // 创建时间, Date对象:
                console.log('birth time: ' + stat.birthtime);
                // 修改时间, Date对象:
                console.log('modified time: ' + stat.mtime);
            }
        }
    });
    ```

* 目录操作

  * mkdir()

    ```javascript
    fs.mkdir('./helloDir',0777, function (err) {
      if (err) throw err;
    });
    ```

  * readdir()

    ```javascript
    fs.readdir(process.cwd(), function (err, files) {
      if (err) {
        console.log(err);
        return;
      }

      var count = files.length;
      console.log(files);
    });
    ```

* **用流的方式读读写**：打开大型的文本文件，读取操作的缓存装不下，只能分成几次发送，每次发送会触发一个`data`事件，发送结束会触发`end`事件。

  ```javascript
  // 读
  var input = fs.createReadStream('lines.txt');
  input.on('data', function(data) {
      remaining += data;
    });

  input.on('end', function() {
  });
  input.on('error',function(err){});
  // 写
  var out = fs.createWriteStream(fileName, {
    encoding: 'utf8'
  });
  out.write(str);out.write(str2);
  out.end();

  var ws2 = fs.createWriteStream('output2.txt');
  ws2.write(new Buffer('使用Stream写入二进制数据...\n', 'utf-8'));
  ws2.end();

  // pipe 可以把两个水管串成一个更长的水管一样，两个流也可以串起来。一个Readable流和一个Writable流串起来后，所有的数据自动从Readable流进入Writable流，这种操作叫pipe。
  var rs = fs.createReadStream('sample.txt');
  var ws = fs.createWriteStream('copied.txt');
  rs.pipe(ws);
  // 当Readable流的数据读取完毕，end事件触发后，将自动关闭Writable流。如果我们不希望自动关闭Writable流，需要传入参数：
  readable.pipe(writable, { end: false });
  ```

* 其他

  * 同步读写(不推荐)

    * fs.readFileSync('sample.txt', 'utf-8');
    * fs.writeFileSync('output.txt', data);
    * `fs.existsSync('output.txt')`
    * mkdirSync()

  * 监控文件变化，如果该文件发生变化，就会自动触发回调函数。`unwatchfile`方法用于解除对文件的监听。

    ```javascript
    fs.watchFile('./testFile.txt', function (curr, prev) {
      console.log('the current mtime is: ' + curr.mtime);
      console.log('the previous mtime was: ' + prev.mtime);
    });
    ```

### child_process

* `exec`方法用于执行bash命令，然后我们通过流监控输入输出（标准输入输出流）

  ```javascript
  var exec = require('child_process').exec;
  var child = exec('ls -l');
  // 监控输入输出流
  child.stdout.on('data', function(data) {
    console.log('stdout: ' + data);
  });
  child.stderr.on('data', function(data) {
    console.log('stdout: ' + data);
  });
  child.on('close', function(code) {
    console.log('closing code: ' + code);
  });

  // 或者这样
  exec('node -v', function(error, stdout, stderr) {
    console.log('stdout: ' + stdout);
    console.log('stderr: ' + stderr);
    if (error !== null) {
      console.log('exec error: ' + error);
    }
  });
  ```

* execFile()：execFile方法直接执行特定的程序，参数作为数组传入，不会被bash解释，因此具有较高的安全性。

  ```javascript
  var path = ".";
  child_process.execFile('/bin/ls', ['-l', path], function (err, result) {
      console.log(result)
  });
  ```

* spawn:创建一个**子进程**来执行特定命令，用法与execFile类似，但是只能监听流

  ```javascript
  var path = '.';
  var ls = child_process.spawn('/bin/ls', ['-l', path]);
  ls.stdout.on('data', function (data) {
    console.log('stdout: ' + data);
  });

  ls.stderr.on('data', function (data) {
    console.log('stderr: ' + data);
  });

  ls.on('close', function (code) {
    console.log('child process exited with code ' + code);
  });
  ```

* fork()&&send()：fork方法直接创建一个子进程，执行Node脚本，`fork('./child.js')` 相当于 `spawn('node', ['./child.js'])` 。与spawn方法不同的是，fork会在父进程与子进程之间，建立一个通信管道，用于进程之间的通信。

  ```javascript
  var n = child_process.fork('./child.js');
  n.on('message', function(m) {
    console.log('PARENT got message:', m);
  });
  // 主线程与子线程通讯
  n.send({ hello: 'world' });

  // ./child.js
  process.on('message', function(m) {
    console.log('CHILD got message:', m);
  });
  // 子线程与主线程通讯
  process.send({ foo: 'bar' });
  ```

* 其他：

  * 同步版本：execSync
  * **cluster模块**,基于child_process的fork封装的一个模块，实现node多进程，创建，通讯，一些**node进程管理工具**就是基于它做的，如PM2，实现应用的多实例运行，死机重启等。
    * cluster.fork()
    * cluster.isMaster
    * 各种on监听和send方法

### crypto

提供通用的加密和哈希算法

* MD5和SHA1 摘要算法

  ```javascript
  const crypto = require('crypto');
  //可以是sha1，sha256，sha512
  const hash = crypto.createHash('md5');
  // 可任意多次调用update():
  hash.update('Hello, world!'); // 参数可以是Buffer，默认UTF-8
  hash.update('Hello, nodejs!');

  console.log(hash.digest('hex'));
  ```

* 带密钥的摘要算法Hmac

  ```javascript
  const hmac = crypto.createHmac('sha256', 'secret-key');

  hmac.update('Hello, world!');
  hmac.update('Hello, nodejs!');

  console.log(hmac.digest('hex')); // 80f7e22570...
  ```

* 对称加密AES

* 非对称DH算法、RSA

* 证书相关

### util

* util.inherits(classone, superClass); 继承

### 其他模块

* proces对象，部署了`EventEmitter`接口
  * 属性
    * **stdout，stdin，stderr**
    * **process.argv**：返回当前进程的命令行参数数组。
    * **process.env**：返回一个对象，成员为当前Shell的环境变量，比如`process.env.HOME`。
    * process.installPrefix：node的安装路径的前缀，比如`/usr/local`，则node的执行文件目录为`/usr/local/bin/node`。
    * **process.pid**：当前进程的进程号。
    * **process.platform**：当前系统平台，比如Linux。
    * process.title：默认值为“node”，可以自定义该值。
    * process.version：Node的版本，比如v0.10.18。
  * 方法
    * process.chdir()：切换工作目录到指定目录。
    * **process.cwd()**：返回运行当前脚本的工作目录的路径。
    * **process.exit()**：退出当前进程。
    * process.getgid()：返回当前进程的组ID（数值）。
    * process.getuid()：返回当前进程的用户ID（数值）。
    * **process.nextTick()**：指定回调函数在当前**执行栈的尾部**、下一次Event Loop之前执行。
    * **process.on()**：监听事件。
      * `data`事件：数据输出输入时触发
      * `SIGINT`事件：接收到系统信号`SIGINT`时触发，主要是用户按`Ctrl + c`时触发。
      * `SIGTERM`事件：系统发出进程终止信号`SIGTERM`时触发
      * `exit`事件：进程退出前触发
      * `uncaughtException`事件
      * `message`，其他进程消息，在cluster模块中使用
    * process.setgid()：指定当前进程的组，可以使用数字ID，也可以使用字符串ID。
    * process.setuid()：指定当前进程的用户，可以使用数字ID，也可以使用字符串ID。
* assert模块
* os模块，用处不大。。。查询系统信息
* net模块：socket底层通讯，分为客户端和服务端
* dns模块，解析域名

## 重要概念

### Buffer对象

`Buffer`对象是Node处理二进制数据的一个接口。它是Node原生提供的全局对象，可以直接使用，不需要`require('buffer')`。fs模块有涉及。

### Event模块

Node对“发布/订阅”模式（publish/subscribe）的实现。一个对象通过这个模块，向另一个对象传递消息。

* EventEmitter

  * 监听
    * on
    * once 类似于on方法，但是即使emit多次，回调函数只触发一次。
    * removeListener
    * removeAllListeners
  * 触发事件 emit

* 部署EventEmitter接口

  ```javascript
  var EventEmitter = require('events').EventEmitter;

  function Dog(name) {
    this.name = name;
  }
  // 继承了EventEmitter，以后该对象就可以各种on
  Dog.prototype.__proto__ = EventEmitter.prototype;
  // 另一种写法
  // Dog.prototype = Object.create(EventEmitter.prototype);
  // util.inherits(Dog, EventEmitter);
  var simon = new Dog('simon');

  simon.on('bark', function(){
    console.log(this.name + ' barked');
  });

  setInterval(function(){
    simon.emit('bark'); // 触发事件
  }, 500);
  ```

### 流-Stream接口

* 概念：Stream把较大的数据，拆成很小的部分。只要命令部署了Stream接口，就可以把一个流的输出接到另一个流的输入。Node引入了这个概念，通过Stream为异步读写数据提供的统一接口

* 应用：文件读写，网络请求（post），内存数据都与流相关。

* Stream接口最大特点就是通过事件通信，具有readable、writable、drain、**data**、**end**、close等事件。（**理解成特殊的EventEmitter**）读写数据时，每读入（或写入）一段数据，就会触发一次data事件，全部读取（或写入）完毕，触发end事件。如果发生错误，则触发error事件。

* 例子：读取文件返回给网络请求

  ```javascript
  var http = require('http');
  var fs = require('fs');

  var server = http.createServer(function (req, res) {
    var stream = fs.createReadStream(__dirname + '/data.txt');
    // pipe 内部处理了data和end事件
    stream.pipe(res);
  });
  server.listen(8000);
  ```


## 异常处理

* try catch，**无法捕获异步异常**
* 回调函数第一个参数err。
* 通过EventEmitter接口，监听error事件

