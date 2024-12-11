---
{"dg-publish":true,"permalink":"/003-Publish/History/JS相关API与Lib学习笔记与知识图谱/","title":"JS相关API/Lib学习笔记与知识图谱","tags":["技术","全栈","javascript","web开发"],"created":"2024-12-11T18:09:35.724+08:00","updated":"2024-12-11T18:09:35.725+08:00"}
---



介绍一下JS常用的原生API：

* 浏览器元素DOM的操作
* 表单操作
* 一些HTML5的API
* AJAX访问的API

一些JS常用的库：

* jQuery：一个dom操作的便捷库，兼容性优化，动画实现
* underscore：函数式编程的库。

知识图谱：[下载](/files/JS相关API与Lib知识图谱.mindnode)

## JS原生API

### 浏览器对象

* `window`：窗口的信息
  * `innerWidth`获取浏览器窗口的内部宽度和高度。内部宽高是指除去菜单栏、工具栏、边框等占位元素后，用于显示网页的净宽高。
  * `innerHeight`
  * `outerWidth` 浏览器窗口的整个宽高。
  * `outerHeight`
* `navigator`：浏览器的信息。可以很容易地被用户修改，所以JavaScript读取的值不一定是正确的
  * navigator.appName：浏览器名称；
  * navigator.appVersion：浏览器版本；
  * navigator.language：浏览器设置的语言；
  * navigator.platform：操作系统类型；
  * navigator.userAgent：浏览器设定的`User-Agent`字符串。
* `screen`：对象表示屏幕的信息，常用的属性有：
  * screen.width：屏幕宽度，以像素为单位；
  * screen.height：屏幕高度，以像素为单位；
  * screen.colorDepth：返回颜色位数，如8、16、24。
* `location`：对象表示当前页面的URL信息
  * `location.href` 完整url
  * `location.protocol`、`location.host`、`location.port`、`location.pathname`、`location.search`、`location.hash`。

    ```javascript
    // http://www.example.com:8080/path/index.html?a=1&b=2#TOP
    location.protocol; // 'http'
    location.host; // 'www.example.com'
    location.port; // '8080'
    location.pathname; // '/path/index.html'
    location.search; // '?a=1&b=2'
    location.hash; // 'TOP'
    ```

  * `location.assign()`：加载一个新页面
    * 相对路径`'/discuss' ` 
    * 新网址
  * `location.reload()`：重新加载当前页面，
* document 表示当前页面。**`document`对象就是整个DOM树的根节点。**
  * `document.title`、`document.head`、`document.body`
  * `document.cookie`：

    > 建议服务器在设置Cookie时可以使用`httpOnly`。防止cookie被其他非法js文件窃取。

  * `document.getElementById()`
  * `document.getElementsByTagName()`
* ~~history~~：由于大量使用AJAX和页面交互，简单粗暴地调用`history.back()`可能会让用户感到非常愤怒。
  * `back()`
  * `forward ()`

### DOM操作

DOM最重要的API就是表示`Document`的`document`与`Element`的节点。分为增删改查。
* 查：
  * `document.getElementById()` 直接定位唯一的一个DOM节点
  * `document.getElementsByTagName()` 按照tag返回element**数组**
  * `document.getElementsByClassName()`按照clsss返回element**数组**
  * `querySelector()`通过选择器查找元素。参数传入选择器ex:`'#q1'`。
  * `querySelectorAll()`:通过选择器查找所有元素。参数传入选择器ex:`'p'`。返回**数组**
  * `xxx.children` 所有子节点。**只读**,但是**遍历增删时时会动态改变**
  * `xxx.firstElementChild`、`xxx.lastElementChild` 第一个，最后一个节点。只读

  ```javascript
  // 返回ID为'test'的节点：
  var test = document.getElementById('test');
  // 先定位ID为'test-table'的节点，再返回其内部所有tr节点：
  var trs = document.getElementById('test-table').getElementsByTagName('tr');
  // 先定位ID为'test-div'的节点，再返回其内部所有class包含red的节点：
  var reds = document.getElementById('test-div').getElementsByClassName('red');
  // 通过querySelector获取ID为q1的节点：
  var q1 = document.querySelector('#q1');
  // 查找所有的p
  var ps = document.querySelectorAll('p');
  ```

  > 概念：DOM节点是指`Element`，但是DOM节点实际上是`Node`，在HTML中，`Node`包括`Element`、`Comment`、`CDATA_SECTION`等很多种，以及根节点`Document`类型，但是，绝大多数时候我们只关心`Element`，也就是实际控制页面结构的`Node`，其他类型的`Node`忽略即可。根节点`Document`已经自动绑定为全局变量`document`。

* 更新、修改节点内容
  * `innerHTML`： `document`和`element`都有。**完全替换所有内容，可以替换成新的element节点。**

    > 注意：innerHTML如果是从网络获取的字符串，可能被插入js脚本`<script>....</script>`。因此建议插入之前进行编码。即会修改成`&lt;script&gt;...&lt;/script&gt;`,改代码不会执行

  * `innerText`：写入改属性的值会自动HTML编码。textContent也是，细节不同是，后者返回完整元素（包括隐藏的）
  * `xxelement.style.属性` 修改节点属性，对应css中的属性名，注意：因为CSS允许`font-size`这样的名称，但它并非JavaScript有效的属性名，所以需要在**JavaScript中改写为驼峰式命名**`fontSize`
* 添加元素，添加属性
  * 直接使用`innerHTML ="<span>child</span>"`:完全替换
  * `parentElement.appendChild(element)`插入到最后。
  * `parentElement.insertBefore(newElement, referenceElement);`插入到引用的元素之前。
  * 新建节点
    * ` var e = document.createElement('p');`
    * 添加属性
      * `e.setAttribute('id', '123')`;
      * `e.id="123"` 与上面功能相同
  * 注意点：如果不新建节点，直接插入已经存在于当前的文档树的节点，**这个节点首先会从原先的位置删除，再插入到新的位置。**

  ```javascript
  var d = document.createElement('style');
  d.setAttribute('type', 'text/css');
  d.innerHTML = 'p { color: red }';
  document.getElementsByTagName('head')[0].appendChild(d);
  ```

* 删除：要删除一个节点，首先要获得该节点本身以及它的父节点
  * 调用父节点的`removeChild`把自己删掉

    ```javascript
    // 拿到待删除节点:
    var self = document.getElementById('to-be-removed');
    // 拿到父节点:
    var parent = self.parentElement;
    // 删除:
    var removed = parent.removeChild(self);
    removed === self; // true
    ```

  * 注意点，删除时读取`children`属性容易出错

    ```javascript
    var parent = document.getElementById('parent');
    parent.removeChild(parent.children[0]);
    parent.removeChild(parent.children[1]); // <-- 浏览器报错
    ```

### 表单操作

* 读写输入框的值：
  * 常见的输入框：HTML表单的输入控件主要有以下几种：
    - 文本框，对应的`<input type="text">`，用于输入文本；
    - 口令框，对应的`<input type="password">`，用于输入口令；
    - 单选框，对应的`<input type="radio">`，用于选择一项；
    - 复选框，对应的`<input type="checkbox">`，用于选择多项；
    - 下拉框，对应的<select>``，用于选择一项；
    - **隐藏文本，对应的`<input type="hidden">`，用户不可见，但表单提交时会把隐藏文本发送到服务器。**
  * 读取：

    ```
    // text、password、hidden以及select。
    var input = document.getElementById('email');
    input.value; // '用户输入的值'

    // 单选框和复选框，value属性返回的永远是HTML预设的值，而我们需要获得的实际是用户是否“勾上了”选项，所以应该用checked判断
    var mon = document.getElementById('monday');
    mon.checked; // true或者false
    ```

  * 设置：类似上面，赋值即可。 

  > html5有很多input控件，使用类似。不支持HTML5的浏览器无法识别新的控件，会把它们当做`type="text"`来显示。支持HTML5的浏览器将获得格式化的字符串。例如，`type="date"`类型的`input`的`value`将保证是一个有效的`YYYY-MM-DD`格式的日期，或者空字符串。

* 表单提交
  * 方式一：设置button的onclick事件，自己调用`form.submit();`提交
  * 方式二：监听summit按钮的提交事件，拦截处理。配合hidden使用。

    ```html
    <!-- HTML -->
    <!--注意到id为md5-password的<input>标记了name="password"，而用户输入的id为input-password的<input>没有name属性。没有name属性的<input>的数据不会被提交。-->
    <form id="login-form" method="post" onsubmit="return checkForm()">
        <input type="text" id="username" name="username">
        <input type="password" id="input-password">
        <input type="hidden" id="md5-password" name="password">
        <button type="submit">Submit</button>
    </form>

    <script>
    function checkForm() {
        var input_pwd = document.getElementById('input-password');
        var md5_pwd = document.getElementById('md5-password');
        // 把用户输入的明文变为MD5:
        md5_pwd.value = toMD5(input_pwd.value);
        // 继续下一步:
        return true;
    }
    </script>
    ```

### 文件操作

* 在HTML表单中，可以上传文件的唯一控件就是`<input type="file">`。

  > 当一个表单包含`<input type="file">`时，表单的**`enctype`必须指定为`multipart/form-data`**，**`method`必须指定为`post`**，浏览器才能正确编码并以`multipart/form-data`格式发送表单的数据。


* HTML5的File API提供了`File`和`FileReader`两个主要对象，可以获得文件信息并读取文件。

### AJAX

* 异步访问—XMLHttpRequest（低版本IE ActiveXObject）

  * `XMLHttpRequest`对象的`open()`方法有3个参数，第一个参数指定是`GET`还是`POST`，第二个参数指定URL地址，第三个参数指定是否使用异步，默认是`true`。
  * 调用`send()`方法才真正发送请求。`GET`请求不需要参数，`POST`请求需要把body部分以字符串或者`FormData`对象传进去。
  * `onreadystatechange`回调函数
    * `readyState === 4`判断请求是否完成，
    * `status === 200` 判断是否是一个成功的响应。

  ```javascript
  var request;
  // 低版本IE兼容
  if (window.XMLHttpRequest) {
      request = new XMLHttpRequest();
  } else {
      request = new ActiveXObject('Microsoft.XMLHTTP');
  }

  request.onreadystatechange = function () { // 状态发生变化时，函数被回调
      if (request.readyState === 4) { // 成功完成
          // 判断响应结果:
          if (request.status === 200) {
              // 成功，通过responseText拿到响应的文本:
              return success(request.responseText);
          } else {
              // 失败，根据响应码判断失败原因:
              return fail(request.status);
          }
      } else {
          // HTTP请求还在继续...
      }
  }

  function success(text) {
      var textarea = document.getElementById('test-response-text');
      textarea.value = text;
  }

  function fail(code) {
      var textarea = document.getElementById('test-response-text');
      textarea.value = 'Error code: ' + code;
  }

  // 发送GET请求，注意地址没有跨域:
  request.open('GET', '/api/categories');
  request.send();

  alert('请求已发送，请等待响应...');
  ```

* 跨域与安全访问

  * 当前页面默认无法访问其他URL域名下的网页。

    > 域名要相同（`www.example.com`和`example.com`不同），协议要相同（`http`和`https`不同），端口号要相同（默认是`:80`端口，它和`:8080`就不同）。

  * 如果要调用其他页面的接口怎么办？解决办法

    * ~~Flash插件~~：过时了

    * 同源域名下架设一个代理服务器来转发：比如访问`'/proxy?url=http://www.sina.com.cn'`。有Server开发工作量。

    * JSONP：利用 `<script src="http://example.com/abc.js"></script>`可以访问外域。一般改js脚本会返回一个`callback("data")`调用。其中callback是调用接口是传给它的。

      ```javascript

      function refreshPrice(data) {
          console.log(data)
      }

      function getPrice() {
          var
              js = document.createElement('script'),
              head = document.getElementsByTagName('head')[0];
        // 访问API，callback通过参数传入refreshPrice 
        // 返回的js是 refreshPrice([...data..]);
          js.src = 'http://api.money.126.net/data/feed/0000001,1399001?callback=refreshPrice';
          head.appendChild(js);
      }
      ```

    * **CROS** 推荐，全称Cross-Origin Resource Sharing，是HTML5规范定义的如何跨域访问资源：

      * 简单请求：**服务端通过设置`Access-Control-Allow-Origin`来控制哪些网站可以获取资源**。`Access-Control-Allow-Origin`为`http://my.com`，或者是`*`，本次请求就可以成功。

        ![](http://www.liaoxuefeng.com/files/attachments/00143640805071744d58164a40e42ef92b9973824451595000)

        >  简单请求包括GET、HEAD和POST（POST的Content-Type类型仅限`application/x-www-form-urlencoded`、`multipart/form-data`和`text/plain`），并且不能出现任何自定义头（例如，`X-Custom: 12345`），通常能满足90%的需求。
        >
        >  **不包括post的Json请求**

      * preflighted请求：对于PUT、DELETE以及其他类型如`application/json`的POST请求，在发送AJAX请求之前，**浏览器**会先发送一个`OPTIONS`请求到这个URL上，询问目标服务器是否接受。（**web页面不需要修改，只有server需要**）

        ```xml
        <!-- request -->
        OPTIONS /path/to/resource HTTP/1.1
        Host: bar.com
        Origin: http://my.com
        Access-Control-Request-Method: POST
        <!-- response -->
        HTTP/1.1 200 OK
        Access-Control-Allow-Origin: http://my.com
        Access-Control-Allow-Methods: POST, GET, PUT, OPTIONS
        Access-Control-Max-Age: 86400
        ```

        浏览器确认服务器响应的`Access-Control-Allow-Methods`头确实包含将要发送的AJAX请求的Method，才会继续发送AJAX，否则，抛出一个错误。



* ajax结合Promise应用

  ```javascript
  // 工具方法：ajax函数将返回Promise对象:
  function ajax(method, url, data) {
      var request = new XMLHttpRequest();
      return new Promise(function (resolve, reject) {
          request.onreadystatechange = function () {
              if (request.readyState === 4) {
                  if (request.status === 200) {
                      resolve(request.responseText);
                  } else {
                      reject(request.status);
                  }
              }
          };
          request.open(method, url);
          request.send(data);
      });
  }
  // 注意访问地址不要跨域
  var p = ajax('GET', '/api/categories');
  p.then(function (text) { // 如果AJAX成功，获得响应内容
      console.log(text);
  }).catch(function (status) { // 如果AJAX失败，获得响应代码
      console.log('ERROR: ' + status);
  });
  ```


### Canvas

html5的一个元素，可以绘制内容。类似android的canvas。

Canvas除了能绘制基本的形状和文本，还可以实现动画、缩放、各种滤镜和像素转换等高级操作。如果要实现非常复杂的操作，考虑以下优化方案：

- 通过创建一个不可见的Canvas来绘图，然后将最终绘制结果复制到页面的可见Canvas中；
- 尽量使用整数坐标而不是浮点数；
- 可以创建多个重叠的Canvas绘制不同的层，而不是在一个Canvas中绘制非常复杂的图；
- 背景图片如果不变可以直接用``标签并放到最底层。


## JQuery

### 功能

- 消除浏览器差异：你不需要自己写冗长的代码来针对不同的浏览器来绑定事件，编写AJAX等代码；
- 简洁的操作DOM的方法：写`$('#test')`肯定比`document.getElementById('test')`来得简洁；
- 轻松实现动画、修改CSS等各种操作。

> 目前jQuery有1.x和2.x两个主要版本，区别在于2.x移除了对古老的IE 6、7、8的支持，因此2.x的代码更精简。选择哪个版本主要取决于你是否想支持IE 6~8。

### Tip

* `$`符号冲突 `jQuery.noConflict();`还原之前的`$`的功能，现在只能用`jQuery`使用jQuery了

### 选择器

选择器是jQuery的核心。一个选择器写出来类似`$('#dom-id')`

```html
<!-- 讲解的HTML结构 -->
<ul class="lang">
    <li class="js dy">JavaScript</li>
    <li class="dy">Python</li>
    <li id="swift">Swift</li>
    <li class="dy">Scheme</li>
    <li name="haskell">Haskell</li>
</ul>
```

* **jQuery对象**

  * jQuery用**`$`查找到的不是Dom元素，而是jQuery对象**。它是一个**对象类似数组**

    * `[<div id="abc">...</div>]`
    * `[]`

  * jQuery的选择器不会返回`undefined`或者`null`，这样的好处是你不必在下一行判断`if (div === undefined)`

  * jQuery对象和DOM对象之间可以互相转化

    ```javascript
    var div = $('#abc'); // jQuery对象
    var divDom = div.get(0); // 假设存在div，获取第1个DOM元素
    // var divDom = div[0];
    var another = $(divDom); // 重新把DOM包装为jQuery对象,然后再使用
    ```

  * **不需要获取DOM对象，直接使用jQuery对象更加方便**，可以简单地调用`$(aDomObject)`把它变成jQuery对象，这样就可以方便地使用jQuery的API了。

  * *注意*，jQuery对象的所有方法都返回一个jQuery对象（可能是新的也可能是自身），这样我们可以进行链式调用，非常方便。

  * 批量操作，返回的jQuery对象可能代表一组节点，此时应用css可以批量操作

* 参考 css选择器，用的比较多的就是 id选择器，类选择器 层次选择器（上下文）子选择器


* jQuery自带的一些

  * :input`：可以选择``，``，``和``；
  * `:file`：可以选择``，和`input[type=file]`一样；
  * `:checkbox`：可以选择复选框，和`input[type=checkbox]`一样；
  * `:radio`：可以选择单选框，和`input[type=radio]`一样；
  * `:focus`：可以选择当前输入焦点的元素，例如把光标放到一个``上，用`$('input:focus')`就可以选出；
  * `:checked`：选择当前勾上的单选框和复选框，用这个选择器可以立刻获得用户选择的项目，如`$('input[type=radio]:checked')`；
  * `:enabled`：可以选择可以正常输入的``、``等，也就是没有灰掉的输入；
  * `:disabled`：和`:enabled`正好相反，选择那些不能输入的。

  此外，jQuery还有很多有用的选择器，例如，选出可见的或隐藏的元素：

  ```javascript
  $('div:visible'); // 所有可见的div
  $('div:hidden'); // 所有隐藏的div
  ```

* **查找和过滤**，jQuery对象的方法

  * `find()`查找子节点，参数是任意的选择器

  * `parent()`当前节点开始向上查找,可以传参数

  * `next()`和`prev()` 同级搜索，可以传参数

    ```javascript
    var ul = $('ul.lang'); // 获得<ul>
    var dy = ul.find('.dy'); // 获得JavaScript, Python, Scheme
    var swf = $('#swift'); // 获得Swift
    var parent = swf.parent(); // 获得Swift的上层节点<ul>
    var a = swf.parent('div.red'); // 从Swift的父节点开始向上查找，直到找到某个符合条件的节点并返回

    ```

  * `filter()`方法可以过滤掉不符合选择器条件的节点

  * `map()`方法把一个jQuery对象包含的若干DOM节点转化为其他对象

  * `first()`、`last()`和`slice()`方法可以返回一个新的jQuery对象，把不需要的DOM节点过滤了。

    ```javascript
    var langs = $('ul.lang li'); 
    var a = langs.filter('.dy'); // 用选择器过滤
    langs.filter(function () { // 用函数过滤
        return this.innerHTML.indexOf('S') === 0; // 返回S开头的节点
    });
    // 把jQuery对象，转换成，字符串数组
    var arr = langs.map(function () {
        return this.innerHTML;
    }).get(); //['JavaScript', 'Python', 'Swift', 'Scheme', 'Haskell']

    var js = langs.first(); // 获取第一个
    var sub = langs.slice(2, 4);  // 获取2--4个
    ```

### 操作dom

* text&&html

  * `selector.text()`获取节点的文本，传入参数表示设置
  * `selector.html()` 原始HTML文本，传入参数表示设置

* 修改CSS

  * 批量操作的特点
  * `css('name', 'value')`方法，作用于DOM节点的`style`属性，具有最高优先级。

  ```javascript
  var div = $('#test-div');
  div.css('color'); // '#000033', 获取CSS属性
  div.css('color', '#336699'); // 设置CSS属性
  div.css('color', ''); // 清除CSS属性

  // 常用操作，通过添加class修改样式
  div.hasClass('highlight'); // false， class是否包含highlight
  div.addClass('highlight'); // 添加highlight这个class
  div.removeClass('highlight'); // 删除highlight这个class
  ```

* 隐藏显示元素

  * 可以设置CSS的`display`属性为`none`，利用`css()`方法就可以实现，不过，要**显示这个DOM就需要恢复原有的`display`属性**，这就得先记下来原有的`display`属性到底是`block`还是`inline`还是别的值。
  * `show()`和`hide()`方法，更方便

* 获取DOM信息

  * 获取信息

    ```javascript
    / 浏览器可视窗口大小:
    $(window).width(); // 800
    $(window).height(); // 600

    // HTML文档大小:
    $(document).width(); // 800
    $(document).height(); // 3500

    // 某个div的大小:
    var div = $('#test-div');
    div.width(); // 600
    div.height(); // 300
    div.width(400); // 设置CSS属性 width: 400px，是否生效要看CSS是否有效
    div.height('200px'); // 设置CSS属性 height: 200px，是否生效要看CSS是否有效
    ```

  * 修改属性：`attr()`和`removeAttr()`方法用于操作DOM节点的属性

    ```javascript
    var div = $('#test-div');
    div.attr('data'); // undefined, 属性不存在
    div.attr('name'); // 'Test'
    div.attr('name', 'Hello'); // div的name属性变为'Hello'
    div.removeAttr('name'); // 删除name属性
    div.attr('name'); // undefined
    ```

    * prop()与attr()对于属性`checked`处理有所不同

      ```javascript
      var radio = $('#test-radio');
      radio.attr('checked'); // 'checked'
      radio.prop('checked'); // true
      ```

      > HTML5规定有一种属性在DOM节点中可以没有值，只有出现与不出现两种,checked,selected

    * 建议`is(':selected')`，判断属性

* 表单操作

  * `val()`：统一的方法获取和设置对应的`value`属性，不论input的类型。

* 操作DOM结构

  * 添加

    * `html()`这种暴力方法

    * `append()`、`prepend()`方法，添加到子节点

    * `after()`、`before()`方法，同级节点间添加

      ```javascript
      var ul = $('#test-div>ul');
      ul.append('<li><span>Haskell</span></li>');
      // 在第二个位置添加
      var js = $('#test-div>ul>li:first-child');
      js.after('<li><span>Lua</span></li>');
      ```

  * 删除
    * 用`remove()`方法

      ```javascript
      var li = $('#test-div>ul>li');
      li.remove(); // 所有<li>全被删除
      ```

### 事件

因为JavaScript在浏览器中以单线程模式运行，页面加载后，一旦页面上所有的JavaScript代码被执行完后，就只能依赖触发事件来执行JavaScript代码。

* 绑定click事件

  ```javascript
  var a = $('#test-link');
  a.on('click', function () {
      alert('Hello!');
  });
  // 推荐
  a.click(function () {
      alert('Hello!');
  });
  ```

* 常见事件

  * 鼠标
    * click: 鼠标单击时触发；
    * dblclick：鼠标双击时触发；
    * mouseenter：鼠标进入时触发；
    * mouseleave：鼠标移出时触发；
    * mousemove：鼠标在DOM内部移动时触发；
    * hover：鼠标进入和退出时触发两个函数，相当于mouseenter加上mouseleave
  * 键盘：作用在当前焦点的DOM上，通常是`<input>`和`<textarea>`。
    * keydown：键盘按下时触发；
    * keyup：键盘松开时触发；
    * keypress：按一次键后触发。

* 事件回调参数：`mousemove`和`keypress`，我们需要获取鼠标位置和按键的值

  ```javascript
  $(function () {
      $('#testMouseMoveDiv').mousemove(function (e) {
          $('#testMouseMoveSpan').text('pageX = ' + e.pageX + ', pageY = ' + e.pageY);
      });
  });
  ```

* 取消绑定

  * `off('click', function)`,**function必须是同一个引用**

    ```javascript
    function hello() {
        alert('hello!');
    }

    a.click(hello); // 绑定事件

    // 10秒钟后解除绑定:
    setTimeout(function () {
        a.off('click', hello);
    }, 10000);
    ```

  * 无参数调用`off()`一次性移除已绑定的所有类型的事件处理函数。

* 代码触发事件：当用户在文本框中输入时，就会触发`change`事件。但是，如果用JavaScript代码去改动文本框的值，将**不会**触发`change`事件

  * `input.change()`  `input.trigger('change')`,手动触发事件，可以减少代码。。

    ```javascript
    var input = $('#test-input');
    input.val('change it!');
    // input.change()相当于input.trigger('change')，它是trigger()方法的简写。
    input.change(); // 触发change事件
    ```

* 其他事件

  * focus：当DOM获得焦点时触发；
  * blur：当DOM失去焦点时触发；
  * change：当`<input>`、`<select>`或`<textarea>`的内容改变时触发；
  * submit：当`<form>`提交时触发；
  * ready：当页面被载入并且DOM树完成初始化后触发。

* **ready:绑定事件的时机**：**如果DOM没有加载完，绑定事件是无效的！**初始化代码必须放到`document`对象的`ready`事件中，保证DOM已完成初始化：

  ```html
  <html>
  <head>
      <script>
          $(document).on('ready', function () {
              $('#testForm).on('submit', function () {
                  alert('submit!');
              });
          });
       	 // 简化:
          $(document).ready(function () { 
      	 $('#testForm).submit(function () {
          		alert('submit!');
      	});
        	// 再简化:
        	$(function () {
      		// init...
  		});
  });
  });
      </script>
  </head>
  <body>
      <form id="testForm">
          ...
      </form>
  </body>
  ```

  * 反复绑定事件处理函数，它们会依次执行

### 动画

* jQuery内置

  * `show(time)` `hide(time)` `toggle(time)`加时间就是动画,效果从左上角变换。
  * `slideUp(time)`和`slideDown(time)` `slideToggle(time)`是在垂直方向逐渐展开或收缩的。
  * `fadeIn(time)` `fadeOut(time)` `fadeToggle(time)`

* 自定义动画

  * `animate()`

    ```javascript
    var div = $('#test-animate');
    div.animate({
        opacity: 0.25,
        width: '256px',
        height: '256px'
    }, 3000, function () {// 在3秒钟内CSS过渡到设定值
        console.log('动画已结束');
        // 恢复至初始状态:
        $(this).css('opacity', '1.0').css('width', '128px').css('height', '128px');
    }); 
    ```

  * 串行动画：`delay()`方法还可以实现暂停

    ```javascript
    var div = $('#test-animates');
    // 动画效果：slideDown - 暂停 - 放大 - 暂停 - 缩小
    div.slideDown(2000)
       .delay(1000)
       .animate({
           width: '256px',
           height: '256px'
       }, 2000)
       .delay(1000)
       .animate({
           width: '128px',
           height: '128px'
       }, 2000);
    }
    ```

  * 细节问题：

    * 没有效果：因为jQuery动画的原理是逐渐改变CSS的值，如`height`从`100px`逐渐变为`0`。但是很多不是block性质的DOM元素，对它们设置`height`根本就不起作用，所以动画也就没有效果。
    * jQuery也没有实现对`background-color`的动画效果，用`animate()`设置`background-color`也没有效果。这种情况下可以使用CSS3的`transition`实现动画效果。

### AJAX

* jQuery在全局对象`jQuery`（也就是`$`）**静态绑定**了`ajax()`函数，可以处理AJAX请求。`ajax(url, settings)`函数需要接收一个URL和一个可选的`settings`对象，常用的选项如下：
  * async：是否异步执行AJAX请求，默认为`true`，千万不要指定为`false`；
  * method：发送的Method，缺省为`'GET'`，可指定为`'POST'`、`'PUT'`等；
  * contentType：发送POST请求的格式，默认值为`'application/x-www-form-urlencoded; charset=UTF-8'`，也可以指定为`text/plain`、`application/json`；
  * data：发送的数据，可以是字符串、数组或object。如果是GET请求，data将被转换成query附加到URL上，如果是POST请求，根据contentType把data序列化成合适的格式；
  * headers：发送的额外的HTTP头，必须是一个object；
  * dataType：接收的数据格式，可以指定为`'html'`、`'xml'`、`'json'`、`'text'`等，缺省情况下根据响应的`Content-Type`猜测。


* 发送&&接收

  ```javascript
  var jqxhr = $.ajax('/api/categories', {
      dataType: 'json'
  }).done(function (data) {
      console.log('成功, 收到的数据: ' + JSON.stringify(data));
  }).fail(function (xhr, status) {
      console.log('失败: ' + xhr.status + ', 原因: ' + status);
  }).always(function () {
      console.log('请求完成: 无论成功或失败都会调用');
  });
  ```

* 辅助方法

  ```javascript
  // get
  var jqxhr = $.get('/path/to/resource', {
      name: 'Bob Lee',
      check: 1
  });
  // post
  var jqxhr = $.post('/path/to/resource', {
      name: 'Bob Lee',
      check: 1
  });
  //get 自动解析为JSON对象
  var jqxhr = $.getJSON('/path/to/resource', {
      name: 'Bob Lee',
      check: 1
  }).done(function (data) {
      // data已经被解析为JSON对象了
  });
  ```

* jsonp：在`ajax()`中设置`jsonp: 'callback'`，让jQuery实现JSONP跨域加载数据。

### jQuery插件

* 扩展jQuery对象（$('p').xxx）

  * 扩展`$.fn`对象（本质添加原型方法）
  * 插件函数最后要`return this;`以支持链式调用
  * 插件函数要有默认值，绑定在`$.fn..defaults`上；（推荐）
  * 用户在调用时可传入设定值以便覆盖默认值。（推荐）

  ```javascript
  $.fn.highlight = function (options) {
      // 用短路运算符设置运算符
      // var bgcolor = options && options.backgroundColor || '#fffceb';
      // var color = options && options.color || '#d85030';
      // 合并默认值和用户设定值:$.extend的功能后面向前合并
      var opts = $.extend({}, $.fn.highlight.defaults, options);
      this.css('backgroundColor', opts.backgroundColor).css('color', opts.color);
      return this;
  }

  // 设定默认值:
  $.fn.highlight.defaults = {
      color: '#d85030',
      backgroundColor: '#fff8de'
  }
  ```

  * 针对特定元素的扩展：在函数内过滤一下

* 静态扩展（$.xxx）

  * extend？


## underscore

* 一个库提供一些函数式方法兼容各种浏览器，入map， filter， some，every

* 以标志符`_`开头调用，如`_.map()`

* 方法有：
  * Collection：Array和Object，暂不支持Map和Set

    * `_.map`  `_.mapObject`
    * `_.filter()`
    * `_.every()` `_.some()`
    * `max()`和`min()`
    * `groupBy()`把集合的元素按照key归类，key由传入的函数返回
    * `shuffle()`用洗牌算法随机打乱一个集合
    * `sample()`随机选择一个或多个元素

  * Arrays

    * `_.first()` `_.last()`取第一个和最后一个元素

    * `_.flatten()`接收一个`Array`，无论这个`Array`里面嵌套了多少个`Array`，最后都把它们变成一个一维数组

    * `_.zip()`把两个或多个数组的所有元素按索引对齐,`unzip()`则是反过

      ```javascript
      var names = ['Adam', 'Lisa', 'Bart'];
      var scores = [85, 92, 59];
      _.zip(names, scores);
      // [['Adam', 85], ['Lisa', 92], ['Bart', 59]]
      ```

    * `_.object()`,为啥不把名字和分数直接对应成Object呢？`object()`函数就是干这个的.

    * `_.range()`快速生成一个序列

  * Function

    * `_.bind()` 类似bind

    * `_.partial() `为一个函数创建偏函数,固定一个参数

    * `_.memoize()`缓存一个函数的运算结果，如果参数不变直接返回结果。

      ```javascript
      function factorial(n) {
          console.log('start calculate ' + n + '!...');
          var s = 1, i = n;
          while (i > 1) {
              s = s * i;
              i --;
          }
          console.log(n + '! = ' + s);
          return s;
      }

      factorial(10); // 3628800
      ```

    * `_.once()`保证某个函数执行且仅执行一次

      ```javascript
      var register = _.once(function () {
          alert('Register ok!');
      });

      register();
      register();
      ```

    * `_.delay()`可以让一个函数延迟执行

      ```javascript
      var log = _.bind(console.log, console);
      _.delay(log, 2000, 'Hello,', 'world!');
      ```

  * Object

    * `_.keys()`返回object自身所有的key，但不包含从原型链继承下来的
    * `_.allKeys()`除了object自身的key，还包含从原型链继承下来的
    * `_.values()`返回object自身但不包含原型链继承的所有值：
    * `_.invert()`把object的每个key-value来个交换，key变成value，value变成key
    * `_.extend()`**把多个object的key-value合并到第一个object并返回**。`extendOwn()`和`extend()`类似，但获取属性时忽略从原型链继承下来的属性
    * `_.clone()`：**浅复制**一个object对象
    * `_.isEqual()`**对两个object进行深度比较**

* 链式调用：underscore提供了把对象包装成能进行链式调用的方法

  * `chain()`函数

    ```javascript
    _.chain([1, 4, 9, 16, 25])
     .map(Math.sqrt)
     .filter(x => x % 2 === 1)
     .value();
    ```

    ​






## 模板代码

* 网络访问

  ```javascript
  // 工具方法：ajax函数将返回Promise对象:
  function ajax(method, url, data) {
      var request = new XMLHttpRequest();
      return new Promise(function (resolve, reject) {
          request.onreadystatechange = function () {
              if (request.readyState === 4) {
                  if (request.status === 200) {
                      resolve(request.responseText);
                  } else {
                      reject(request.status);
                  }
              }
          };
          request.open(method, url);
          request.send(data);
      });
  }
  // 注意访问地址不要跨域
  var p = ajax('GET', 'url address');
  p.then(function (text) { // 如果AJAX成功，获得响应内容
      console.log(text);
  }).catch(function (status) { // 如果AJAX失败，获得响应代码
      console.log('ERROR: ' + status);
  });
  ```

  ​
