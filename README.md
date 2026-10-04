# Paddle / ATB 机制图解

以 Llama FFN 为例，解释 Paddle 逐算子调用与 ATB 子图调用的执行顺序和内存管理。

网站：https://rawcdn.githack.com/ShuZihan/PaddleNLP/6b481388130c456b239a8980ee4c421326f9d63e/index.html

页面为独立 HTML，包含 SVG 图解与交互，无需构建或安装依赖。页面右上角可下载 HTML 文件，保存后可离线阅读与操作图解。

托管方式：文件保存在本仓库的 `atb-guide-site` 分支，网页通过 githack 提供 HTTPS 访问。链接固定到对应文件版本。
