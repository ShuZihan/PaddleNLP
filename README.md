# Paddle / ATB 机制图解

以 Llama FFN 为例，解释 Paddle 逐算子调用与 ATB 子图调用的执行顺序和内存管理。

网站：https://rawcdn.githack.com/ShuZihan/PaddleNLP/ae6f22b9cc8fa9ac5a0a5a307539202829532b07/index.html

页面为独立 HTML，包含 SVG 图解与交互，无需构建或安装依赖。页面右上角可下载 HTML 文件，保存后可离线阅读与操作图解。

托管方式：文件保存在本仓库的 `atb-guide-site` 分支，网页通过 githack 提供 HTTPS 访问。链接固定到对应文件版本。

FFN 张量并行与权重拼接审阅页：https://rawcdn.githack.com/ShuZihan/PaddleNLP/ae6f22b9cc8fa9ac5a0a5a307539202829532b07/ffn-review.html
