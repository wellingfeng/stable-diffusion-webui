# Stable Diffusion WebUI Chinese 1021

[![](https://img.shields.io/badge/Telegram-B站主页-purple)](https://space.bilibili.com/22970812)
[![](https://img.shields.io/badge/Telegram-交流群-purple)](https://jq.qq.com/?_wv=1027&k=wEbRm1eU)

## 解释说明

此版本为[Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)原版汉化页面。

当前版本为1021，即基于官方webui本地化模板10月21日之前的最新版本进行的汉化。

10月16日以后官方推出了本地化模板功能，汉化跟之前的方式不同了，不再需要更改文件来汉化。
所以WebUI的版本至少是10月16日以后的WebUI。
而且也打上汉化以后也可以正常`git pull`，最多就是汉化不完全。
因为是根据官方的本地化模板进行汉化，所以汉化基本不会出现BUG，使用方法视频看我B站发的视频。

BUG提示：目前鼠标悬停提示仍显示英文不是汉化的问题，是WebUI的本地化模块功能还有点BUG。

⚠️汉化免费，请注意欺诈

### 在SD WebUI的localizations目录安装

1.更新到最新版本的stable diffusion webui！！

2.在stable diffusion webui目录下使用`git clone https://github.com/VinsonLaro/stable-diffusion-webui-chinese localizations`

3.打开stable diffusion webui

4.然后进到设置Setting

5.找到User interface，下面有一个Localization (requires restart)

6.把选项切换到Chinese-English

7.然后划到最上面，点几下Apply setting

8.再划到最下面，点击

Restart Gradio and Refresh components (Custom Scripts, ui.py, js and css only)

9.这时候界面会重新加载，重新加载完了以后就汉化好了

### 后继汉化更新

1.在localizations目录下使用`git pull`


### 手动安装汉化

1.更新到最新版本的stable diffusion webui！！

2.把Chinese-English.json和Chinese-All.json文件放到stable-diffusion-webui\localizations目录下

3.打开stable diffusion webui

4.然后进到设置Setting

5.找到User interface，下面有一个Localization (requires restart)

6.把选项切换到Chinese-English

7.然后划到最上面，点几下Apply setting

8.再划到最下面，点击

Restart Gradio and Refresh components (Custom Scripts, ui.py, js and css only)

9.这时候界面会重新加载，重新加载完了以后就汉化好了

---

如有侵权，请私信我删除。




