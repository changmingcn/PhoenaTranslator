# PhoenaTranslator

将英文 PDF 和 EPUB 翻译为中文的 Web 工具，重点照顾技术文档、论文和电子书的排版质量。

## 主要特点

- 支持英文 PDF 与 EPUB 翻译为中文
- 尽量保留 PDF 的原始版式、数学公式和表格结构
- 对数学内容采用保守策略：当公式过于密集或结构过于复杂、无法可靠翻译时，会保留原英文，避免错误改写或排版损坏
- 通过浏览器上传文件、查看进度并下载结果
- 使用 OpenAI 兼容的大语言模型 API，安装时填写 API Key、Base URL 和模型名

## 快速安装

从 [Releases](https://github.com/changmingcn/PhoenaTranslator/releases/latest) 下载安装包和 SHA-256 校验文件，然后在服务器执行：

```bash
sha256sum -c PhoenaTranslator-20260722-clean-installer.tar.gz.sha256
tar -xzf PhoenaTranslator-20260722-clean-installer.tar.gz
cd PhoenaTranslator-20260722-clean-installer
./install.sh --verify-only   # 可选预检
sudo ./install.sh
```

安装器会依次询问：

1. 大语言模型 API Key（隐藏输入）
2. OpenAI 兼容 API 的 Base URL
3. 模型名（直接回车使用默认值）

安装完成后访问：

```text
http://服务器IP/
```

安装过程本身不会调用翻译 API。API Key 保存于 `/etc/phoena-translator/translator.env`，权限为 `0600`。

## 系统要求

- 已确认：Debian 13、x86_64、Python 3.13
- 安装器支持：Debian 12/13、Ubuntu 24.04
- 需要 root 权限，以及访问 apt、PyPI 和所选 LLM API 的网络

## 注意事项

- 翻译结果由所选模型和原文复杂度共同决定，建议对重要文档人工复核
- 默认部署只提供 HTTP，且网页没有内置登录限制；如暴露到公网，请自行配置 TLS、防火墙或外部身份验证
- 安装包不包含旧任务、原始文档、翻译结果、日志、缓存或 API Key

当前版本：`2026.07.22-r1`
