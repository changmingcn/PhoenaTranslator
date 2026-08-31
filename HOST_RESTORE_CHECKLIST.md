# 服务器重装恢复清单

本安装包只备份 PhoenaTranslator 的可重装代码与部署模板，不备份凭据、TLS 身份或业务数据。

## 重装服务器前

- 确认 Google Drive 中的 `.tar.gz` 和 `.sha256` 均已上传，并在另一端重新下载后通过 SHA-256 校验。
- 如需沿用现有 API 配置，用独立的加密或受控方式备份
  `/etc/phoena-translator/translator.env`；不要把它放进本代码包。
- 如需保留域名和 HTTPS，独立备份 nginx 的代理配置、DNS 记录以及
  `/etc/proxy-stack/certs/` 下的证书和私钥。
- 如需保留旧任务或翻译结果，独立备份
  `/var/lib/phoena-translator/{uploads,outputs,progress}`；本包不会携带这些内容。
- 记录防火墙、云安全组、域名解析和自定义中文字体路径。

## 新服务器准备

- 使用 Debian 12/13 或 Ubuntu 24.04，准备 root 权限。
- 保证能访问 apt 和 PyPI；本包没有离线 wheelhouse，也不携带旧虚拟环境。
- 若需要 PDF 中文输出，安装可用的 TrueType 中文字体；需要英文 OCR 时确认
  Tesseract 英文数据可用。

## 安装与校验

```bash
sha256sum -c PhoenaTranslator-20260813-cross-page-boundary-guard-reinstall.tar.gz.sha256
tar -xzf PhoenaTranslator-20260813-cross-page-boundary-guard-reinstall.tar.gz
cd PhoenaTranslator-20260813-cross-page-boundary-guard-reinstall
./install.sh --verify-only
sudo ./install.sh
```

安装器会要求输入 API Key、兼容 API 的 Base URL 和模型名。安装过程不会调用翻译 API。

## 安装后

- 如有需要，再恢复 TLS/代理、DNS、防火墙、字体与业务数据；恢复前先核对所有者和权限。
- 检查服务：

```bash
systemctl status phoena-translator --no-pager
systemctl status nginx --no-pager
curl -fsS http://127.0.0.1:8501/api/health
curl -fsS http://127.0.0.1/api/health
```

- 用一份非敏感的小型测试文档验证上传、翻译进度和下载流程。
- 确认 `/etc/phoena-translator/translator.env` 为 `root:root`、权限 `0600`。
