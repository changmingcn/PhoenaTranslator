# PhoenaTranslator 干净重装包

版本：`2026.08.13-cross-page-boundary-guard`

这是当前已修正翻译程序的恢复安装版。它只包含 53 个源码文件（51 个运行时文件 + 2 个测试及测试依赖文件）和安装元数据，不包含任何旧任务、PDF/EPUB、翻译结果、缓存、日志、虚拟环境或密钥。

## 相对上一版（2026.08.03-inline-render-fallback）的变化

本包以生产发布
`release-2026.08.03-inline-render-fallback-20260812T092040Z-16943`
的有效载荷为基线，修正 PDF 同页段落被误当作跨页续句的问题；文件集合未变（仍是
53 个）。

- **黑体首行不再把后续常规体误搬到上一页**：跨页吸收前新增同页归属检查；若小写开头
  的候选行紧接在同字号、同栏且句意未完的本页行之后，则保留在本页。该检查不要求两行
  的粗细相同，覆盖“首句/首行局部黑体、后续行常规体”的排版。
- **表格边界不再吞掉邻近正文首行**：表格横向容差保持 12 pt，纵向容差从 12 pt 收紧为
  3 pt，避免图表下方正文被错误标成表格元素并拆段。
- **完整句后的弱跨页证据更保守**：只有位于下一页顶部区域的小写开头文本才可作为弱
  续句候选；审计阶段也会拒绝存在同页前导行的错误搬移记录。
- **旧错误缓存强制失效**：PDF 版面语义版本提升至 v38，重新翻译时不会复用 v37 中已经
  被错误拆分和跨页搬移的页面缓存。

三个实际问题 PDF 的确定性抽取复核均已通过；自动化测试从 148 增至 153。

## 支持范围

- 已在当前生产基线确认：Debian 13、x86_64、Python 3.13。
- 安装器允许：Debian 12/13、Ubuntu 24.04；若系统尚无 Python，正式安装会先从 apt 引导安装 Python 3，再要求版本不低于 3.11。
- Debian 12 和 Ubuntu 24.04 是依据包元数据与安装守卫支持，未在本次环境中进行完整虚拟机启动测试。
- Ubuntu 22.04 的默认 Python 3.10 不支持当前锁定的 NumPy 版本。
- 目标主机需要 root 权限，以及访问 apt、PyPI 和翻译 LLM API 的网络。

## 安装

先在下载目录校验压缩包（校验文件与压缩包应在同一目录）：

```bash
sha256sum -c PhoenaTranslator-20260831-identifier-row-fastpath-reinstall.tar.gz.sha256
tar -xzf PhoenaTranslator-20260831-identifier-row-fastpath-reinstall.tar.gz
cd PhoenaTranslator-20260831-identifier-row-fastpath-reinstall
# 可选的完整预检（需要系统已装 python3）
./install.sh --verify-only
sudo ./install.sh
```

安装器会依次询问：

1. 翻译 LLM 的 API Key（隐藏输入）；
2. OpenAI 兼容 API 的 Base URL；
3. 模型名（直接回车采用 `deepseek-v4-flash`）。

安装过程不会调用翻译 LLM，因此安装本身不产生 LLM API 费用。API Key 不会出现在命令行参数、终端回显、普通日志或安装包中。正常完成后的持久副本位于 `/etc/phoena-translator/translator.env`，权限为 root:root、`0600`；安装事务期间的临时副本只存在于 root-only 回滚目录。

完成后在浏览器访问：

```text
http://服务器IP/
```

通用 nginx 配置只提供 HTTP 和 `/api/` 反向代理，不配置域名、DNS 或 TLS。应用进程只监听 `127.0.0.1:8501`。网站没有管理员令牌或登录限制，能访问该地址的人都能使用；如暴露到公网，请自行配置防火墙、TLS 或外部访问控制。

## 安装后的路径

```text
/opt/phoena-translator/releases/       版本化程序、网页与虚拟环境
/opt/phoena-translator/current         原子切换的当前版本符号链接
/etc/phoena-translator/translator.env  API 与运行配置（0600）
/var/lib/phoena-translator              root:phoena-translator、0750 的受保护数据根
/var/lib/phoena-translator/uploads      新建空上传目录
/var/lib/phoena-translator/outputs      新建空输出目录
/var/lib/phoena-translator/progress     新建空进度目录
/var/log/phoena-translator              日志目录
/var/lib/phoena-translator-installer    root-only 安装锁、事务状态与回滚副本
```

安装器不会导入或清空既有任务数据目录。数据根目录本身永久由 root 控制，服务账户只能写入 `uploads`、`outputs`、`progress` 和日志目录；重复安装会先停稳旧服务并确认没有遗留的服务账户进程、锁定父目录，再以 `O_NOFOLLOW` 打开每个子目录并通过文件描述符修复权限，从而避免 root 权限操作跟随服务账户可替换的路径。重复运行时，它会保留上述数据目录，把新版本完整构建在独立目录，通过一个原子 `current` 符号链接切换，并保留上一版本目录供人工恢复。nginx 的静态根目录只包含 `index.html`，不会公开 Python 源码。

安装器直接锁定已验证为 root:root、`0700` 的状态目录，使用 `flock` 防止并发运行，并在 `/var/lib/phoena-translator-installer/transaction.state` 写入不含密钥的持久事务状态。systemd 服务与该事务状态联锁：只要事务未提交，即使机器重启也不能自行启动。切换后先在不连接 nginx 的备用 loopback 端口运行降权、禁用旧任务恢复且不加载 API Key 的单进程探针；探针随安装器意外死亡而由内核终止。探针和静态网页检查通过、事务状态被持久清除后，生产服务才会启动，应用本身通过 loopback 健康检查后才启动 nginx 对外提供 API。SIGINT、普通失败、SIGKILL 或断电后，再次运行同一安装器会先执行幂等恢复，再开始新安装。事务中的 API 配置临时文件和旧配置备份位于 root-only 回滚目录，正常成功后删除；恢复时会先清除事务状态，再尽力清理回滚副本，因此不会出现“状态尚在、恢复副本已删”的提交窗口。若恰在提交后的极短清理窗口断电，下一次运行会清理遗留目录。若文件或配置恢复无法完成，安装器会保留状态、不继续覆盖，并尝试且核验服务停止；若文件已经安全恢复而仅服务重新激活失败，则事务状态保持已提交、两项服务保持停止、root-only 回滚副本保留，并明确报错。

## 常用检查

```bash
systemctl status phoena-translator --no-pager
systemctl status nginx --no-pager
curl -fsS http://127.0.0.1/api/health
journalctl -u phoena-translator -n 100 --no-pager
```

若要更换 API Key、地址或模型，可重新运行 `sudo ./install.sh`。安装器会重新验包并重建虚拟环境，不会删除任务数据。

## 设计限制

- 依赖通过网络从 apt/PyPI 安装；本包没有携带体积很大的离线 wheelhouse。
- Python 依赖锁定版本，但没有内置 wheel 哈希或发布者签名；内部清单与外部压缩包 SHA-256 用于完整性校验，不构成独立发布者身份认证。
- `constraints.txt` 是 Python 3.13 上解析出的完整依赖闭包；其他允许系统仍会由 pip 进行兼容性判定。
- 程序由 gunicorn/systemd 启动；源码中的 `python app.py` 直接启动路径不是本发行版支持的服务入口。
- 若安装在完整验包前需要为最小化系统引导 Python，或在切换程序前失败，apt 已安装的软件包、专用系统账户和新建的空数据目录会保留；它们不含 API Key 或旧任务数据。程序、环境文件、systemd 与 nginx 的切换受持久事务恢复保护。
- 第三方依赖仍受各自许可证约束。

## 完整性

- `SOURCE_ALLOWLIST.txt`：53 个源码、测试及依赖清单相对路径。
- `MANIFEST.sha256`：安装目录中除该清单自身外的每个文件哈希。
- `verify_release.py`：确定性检查清单闭包、文件类型、排除项、源码语法和常见密钥形态。
- `SOURCE_IDENTITY.json`：记录生产发布路径、文件数、总字节数和有效载荷清单摘要。
- `HOST_RESTORE_CHECKLIST.md`：列出重装前需另行保存、重装后需恢复的主机级配置。
- 压缩包旁的 `.sha256`：校验整个下载文件。
