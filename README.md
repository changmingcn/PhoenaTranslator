# PhoenaTranslator

将英文 PDF 和 EPUB 翻译为中文的 Web 工具，重点照顾技术文档、论文和电子书的排版质量。

## 主要特点

- 支持英文 PDF 与 EPUB 翻译为中文
- 尽量保留 PDF 的原始版式、数学公式和表格结构
- 对数学内容采用保守策略：当公式过于密集或结构过于复杂、无法可靠翻译时，会保留原英文，避免错误改写或排版损坏
- 通过浏览器上传文件、查看进度并下载结果
- 使用 OpenAI 兼容的大语言模型 API，安装时填写 API Key、Base URL 和模型名
- 可选原生 Claude 后端：在 `/etc/phoena-translator/translator.env` 设 `TRANSLATOR_LLM_PROVIDER=claude`、`ANTHROPIC_API_KEY`，可选 `CLAUDE_MODEL`（默认 `claude-opus-5-5`，另支持 `claude-sonnet-5-5`、`claude-haiku-5-5`）；不设时仍走 DeepSeek

## 快速安装

从 [Releases](https://github.com/changmingcn/PhoenaTranslator/releases/latest) 下载安装包和 SHA-256 校验文件，然后在服务器执行：

```bash
sha256sum -c PhoenaTranslator-20261119b-telegram-notify-reinstall.tar.gz.sha256
tar -xzf PhoenaTranslator-20261119b-telegram-notify-reinstall.tar.gz
cd PhoenaTranslator-20261119b-telegram-notify-reinstall
./install.sh --verify-only   # 可选预检
sudo ./install.sh
```

安装器会依次询问：

1. 大语言模型 API Key（隐藏输入）
2. OpenAI 兼容 API 的 Base URL
3. 模型名（直接回车使用默认值）
4. 可选的 Telegram 通知：`Telegram bot token for notifications (Enter to skip)`（隐藏输入，直接回车跳过）；填了 token 再问 `Telegram chat id` 和 `Telegram: send a message when each file finishes? (yes/no) [no]`。`translator.env` 里已有 token 和 chat id 时不再问，只问最后一项，回车保持原设置

重装或升级时，旧 `translator.env` 里的 `LLM_*`、`OPENAI_*`、`CLAUDE_*`、`ANTHROPIC_*`、`TRANSLATOR_LLM_PROVIDER` 与 `TRANSLATOR_TELEGRAM_*` 行原样保留，其余行按问答重写。

安装完成后访问：

```text
http://服务器IP/
```

安装过程本身不会调用翻译 API。API Key 保存于 `/etc/phoena-translator/translator.env`，权限为 `0600`。

## 主力/备用 Key 自动切换（可选）

在 `translator.env` 设 `LLM_PRIMARY_API_KEY` 即进入双 Key 模式，此时 `TRANSLATOR_LLM_PROVIDER`、`ANTHROPIC_API_KEY`、`DEEPSEEK_API_KEY` 不再使用。不设时行为与原来完全一致。

| 变量 | 含义 |
|---|---|
| `LLM_PRIMARY_API_KEY` / `LLM_BACKUP_API_KEY` | 主力、备用 Key，可来自 Claude、OpenAI、DeepSeek 任一家，两把也可同属一家。只设备用、不设主力时忽略备用并记一条警告 |
| `LLM_PRIMARY_PROVIDER` / `LLM_BACKUP_PROVIDER` | 可选，`claude`、`openai` 或 `deepseek`。不设时按 Key 形状判断：`sk-ant-` 为 Claude，`sk-proj-`、`sk-svcacct-`、`sk-admin-` 为 OpenAI，`sk-` 加 32 位十六进制为 DeepSeek，其他 `sk-` 开头为 OpenAI，都不像时启动报错 |
| `LLM_FAILOVER_RETRY_PRIMARY_SECONDS` | 切到备用后多久再把下一个真实请求先发给主力，默认 `3600` 秒，不发探测请求 |
| `OPENAI_MODEL` / `OPENAI_BASE_URL` | OpenAI 模型与地址，默认 `gpt-6.2-sol`、`https://api.openai.com/v1`。Claude、DeepSeek 沿用 `CLAUDE_MODEL`、`DEEPSEEK_MODEL` 等原有变量 |
| `OPENAI_REASONING_EFFORT` | 默认 `low` |
| `OPENAI_PRICE_INPUT_USD_PER_M` / `OPENAI_PRICE_CACHED_INPUT_USD_PER_M` / `OPENAI_PRICE_OUTPUT_USD_PER_M` | 用量台账的计价，默认 `2` / `0.1` / `10`（`gpt-6.2-sol` 的价目，美元每百万 token；换模型时一并改） |

切换条件：HTTP 401、402、403，余额、账单或用量上限类 400（如 Anthropic “credit balance is too low”“specified API usage limits”），额度用尽的 429（如 OpenAI `insufficient_quota`）立即切换；普通 429 在原有重试次数用完后才切换。网络错误、超时、5xx/529、其他 400、404、拒答和截断都不切换，处理方式与原来相同；只有已切到备用后回试主力的那个请求例外，主力因任何原因没成功都改由备用当场接手，并重新计时。每个请求每把 Key 最多试一次，两把都不可用时报错并写明两边的状态码。每次切换在日志记一条以 `llm-failover:` 开头的 WARNING，用量台账 `deepseek_usage.jsonl` 的新记录带 `key_slot`（`primary`/`backup`）和 `provider`。

## Telegram 通知（可选）

在 `translator.env` 同时设 `TRANSLATOR_TELEGRAM_BOT_TOKEN` 和 `TRANSLATOR_TELEGRAM_CHAT_ID` 才会发消息，两项缺一不发。消息为纯文本，后台发送，超时 10 秒，不重试，发不出去只记一条 WARNING，不影响翻译，也不改任务状态；日志不含 token。

| 变量 | 含义 |
|---|---|
| `TRANSLATOR_TELEGRAM_BOT_TOKEN` / `TRANSLATOR_TELEGRAM_CHAT_ID` | 机器人 token 与接收的 chat id |
| `TRANSLATOR_TELEGRAM_KEY_ALERTS` | 默认开。双 Key 模式下 Key 状态变化时提醒：主力不可用、已改用备用（`switched`），切回主力（`restored`），两把 Key 都不可用（`all_dead`）。每次状态变化只发一条，不随请求数重复。设 `0` 关闭 |
| `TRANSLATOR_TELEGRAM_NOTIFY_DONE` | 默认关，设 `1` 后每个文件翻译结束发一条：成功为 `✅【标签】翻译成功：文件名`，下接任务记录里已有的页数、保留原文的页、用时；失败为 `❌【标签】翻译失败：文件名`，下一行 `失败原因：…`，能分辨的类别（两把 Key 都不可用、所有页退回原文、文件无法解析、任务被中断、程序内部错误）先说明类别，再接记录的错误，最多 300 字。每个任务只发一次，服务重启不会为已结束的任务重发；对已结束的 EPUB 再点“重试未通过段落”，重跑结束后会再发一条 |
| `TRANSLATOR_TELEGRAM_LABEL` | 消息里的【标签】，默认 `PhoenaTranslator` |
| `TRANSLATOR_TELEGRAM_MIN_INTERVAL_SECONDS` | 仅对 Key 提醒：同一种提醒的最短间隔，默认 `1800` 秒，间隔内压下的次数写进下一条同类提醒。完成通知不节流 |
| `TRANSLATOR_TELEGRAM_DISABLE` | 设 `1` 关闭全部 Telegram 消息 |

## 系统要求

- 已确认：Debian 13、x86_64、Python 3.13
- 安装器支持：Debian 12/13、Ubuntu 24.04
- 需要 root 权限，以及访问 apt、PyPI 和所选 LLM API 的网络

## 注意事项

- PDF 输出需要 TrueType 中文字体（.ttf）：字体大于 4MB 且无法子集化（如 CFF 风味 OTF）时任务会在启动阶段明确报错，可用 `TRANSLATOR_PDF_FONT_REGULAR` / `TRANSLATOR_PDF_FONT_BOLD` 指定字体文件；沿用 `~/fonts` 下 TTF 的现有部署无需任何改动
- 翻译结果由所选模型和原文复杂度共同决定，建议对重要文档人工复核
- 默认部署只提供 HTTP，且网页没有内置登录限制；如暴露到公网，请自行配置 TLS、防火墙或外部身份验证
- 安装包不包含旧任务、原始文档、翻译结果、日志、缓存或 API Key
- 重装前后的主机级恢复项目见 `HOST_RESTORE_CHECKLIST.md`
- `SOURCE_IDENTITY.json` 记录本包对应的生产发布、文件数和内容清单摘要

当前版本：`2026.11.19b-telegram-notify`
