---
title: 機器可讀的書
description: llms.txt、llms-full.txt、atlas.json 與 MCP 伺服器，讓學習者自己的 AI 代理能直接取得有版本、附引用的內容。
sidebar: { label: 機器可讀的書, order: 3 }
---

複製提示詞很脆弱：學習者會漏掉脈絡，模型也會漂移。所以本書也以 AI 代理能**直接**讀取的形式發布自己，讓回答能以經審查的正文為依據並引用它。

| 端點 | 內容 | 標準 |
|---|---|---|
| [`/llms.txt`](/llms.txt) | 單元、Atlas 節點與 AI 導師頁面的索引，附一行摘要 | [llms.txt 提案](/zh-tw/references/#llmstxt-2024) |
| [`/llms-full.txt`](/llms-full.txt) | 核心教材、Atlas 與 AI 導師頁面的完整 Markdown | — |
| [`/atlas.json`](/atlas.json) | 每個單元（含其各小節頁面）與節點，附有型別的連結、掌握層次、參考文獻與原始 Markdown | 格式 `dlis-atlas/1.1` |
| `mcp/atlas_server.py` | 在 `atlas.json` 上提供搜尋與檢索工具的 MCP 伺服器 | [Model Context Protocol](/zh-tw/references/#mcp-2024) |

目前機器可讀端點提供的是英文原文；繁體中文的提示詞可透過 MCP 工具 `get_prompt` 加上 `lang="zh-tw"` 取得。

## 使用 MCP 伺服器

伺服器是一個**不需任何相依套件**（只用 Python 標準函式庫）的單一檔案。它透過 stdio 使用 MCP 協定，從已發布的網站讀取 `atlas.json`，或以 `--source dist/atlas.json` 讀取本機建置的版本。

```bash
git clone https://github.com/yhxuee/deep-learning-intelligent-systems
python3 deep-learning-intelligent-systems/mcp/atlas_server.py --self-test
```

在任何支援 MCP 的用戶端註冊它。典型的用戶端設定：

```json
{
  "mcpServers": {
    "dlis-atlas": {
      "command": "python3",
      "args": ["/path/to/deep-learning-intelligent-systems/mcp/atlas_server.py"]
    }
  }
}
```

提供的工具：

| 工具 | 用途 |
|---|---|
| `search_atlas(query, limit?)` | 以關鍵字比對標題、摘要與正文，為單元、小節與節點排序 |
| `get_node(id)` | 完整的 Atlas 節點：不變量、已知數值、常見錯誤、有型別的連結、參考文獻 |
| `get_unit(number)` | 一個核心單元：問題、錨定範例、主幹專案進度、檢核點，以及十個小節的全文 |
| `get_prompt(name, lang?)` | AI 導師提示詞（`base-policy`、`tutor`、`critic`、`collaborator`、`practice` 或單元提示詞）；`lang` 可為 `en` 或 `zh-tw` |
| `list_prerequisites(id)` | 一個節點的遞移先備知識鏈，依學習順序排列 |

伺服器只回傳內容；它從不寫入本書，也從不評分。檢索不等於正確：工具結果包含節點的 `last_reviewed` 日期與參考文獻，讓代理能引用它們，而基礎政策要求代理把其他內容標示為自己的一般知識。
