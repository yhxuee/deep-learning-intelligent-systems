---
title: 机器可读的书
description: llms.txt、llms-full.txt、atlas.json 与 MCP 服务器，让学习者自己的 AI 智能体能直接取得有版本、附引用的内容。
sidebar: { label: 机器可读的书, order: 3 }
---

复制提示词很脆弱：学习者会漏掉背景，模型也会漂移。所以本书也以 AI 智能体能**直接**读取的形式发布自己，让回答能以经审查的正文为依据并引用它。

| 端点 | 内容 | 标准 |
|---|---|---|
| [`/llms.txt`](/llms.txt) | 单元、数学附录、Atlas 节点与 AI 导师页面的索引，附一行摘要 | [llms.txt 提案](/zh-cn/references/#llmstxt-2024) |
| [`/llms-full.txt`](/llms-full.txt) | 核心教材、附录、Atlas 与 AI 导师页面的完整 Markdown | — |
| [`/atlas.json`](/atlas.json) | 每个单元（含其各小节页面）、附录与节点，附带类型的联系、掌握层次、参考文献与原始 Markdown | 格式 `dlis-atlas/1.3`（1.2 加入小节页面；1.3 加入 `appendices`） |
| `mcp/atlas_server.py` | 在 `atlas.json` 上提供搜索与检索工具的 MCP 服务器 | [Model Context Protocol](/zh-cn/references/#mcp-2024) |

目前机器可读端点提供的是英文原文；中文提示词可通过 MCP 工具 `get_prompt` 取得：繁体中文加上 `lang="zh-tw"`，简体中文加上 `lang="zh-cn"`。

## 使用 MCP 服务器

服务器是一个**不需任何依赖包**（只用 Python 标准库）的单一文件。它通过 stdio 使用 MCP 协定，从已发布的网站读取 `atlas.json`，或以 `--source dist/atlas.json` 读取本地构建的版本。

```bash
git clone https://github.com/yhxuee/deep-learning-intelligent-systems
python3 deep-learning-intelligent-systems/mcp/atlas_server.py --self-test
```

在任何支持 MCP 的客户端注册它。典型的客户端设定：

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
| `search_atlas(query, limit?)` | 以关键字比对标题、摘要与正文，为单元、小节、附录与节点排序 |
| `get_node(id)` | 完整的 Atlas 节点：不变量、已知数值、常见错误、带类型的联系、参考文献 |
| `get_unit(number)` | 一个核心单元：问题、锚定范例、主干项目进度、检查点，以及十个小节的全文 |
| `get_appendix(letter)` | 一篇数学附录（`A`–`H`）的全文 |
| `get_prompt(name, lang?)` | AI 导师提示词（`base-policy`、`tutor`、`critic`、`collaborator`、`practice` 或单元提示词）；`lang` 可为 `en`、`zh-tw` 或 `zh-cn` |
| `list_prerequisites(id)` | 一个节点的递移先修知识链，依学习顺序排列 |

服务器只返回内容；它从不写入本书，也从不评分。检索不等于正确：工具结果包含节点的 `last_reviewed` 日期与参考文献，让代理能引用它们，而基础政策要求代理把其他内容标记为自己的一般知识。
