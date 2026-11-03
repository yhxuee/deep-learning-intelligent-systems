"""The Simplified Chinese edition is generated from the Traditional Chinese one (scripts/zh_cn.py).
These tests keep the two in sync and guard the conversions that make the mainland text native."""
import importlib.util
import pathlib
import re

import pytest

pytest.importorskip("opencc")
ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("zh_cn", ROOT / "scripts/zh_cn.py")
zh_cn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zh_cn)


def test_committed_edition_matches_a_fresh_conversion():
    written = zh_cn.build()
    stale = [str(p.relative_to(ROOT)) for p, t in written.items() if not p.exists() or p.read_text() != t]
    assert not stale, f"run python scripts/zh_cn.py: {stale[:5]}"
    extra = [p for root in (zh_cn.DST_DOCS, zh_cn.DST_PROMPTS) for p in root.rglob("*.md*") if p not in written]
    assert not extra


@pytest.mark.parametrize("tw, cn", [
    ("把它們排成 $Q$ 的各行", "把它们排成 $Q$ 的各列"),           # Taiwan 行 = column
    ("$E$ 的每一列是獨熱向量", "$E$ 的每一行是独热向量"),           # Taiwan 列 = row
    ("字典的每個欄向量", "字典的每个列向量"),                     # Taiwan column terminology
    ("序列的排列與執行", "序列的排列与执行"),                     # not rows or columns
    ("只用一行線性代數", "只用一行线性代数"),                     # a line of text
    ("資料的變異數與共變異數", "数据的方差与协方差"),
    ("拔靴法區間", "自助法区间"),
    ("最大概似與不偏估計量", "最大似然与无偏估计量"),
    ("「查詢」跟著內容", "“查询”跟着内容"),
    ("顯著水準與檢定力", "显著性水平与统计功效"),
    ("連鎖律", "链式法则"),
    ("小型專題＋口試答辯", "小型专题＋答辩"),
    ("期中考＋口試", "期中考＋口试"),
    ("向量", "向量"),
    ("指標", "指标"),
])
def test_conversions(tw, cn):
    assert zh_cn.convert_text(tw) == cn


def test_heading_anchors_in_links_follow_the_heading():
    md = "見[撰寫標準](/zh-tw/project/writing-standard/#正文的規則)與`程式`。"
    assert zh_cn.convert_markdown(md) == "见[写作规范](/zh-cn/project/writing-standard/#正文的规则)与`程式`。"


TAIWAN_USAGE = ["资料", "变异数", "机率", "几率", "拔靴", "概似", "不偏", "透过", "程式", "档案", "专案", "讯号", "杂讯",
                "网路", "记忆体", "快取", "位元", "预设", "支援", "软体", "硬体", "互动", "视窗", "最佳化", "正规化",
                "递回", "递迴", "吃角子", "母体", "常态分", "检定", "信赖", "虚无假设", "字元", "变数", "系理", "实作",
                "建构", "后设", "矢量", "串行", "若且唯若", "连锁律", "跟著", "随著", "鉴别度", "效能", "正确率"]


def test_no_taiwan_usage_left():
    text = "\n".join(p.read_text() for p in zh_cn.DST_DOCS.rglob("*.md*") if p.name != "glossary.mdx")
    text = re.sub(r"`[^`\n]*`", "", text)
    found = {w: text.count(w) for w in TAIWAN_USAGE if w in text}
    assert not found, found
