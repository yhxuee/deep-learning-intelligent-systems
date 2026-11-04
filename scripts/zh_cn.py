"""Generate the Simplified Chinese (mainland) edition from the Traditional Chinese (Taiwan) edition.

    python scripts/zh_cn.py            # regenerate src/content/docs/zh-cn/ and prompts/zh-cn/
    python scripts/zh_cn.py --check    # exit 1 if the committed zh-cn files differ from a fresh run

The zh-TW edition is the source of truth for both Chinese editions. Character conversion alone does not
give a native mainland text: Taiwan and mainland usage differ in vocabulary (資料 -> 数据, 變異數 -> 方差,
程式 -> 程序, 拔靴法 -> 自助法), in punctuation (「」 -> “”) and, for matrices, in the meaning of 行 and 列
(Taiwan: 行 = column, 列 = row; mainland: 行 = row, 列 = column). The pipeline is:

  1. swap matrix 行/列, protecting words in which they do not mean row or column (執行, 序列, ...);
  2. replace Taiwan terms with mainland terms from TERMS (longest match first);
  3. convert the remaining characters with OpenCC (t2s: characters only — its Taiwan-phrase mode maps
     查詢 -> 查找, 指標 -> 指针, 核心 -> 内核 and 序列 -> 串行, which is wrong in this domain);
  4. convert quotation marks, links (/zh-tw/ -> /zh-cn/) and language tags;
  5. apply per-file OVERRIDES for pages that talk about the editions themselves.

Do not edit files under zh-cn/ by hand: fix the zh-TW source, TERMS or OVERRIDES and regenerate.
"""
import argparse
import pathlib
import re
import sys

from opencc import OpenCC

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DOCS, DST_DOCS = ROOT / "src/content/docs/zh-tw", ROOT / "src/content/docs/zh-cn"
SRC_PROMPTS, DST_PROMPTS = ROOT / "prompts/zh-tw", ROOT / "prompts/zh-cn"

_cc = OpenCC("t2s")

# ----------------------------------------------------------------------------- 1. 行 / 列
# Words in which 行 or 列 does not mean column or row.  Protected before the swap.
KEEP = [
    # 行
    "執行緒", "執行", "進行", "行為", "運行", "並行", "流行", "平行", "銀行", "行業", "行動", "行走", "旅行", "自行",
    "施行", "發行", "先行", "盛行", "試行", "實行", "推行", "可行", "行得", "行不", "行使", "行程", "行駛", "行進",
    "行星", "履行", "舉行", "行政", "行內", "單行", "多行", "行銷", "風行", "另行", "現行", "行李", "逆行", "行事",
    "例行", "飛行", "隨行", "同行", "行人", "分行", "成行", "行家", "行文", "行號", "行首", "排行", "強行", "爬行",
    "不行", "換行", "兩行", "三行", "四行", "五行", "這行", "該行", "行列式", "外行", "內行", "罪行", "行善", "言行",
    "行情", "行經", "徒行", "步行", "暢行", "行之", "行星", "列行",
    # 列
    "序列", "排列", "列出", "系列", "列表", "佇列", "隊列", "並列", "列舉", "陣列", "列印", "名列", "列為", "列入",
    "列明", "下列", "前列", "列車", "列於", "列成", "所列", "列了", "列在", "列項", "數列", "陳列", "列隊", "羅列",
    "列示",
]
# 一行 is "one line" (of code or text) except after 每 / 的 ("each column", "a column of the matrix").
LINE_ONE = re.compile(r"(?<![每的])一行")
LINE_NUM = re.compile(r"(\d)(\s*)行")


def swap_rows_columns(s: str) -> str:
    keep = {}

    def protect(text):
        key = f"{len(keep):04d}"
        keep[key] = text
        return key

    s = LINE_ONE.sub(lambda m: protect(m.group()), s)
    s = LINE_NUM.sub(lambda m: m.group(1) + m.group(2) + protect("行"), s)
    for w in sorted(KEEP, key=len, reverse=True):
        s = s.replace(w, protect(w)) if w in s else s
    s = s.replace("行", "").replace("列", "行").replace("", "列")
    for k, v in keep.items():
        s = s.replace(k, v)
    return s


# ----------------------------------------------------------------------------- 2. Terms
# Taiwan (Traditional) -> mainland (Simplified).  Applied longest-first, after the row/column swap.
TERMS = {
    # data, computing, tooling
    "參考資料": "参考资料", "保留資料": "留出数据", "保留樣本": "留出样本", "保留字元": "留出字符", "保留配對": "留出配对", "保留事件": "留出事件", "保留故事": "留出故事", "保留部分": "留出部分", "保留平均值": "留出平均值",
    "語料產生器": "语料生成器", "新產生的語料": "新生成的语料", "重新產生": "重新生成",
    "資料夾": "文件夹", "資料集": "数据集", "資料庫": "数据库", "資料流": "数据流", "資料表": "数据表",
    "後設資料": "元数据", "資料平行": "数据并行", "資料": "数据", "資訊理論": "信息论", "互資訊": "互信息", "資訊量": "信息量", "資訊": "信息",
    "音訊": "音频", "訊息": "信息", "訊號": "信号", "雜訊": "噪声", "原始碼": "源代码", "程式碼": "代码", "程式語言": "编程语言",
    "函式簽章": "函数签名", "程式": "程序", "函式": "函数", "軟體": "软件", "硬體": "硬件", "記憶體": "内存", "快取": "缓存", "位元組": "字节",
    "位元": "比特", "字元": "字符", "母音": "元音", "網路": "网络", "網頁": "网页", "伺服器": "服务器", "用戶端": "客户端", "介面": "接口",
    "預設": "默认", "支援": "支持", "檔案": "文件", "專案": "项目", "列印": "打印", "視窗": "窗口", "螢幕": "屏幕",
    "使用者": "用户", "帳號": "账号", "登入": "登录", "物件": "对象", "模組": "模块", "元件": "组件", "實作": "实现",
    "建構": "构建", "建置": "构建", "呼叫": "调用", "回傳": "返回", "引數": "参数", "儲存庫": "仓库", "儲存": "存储",
    "擷取": "提取", "偵測": "检测", "辨識": "识别", "品質": "质量", "影片": "视频", "互動式": "交互式", "互動圖表": "交互式图表",
    "互動": "交互", "視覺化": "可视化", "小數位": "小数位", "數位": "数字", "離線": "离线", "本機": "本地", "執行緒": "线程",
    "散射加總": "散射加法", "平行前綴掃描": "并行前缀扫描", "平行掃描": "并行扫描", "平行化": "并行化", "平行計算": "并行计算",
    "平行運算": "并行计算", "平行硬體": "并行硬件", "平行執行": "并行执行", "平行訓練": "并行训练", "平行處理": "并行处理",
    "模型平行": "模型并行", "張量平行": "张量并行", "管線平行": "流水线并行", "管線": "流水线", "叢集": "集群",
    "頻寬": "带宽", "晶片": "芯片", "顯示卡": "显卡", "陣列": "数组", "純量": "标量", "迴圈": "循环", "捨入": "舍入", "二進位檔": "二进制文件", "二進位": "二进制", "計帳": "记账",
    "竄改": "篡改", "行動裝置": "移动设备", "裝置": "设备", "筆電": "笔记本电脑", "硬碟": "硬盘", "部落格": "博客",
    "選單": "菜单", "貼上": "粘贴", "點選": "点击", "滑鼠": "鼠标", "回覆": "回复", "回饋": "反馈", "紀錄": "记录",
    "授權": "许可", "空連結": "空链接", "內部連結": "内部链接", "失效連結": "失效链接", "連結檢查": "链接检查",
    "跨領域連結": "跨领域联系", "連結": "联系", "個人化": "个性化", "排程": "调度", "文字": "文本", "範本": "模板",
    "透過": "通过", "藉由": "通过", "藉此": "借此", "於": "于", "計畫": "计划", "評量": "评估", "評分規準": "评分标准", "內含": "包含", "語意": "语义",
    "檢核點": "检查点", "檢核": "检查", "大學部": "本科", "週": "周", "總結專案": "综合项目",
    "口試答辯": "答辩", "雜湊": "哈希", "除錯": "调试", "想像": "想象", "每一欄": "每一列", "第一欄": "第一列", "欄向量": "列向量", "欄由": "列由", "欄位": "字段", "運算式": "表达式", "運算子": "运算符", "相容": "兼容",
    "駭客": "黑客", "解析度": "分辨率", "視訊": "视频", "原始檔": "源文件", "檢視": "查看", "宣告": "声明", "暫存器": "寄存器",
    "金鑰": "密钥", "訊雜比": "信噪比", "片語": "词组", "人工智慧": "人工智能", "智慧": "智能", "搜尋": "搜索", "通訊": "通信",
    "程式直譯器": "解释器", "直譯器": "解释器", "列舉": "枚举", "文件": "文档", "預設值": "默认值", "演算法": "算法",
    "若且唯若": "当且仅当", "鏡射": "反射", "暫態": "瞬态", "散佈": "散布", "影響力": "影响力", "預訓練": "预训练",
    # mathematics
    "變數變換": "变量替换", "隨機變數": "随机变量", "改變數學": "改变数学", "變數": "变量", "往回傳": "向后传递", "座標": "坐标", "系理": "推论", "正交歸一": "标准正交",
    "後驗平均": "后验均值", "先驗平均": "先验均值", "全域最小值": "全局最小值", "全域": "全局",
    "線性獨立": "线性无关", "列彼此獨立": "列线性无关", "列獨立": "列线性无关", "秩–零度定理": "秩–零化度定理",
    "畢氏定理": "勾股定理", "最小平方法": "最小二乘法", "最小平方": "最小二乘", "正規方程式": "正规方程", "方程式": "方程",
    "黑塞": "海森", "偽逆": "伪逆", "疊代": "迭代", "摺積": "卷积", "內插": "插值", "門檻": "阈值", "拉格朗日乘數": "拉格朗日乘子", "乘數": "乘子",
    "互補鬆弛": "互补松弛", "壓縮映射": "压缩映射", "不動點": "不动点", "柴比雪夫": "切比雪夫", "詹森": "琴生",
    "馬可夫": "马尔可夫", "波茲曼": "玻尔兹曼", "費雪": "费希尔", "貝氏": "贝叶斯", "交互作用": "交互",
    "$-平滑": "$-光滑", "-平滑的": "-光滑的", "平滑性": "光滑性",
    # probability and statistics
    "機率": "概率", "共變異數": "协方差", "共變異": "协方差", "變異數": "方差", "變異代理": "方差代理", "變異": "差异",
    "不偏": "无偏", "歸納偏誤": "归纳偏置", "偏誤": "偏差", "最大概似": "最大似然", "對數概似": "对数似然", "概似": "似然",
    "拔靴法": "自助法", "拔靴": "自助", "蒙地卡羅": "蒙特卡洛", "重抽樣": "重采样", "重抽": "重采样", "抽樣分布": "抽样分布", "抽樣誤差": "抽样误差",
    "抽樣雜訊": "抽样噪声", "重複抽樣": "重复抽样", "不放回抽樣": "不放回抽样", "放回抽樣": "放回抽样",
    "重要性抽樣": "重要性采样", "祖先抽樣": "祖先采样", "核心抽樣": "核采样", "抽樣器": "采样器", "取樣": "采样",
    "抽樣": "采样", "樣本數": "样本量", "母群體": "总体", "母體": "总体", "常態分布": "正态分布", "常態近似": "正态近似",
    "標準常態": "标准正态", "對數常態": "对数正态", "大致常態": "大致正态", "漸近常態性": "渐近正态性",
    "檢定力": "统计功效", "假設檢定": "假设检验", "檢定": "检验", "虛無假設": "零假设", "對立假設": "备择假设",
    "顯著水準": "显著性水平", "信心水準": "置信水平", "水準": "水平", "第一型錯誤": "第一类错误", "第二型錯誤": "第二类错误",
    "信賴區間": "置信区间", "更有信心": "更有把握", "有信心": "有把握", "信心": "置信度", "邊際分布": "边缘分布",
    "邊際機率": "边缘概率", "邊際速度": "边缘速度", "邊際場": "边缘场", "*邊際*場": "*边缘*场", "**邊際**分布": "**边缘**分布",
    "支撐集": "支撑集", "完整支撐": "全支撑", "的支撐上": "的支撑集上", "支撐上的": "支撑集上的", "鑑別度": "分辨力",
    "保形預測": "共形预测", "共變量偏移": "协变量偏移", "可靠度": "可靠性",
    # learning, models and methods
    "依此類推": "依此类推", "類推": "泛化",
    "不變量測試": "不变量测试", "量測科學": "计量科学", "量測儀器": "测量工具", "量測紀律": "评估规范", "量測雜訊": "测量噪声",
    "量測結果": "测量结果", "量測出來的": "可度量的", "量測到的": "测得的", "量測數字": "测得的数值", "被量測的": "可度量的",
    "重新量測": "重新测量", "量測": "度量", "相似度量度": "相似性度量", "量度": "度量",
    "最佳化器": "优化器", "過度最佳化": "过度优化", "最佳化": "优化", "最佳傳輸": "最优传输", "最佳輸運": "最优传输",
    "最佳": "最优", "正規化流": "标准化流", "層正規化": "层归一化", "正規化": "归一化", "正規矩陣": "正规矩阵",
    "遞迴神經網路": "循环神经网络", "類神經網路": "神经网络", "神經網路": "神经网络", "遞迴狀態": "循环状态",
    "遞迴架構": "循环架构", "閘控遞迴": "门控循环", "遞迴模型": "循环模型", "遞迴網路": "循环网络", "遞迴層": "循环层",
    "遞迴": "递推", "閘控": "门控", "邏輯斯迴歸": "逻辑回归", "邏輯斯函數": "logistic 函数", "邏輯斯": "logistic",
    "迴歸": "回归", "小批次": "小批量", "批次大小": "批量大小", "規模定律": "缩放定律", "計算最佳": "计算最优",
    "分散式": "分布式", "自我監督": "自监督", "遮蔽語言模型": "掩码语言模型", "遮蔽": "掩码", "對比式學習": "对比学习", "對比式": "对比",
    "詞彙表": "词表", "詞彙量": "词表大小", "學習率排程": "学习率调度",
    "吃角子老虎機": "多臂老虎机", "情境式多臂老虎机": "上下文多臂老虎机", "行動者–評論者": "演员–评论家",
    "貪婪": "贪心", "獎勵駭取": "奖励破解", "駭取": "破解", "規格投機": "规范博弈", "憲章式 AI": "宪法式 AI",
    "基準測試": "基准测试", "基準率": "基础率", "基準點": "基准点", "基準": "基线",
    "AI 代理工具": "AI 智能体工具", "自主研究代理": "自主研究智能体", "使用工具的代理": "使用工具的智能体",
    "代理與工具": "智能体与工具", "演化式程式代理": "演化式编程智能体", "在環境中行動的代理": "在环境中行动的智能体",
    "在世界中行動的代理": "在世界中行动的智能体",
    "激活修補": "激活修补", "介入": "干预", "疊加": "叠加",
    # the book
    "主幹專案": "主干项目", "詳解例題": "例题", "預先登記": "预注册", "登記": "注册", "可重現性": "可复现性", "重現": "复现",
    "撰寫標準": "写作规范", "工具組": "工具包", "先備知識": "先修知识",
    "型別的連結": "带类型的链接", "有型別": "带类型", "型別": "类型", "懸空連結": "悬空链接", "引用連結": "引用链接",
    "連結與引用": "链接与引用", "或它的連結": "或它的链接", "正確連結到": "正确链接到", "任何連結": "任何链接",
    "冷氣": "空调", "同儕審查": "同行评审", "直覺的": "直观的", "精熟": "掌握", "標示": "标记",
    "連鎖律": "链式法则", "AI 代理": "AI 智能体", "相依套件": "依赖包", "相依": "依赖", "聯集界": "联合界", "聯集": "并集",
    "註解": "注释", "例行": "常规", "脈絡": "背景", "監督式": "监督", "函式庫": "库", "程式庫": "库", "激勵函數": "激活函数",
    "模型設定": "模型配置", "亦即": "即", "採 CC": "采用 CC", "預估": "预计", "公設": "公理", "釋出": "发布", "譬如": "比如",
    "線上 softmax": "在线 softmax", "欄：鍵": "列：键", "感測讀數": "传感器读数", "感測": "传感", "載入": "加载", "線上執行": "在线运行",
    "效能": "性能", "正確率": "准确率", "偽陽性": "假阳性", "偽陰性": "假阴性", "盛行率": "患病率", "封閉形式": "闭式",
    "亂數種子": "随机数种子", "亂數": "随机数", "串流": "流式", "感測器": "传感器", "自駕車": "自动驾驶汽车", "實務上": "实践中",
    "實務": "实践", "套用": "应用", "中央差分": "中心差分", "突波": "尖峰", "執行時間": "运行时间", "甚麼": "什么",
    "直列向量": "列向量", "橫行向量": "行向量", "直行向量": "列向量", "橫列向量": "行向量", "考量 IO 的精確注意力核心": "面向内存读写优化的精确注意力内核", "考量 IO 的注意力核心": "面向内存读写优化的注意力内核", "考量 IO 的精確注意力": "面向内存读写优化的精确注意力", "考量 IO 的核心": "面向内存读写优化的内核", "注意力核心": "注意力内核", "運算核心": "计算内核", "硬式定址": "硬寻址", "軟式定址": "软寻址", "儲存樣式": "存储模式", "樣式數": "模式数", "無母數": "非参数", "查詢欄": "查询槽", "鍵欄": "键槽", "值欄": "值槽", "位址": "地址", "酬載": "载荷", "研究所學習": "研究生学习", "研究所課程": "研究生课程",
    "上兆個": "上万亿个", "兆級": "万亿级", "1.4 兆": "1.4 万亿", "一兆": "一万亿", "資工": "计算机", "統計推論": "统计推断", "貝氏推論": "贝叶斯推断", "變分推論": "变分推断", "強推論": "强推断",
    "推論成本": "推理成本", "推論系統": "推理系统", "推論服務": "推理服务", "推論計算": "推理计算", "推論經濟": "推理经济",
    "推論 FLOP": "推理 FLOP", "推論": "推断", "尺度定律": "缩放定律", "字串": "字符串", "影像": "图像", "逐步演算": "逐步推演",
    "批次": "批量", "平行**掃描": "并行**扫描", "**平行（上掃）": "**并行（上扫）", "個別": "各个", "版本紀錄": "更新日志", "近期發展": "最新进展", "限制與反例": "局限与反例",
}
_TERM_RE = re.compile("|".join(re.escape(k) for k in sorted(TERMS, key=len, reverse=True)))


# 著 is the aspect particle zhe (跟著, 隨著 -> 跟着, 随着) except in words where it is zhù.
_ZHE = re.compile(r"(?<!显)(?<!卓)(?<!昭)(?<!土)(?<!编)(?<!原)(?<!专)(?<!名)(?<!巨)(?<!论)(?<!遗)(?<!译)著(?![名作称者])")


def convert_text(s: str) -> str:
    s = swap_rows_columns(s)
    s = _TERM_RE.sub(lambda m: TERMS[m.group()], s)
    s = _cc.convert(s)
    s = _ZHE.sub("着", s)
    s = s.replace("「", "“").replace("」", "”").replace("『", "‘").replace("』", "’")
    s = s.replace("/zh-tw/", "/zh-cn/").replace('lang="zh-TW"', 'lang="zh-CN"')
    s = s.replace("prompts/zh-tw/", "prompts/zh-cn/")
    return s


# Inline code and link targets are copied unchanged, apart from /zh-tw/ in links and a Chinese heading
# anchor (#...), which is converted like the heading it points to; comments inside fenced code blocks are
# Chinese text and are converted like prose.
_PROTECT = re.compile(r"```.*?```|`[^`\n]*`|\]\([^)]*\)", re.S)


_CJK = re.compile(r"[\u3400-\u9fff]")


def _convert_link(t: str) -> str:
    t = t.replace("/zh-tw/", "/zh-cn/").replace("zh-tw/units/", "zh-cn/units/")
    if t.startswith("](") and "#" in t and _CJK.search(t.split("#", 1)[1]):
        target, frag = t.split("#", 1)
        t = target + "#" + convert_text(frag)
    return t


def convert_markdown(s: str) -> str:
    out, last = [], 0
    for m in _PROTECT.finditer(s):
        out.append(convert_text(s[last:m.start()]))
        if m.group().startswith("```"):
            out.append(convert_text(m.group()))
        else:
            out.append(_convert_link(m.group()))
        last = m.end()
    out.append(convert_text(s[last:]))
    return "".join(out)


# ----------------------------------------------------------------------------- 5. Edition-specific text
# Pages written natively for the mainland edition (they describe the edition itself) live in
# scripts/zh_cn_overrides/ and replace the converted page.  Smaller differences are string replacements
# applied to the converted text.
FULL_OVERRIDES = ROOT / "scripts/zh_cn_overrides"
OVERRIDES: dict[str, list[tuple[str, str]]] = {
    "core/unit-04/index.md": [
        ("“台湾最高峰是玉山。日本最高峰是富士山。台湾最高峰是……”", "“中国最高峰是珠穆朗玛峰。日本最高峰是富士山。中国最高峰是……”"),
    ],
    "mentor/modes.mdx": [("（繁体中文版）", "（简体中文版）")],
    "mentor/unit-prompts.mdx": [("（繁体中文版）", "（简体中文版）")],
    "index.mdx": [
        ("本书关键术语的台湾学术用语、英文原文，以及常见的中国大陆用法。", "本书关键术语的简体中文译名、英文原文，以及台湾常用译法。"),
        ('<LinkCard title="简体中文版 · 简体中文" href="../zh-cn/" description="全书中国大陆用语版，内容与繁体版同步。" />',
         '<LinkCard title="繁體中文版 · 繁体中文" href="../zh-tw/" description="全书台湾用语版，内容与简体版同步。" />'),
    ],
    "base-policy.md": [("请以繁体中文（台湾用语）回复", "请以简体中文（中国大陆通行用语）回复")],
}


def apply_overrides(rel: str, s: str) -> str:
    for old, new in OVERRIDES.get(rel, []):
        if old not in s:
            raise SystemExit(f"override for {rel} no longer matches: {old[:40]}")
        s = s.replace(old, new)
    return s


GLOSSARY_SRC, GLOSSARY_DST = ROOT / "src/data/glossary.yaml", ROOT / "src/data/glossary.zh-cn.json"
_FIELD = re.compile(r'(\w+): "([^"]*)"')


def glossary_json() -> str:
    """Mainland term for every glossary entry: the curated `cn` variant if there is one, else the converted
    Taiwan term.  Written as JSON so that it can be reviewed entry by entry."""
    import json

    out = {}
    for line in GLOSSARY_SRC.read_text().splitlines():
        m = re.match(r"^([a-z0-9-]+): \{(.*)\}\s*$", line)
        if not m:
            continue
        fields = dict(_FIELD.findall(m.group(2)))
        term = _cc.convert(fields["cn"]) if "cn" in fields else convert_text(fields["zh"])
        out[m.group(1)] = {"term": term}  # the zh-TW notes discuss Taiwan usage and are not carried over
    return json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def build(dst_docs=DST_DOCS, dst_prompts=DST_PROMPTS):
    written = {GLOSSARY_DST: glossary_json()}
    for src_root, dst_root in ((SRC_DOCS, dst_docs), (SRC_PROMPTS, dst_prompts)):
        for f in sorted(src_root.rglob("*")):
            if f.is_dir() or f.suffix not in {".md", ".mdx"}:
                continue
            rel = f.relative_to(src_root).as_posix()
            full = FULL_OVERRIDES / rel
            if src_root == SRC_DOCS and full.exists():
                written[dst_root / rel] = full.read_text()
                continue
            written[dst_root / rel] = apply_overrides(rel, convert_markdown(f.read_text()))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    written = build()
    if args.check:
        stale = [p for p, t in written.items() if not p.exists() or p.read_text() != t]
        extra = [p for root in (DST_DOCS, DST_PROMPTS) for p in root.rglob("*.md*") if p not in written]
        for p in stale + extra:
            print("out of date:", p.relative_to(ROOT))
        sys.exit(1 if stale or extra else 0)
    for p, t in written.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(t)
    print(f"wrote {len(written)} files")


if __name__ == "__main__":
    main()
