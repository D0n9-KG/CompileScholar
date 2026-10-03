"""DeepScholar-base intro 的引用链接 -> [n]（DeepScholarBaseParser._to_autoais 等价），供 gen_b1 与判分前清洗共用。

官方 _postprocess_citation 产出形如  \\[[作者' 日期](url)\\]  的引用；作者串取自 BibTeX，可能含换行。
做法：先定位每个 "](http...)" 链接尾，用 url 编号；再删掉残留的反斜杠外壳。"""
import re

LINK_TAIL = re.compile(r"\]\((https?://[^)\s]+)\)")
BACKSLASH_OPEN = "\\["
BACKSLASH_CLOSE = "\\]"


def to_numbered(text):
    url2n = {}
    out = []
    pos = 0
    for m in LINK_TAIL.finditer(text):
        url = m.group(1)
        if url not in url2n:
            url2n[url] = len(url2n) + 1
        # 链接文本起点：向左找最近的 "\[[" 外壳（官方格式），否则最近的 "["
        seg_start = text.rfind(BACKSLASH_OPEN + "[", pos, m.start())
        if seg_start != -1:
            left = seg_start
        else:
            left = text.rfind("[", pos, m.start())
            if left == -1:
                continue
        out.append(text[pos:left])
        out.append(f"[{url2n[url]}]")
        end = m.end()
        if text.startswith(BACKSLASH_CLOSE, end):
            end += len(BACKSLASH_CLOSE)
        pos = end
    out.append(text[pos:])
    res = "".join(out)
    # 模型偶尔在文末自行追加 "[n] url" 参考列表：剥掉（与 B0 剥参考文献列表同规则，不计入判分正文）
    lines = res.rstrip().split("\n")
    while lines and re.fullmatch(r"\s*\[\d+\]\s+https?://\S+\s*", lines[-1] or ""):
        lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()
