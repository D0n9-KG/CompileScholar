# -*- coding: utf-8 -*-
"""arXiv HTML -> markdown-ish text with headers preserved.

The smoke v1 conversion stripped all tags including <h1>-<h6>, which
destroyed chunk_text's section detection (fixed-window fallback ->
semantic routing all "other"). This version converts headings to
markdown ## first.
"""
import re
import sys
import urllib.request


def fetch_text(arxiv_id: str) -> str:
    url = f"https://arxiv.org/html/{arxiv_id}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 research"})
    with urllib.request.urlopen(req, timeout=120) as r:
        html = r.read().decode("utf-8", errors="replace")
    # 页面 chrome 清除（arXiv HTML 的 UI 元素——冒烟实测 c0 chunk 全是
    # "Report GitHub Issue × Title:" 这类噪声）
    html = re.sub(r'<div class="ltx_page_logo".*?</div>', "", html,
                  flags=re.S)
    html = re.sub(r'<div class="ltx_page_footer".*?</div>', "", html,
                  flags=re.S)
    html = re.sub(r"<nav.*?</nav>|<header.*?</header>|<footer.*?</footer>",
                  "", html, flags=re.S)
    # 标题→markdown（先做，保住结构）
    html = re.sub(r"<h1[^>]*>(.*?)</h1>", r"\n# \1\n", html, flags=re.S)
    html = re.sub(r"<h2[^>]*>(.*?)</h2>", r"\n## \1\n", html, flags=re.S)
    html = re.sub(r"<h3[^>]*>(.*?)</h3>", r"\n### \1\n", html, flags=re.S)
    html = re.sub(r"<h4[^>]*>(.*?)</h4>", r"\n#### \1\n", html, flags=re.S)
    # 段落/列表断行
    html = re.sub(r"</p>|</li>|</tr>|</table>", "\n", html)
    # 表格行内分隔
    html = re.sub(r"</t[dh]>", " | ", html)
    # 去其余标签
    html = re.sub(r"<script.*?</script>|<style.*?</style>", "", html,
                  flags=re.S)
    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"&[a-z]+;", " ", html)
    # 压空白但保留换行
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()


if __name__ == "__main__":
    out = fetch_text(sys.argv[1])
    print(f"{len(out)} chars")
    open(sys.argv[2], "w", encoding="utf-8").write(out)
