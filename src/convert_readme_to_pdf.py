"""
Convert README.md to a publication-quality PDF with:
  1. Full MathML LaTeX math rendering via latex2mathml (with complete placeholder protection)
  2. Beautiful Twemoji SVG replacement for crisp, colorful emojis
  3. Clean table parsing and page-break optimization
  4. Professional typography and header/footer styling
"""

import os
import re
from pathlib import Path
import emoji
import markdown
import latex2mathml.commands
from weasyprint import HTML, CSS


def render_math_and_emojis_to_pdf(readme_path: Path, output_pdf_path: Path):
    print(f"Reading {readme_path} ...")
    raw_content = readme_path.read_text(encoding="utf-8")
    base_dir = readme_path.parent.resolve()

    # Step 1: Fix relative image paths to absolute file URIs
    def replace_img_src(match):
        alt = match.group(1)
        src = match.group(2)
        if not src.startswith("http") and not src.startswith("file://") and not src.startswith("data:"):
            abs_img_path = (base_dir / src).resolve()
            if abs_img_path.exists():
                return f"![{alt}](file://{abs_img_path})"
        return match.group(0)

    content = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_img_src, raw_content)

    # Step 2: Unwrap details and summary tags
    content = re.sub(r'<details.*?>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</details>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'<summary>(.*?)</summary>', r'\n\n### \1\n\n', content, flags=re.IGNORECASE | re.DOTALL)
    content = re.sub(r'<br\s*/?>', '\n', content, flags=re.IGNORECASE)

    # Step 3: Extract and protect LaTeX math expressions
    math_store = {}

    def extract_block_math(match):
        latex = match.group(1).strip()
        key = f"XYZMATHBLOCK{len(math_store)}ZYX"
        try:
            mathml = latex2mathml.commands.process_latex(latex)
            math_store[key] = f'<div class="math-block">{mathml}</div>'
        except Exception:
            math_store[key] = f'<div class="math-block"><code>{latex}</code></div>'
        return f"\n\n{key}\n\n"

    def extract_inline_math(match):
        latex = match.group(1).strip()
        if not latex:
            return "$$"
        key = f"XYZMATHINLINE{len(math_store)}ZYX"
        try:
            mathml = latex2mathml.commands.process_latex(latex)
            math_store[key] = f'<span class="math-inline">{mathml}</span>'
        except Exception:
            math_store[key] = f'<span class="math-inline"><code>{latex}</code></span>'
        return key

    # First extract block math $$...$$
    content = re.sub(r'\$\$(.*?)\$\$', extract_block_math, content, flags=re.DOTALL)
    # Next extract inline math $...$
    content = re.sub(r'(?<!\\)\$(.*?)(?<!\\)\$', extract_inline_math, content)

    # Step 4: Replace Unicode Emojis with crisp Twemoji SVGs
    emoji_pattern = re.compile('|'.join(re.escape(k) for k in sorted(emoji.EMOJI_DATA.keys(), key=len, reverse=True)))

    def emoji_repl(m):
        char = m.group(0)
        codepoints = '-'.join(f'{ord(c):x}' for c in char if ord(c) != 0xfe0f)
        url = f'https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/{codepoints}.svg'
        return f'<img class="emoji" src="{url}" alt="{char}"/>'

    content = emoji_pattern.sub(emoji_repl, content)

    # Step 5: Clean table spacing
    lines = content.splitlines()
    processed_lines = []
    in_table = False
    for line in lines:
        is_table_line = bool(re.match(r'^\s*\|.*\|\s*$', line))
        if is_table_line and not in_table:
            if processed_lines and processed_lines[-1].strip() != "":
                processed_lines.append("")
            in_table = True
        elif not is_table_line and in_table:
            if processed_lines and processed_lines[-1].strip() != "":
                processed_lines.append("")
            in_table = False
        processed_lines.append(line)
    content = "\n".join(processed_lines)

    # Step 6: Convert Markdown to HTML
    extensions = [
        "markdown.extensions.extra",
        "markdown.extensions.tables",
        "markdown.extensions.fenced_code",
        "markdown.extensions.toc",
        "markdown.extensions.nl2br",
        "markdown.extensions.sane_lists",
    ]
    html_body = markdown.markdown(content, extensions=extensions)

    # Step 7: Re-inject MathML expressions
    for key, mathml_code in math_store.items():
        # Match standalone <p>XYZMATHBLOCK...</p>
        html_body = html_body.replace(f"<p>{key}</p>", mathml_code)
        html_body = html_body.replace(key, mathml_code)

    css_style = """
    @page {
        size: A4;
        margin: 18mm 14mm 18mm 14mm;
        @bottom-right {
            content: "Page " counter(page) " / " counter(pages);
            font-size: 8pt;
            font-family: 'Noto Sans CJK JP', 'Noto Sans JP', sans-serif;
            color: #64748b;
        }
        @bottom-left {
            content: "Two-Tower Recommendation with Exploration & Embedding Scaling";
            font-size: 8pt;
            font-family: 'Noto Sans CJK JP', 'Noto Sans JP', sans-serif;
            color: #64748b;
        }
        @top-right {
            content: "Technical Report";
            font-size: 8pt;
            font-family: 'Noto Sans CJK JP', 'Noto Sans JP', sans-serif;
            color: #94a3b8;
        }
    }

    body {
        font-family: 'Noto Sans CJK JP', 'Noto Sans JP', 'DejaVu Sans', sans-serif;
        font-size: 9.0pt;
        line-height: 1.55;
        color: #1e293b;
        background-color: #ffffff;
    }

    h1 {
        font-size: 16pt;
        font-weight: 700;
        color: #0f172a;
        border-bottom: 2px solid #2563eb;
        padding-bottom: 5px;
        margin-top: 0;
        margin-bottom: 12px;
        page-break-after: avoid;
    }

    h2 {
        font-size: 11.5pt;
        font-weight: 700;
        color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding-left: 8px;
        margin-top: 18px;
        margin-bottom: 8px;
        page-break-after: avoid;
        background-color: #f8fafc;
        padding-top: 3px;
        padding-bottom: 3px;
    }

    h3 {
        font-size: 10pt;
        font-weight: 700;
        color: #334155;
        margin-top: 14px;
        margin-bottom: 6px;
        page-break-after: avoid;
    }

    h4 {
        font-size: 9pt;
        font-weight: 700;
        color: #475569;
        margin-top: 10px;
        margin-bottom: 4px;
        page-break-after: avoid;
    }

    p {
        margin-top: 0;
        margin-bottom: 8px;
        text-align: justify;
    }

    /* Emoji styling */
    img.emoji {
        height: 1.15em;
        width: 1.15em;
        vertical-align: -0.18em;
        margin: 0 0.15em;
        display: inline-block;
        border: none;
    }

    /* Links */
    a {
        color: #2563eb;
        text-decoration: none;
        font-weight: 500;
    }

    /* Tables */
    table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 8px;
        margin-bottom: 12px;
        font-size: 7.2pt;
        page-break-inside: avoid;
    }

    th {
        background-color: #f1f5f9;
        color: #0f172a;
        font-weight: 700;
        text-align: center;
        padding: 4.5px 5px;
        border: 1px solid #cbd5e1;
    }

    td {
        padding: 4px 5px;
        border: 1px solid #e2e8f0;
        text-align: left;
    }

    tr:nth-child(even) {
        background-color: #f8fafc;
    }

    /* MathML math styling */
    .math-block {
        display: block;
        margin: 7px 0;
        text-align: center;
        font-size: 9.8pt;
        page-break-inside: avoid;
    }

    .math-inline {
        display: inline;
        font-size: 9.0pt;
        vertical-align: -0.05em;
    }

    math {
        font-family: 'DejaVu Serif', 'Cambria Math', serif;
    }

    /* Code and ASCII diagrams */
    pre {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 3.5px solid #64748b;
        border-radius: 4px;
        padding: 6px 8px;
        font-family: 'DejaVu Sans Mono', 'Ubuntu Mono', monospace;
        font-size: 7.0pt;
        line-height: 1.3;
        color: #334155;
        overflow-x: auto;
        white-space: pre-wrap;
        word-break: break-all;
        margin-top: 6px;
        margin-bottom: 10px;
        page-break-inside: avoid;
    }

    code {
        font-family: 'DejaVu Sans Mono', 'Ubuntu Mono', monospace;
        font-size: 7.8pt;
        background-color: #f1f5f9;
        color: #0f172a;
        padding: 1px 3px;
        border-radius: 3px;
        border: 1px solid #e2e8f0;
    }

    pre code {
        background-color: transparent;
        border: none;
        padding: 0;
        font-size: 7.0pt;
    }

    /* Images */
    img:not(.emoji) {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 8px auto;
        border-radius: 4px;
        border: 1px solid #e2e8f0;
        page-break-inside: avoid;
    }

    /* Blockquotes & Callouts */
    blockquote {
        margin: 6px 0;
        padding: 6px 10px;
        background-color: #f0f9ff;
        border-left: 3.5px solid #0284c7;
        color: #0369a1;
        font-size: 8.5pt;
        border-radius: 0 4px 4px 0;
        page-break-inside: avoid;
    }

    blockquote p {
        margin: 0;
    }

    /* Lists */
    ul, ol {
        margin-top: 0;
        margin-bottom: 6px;
        padding-left: 16px;
    }

    li {
        margin-bottom: 2.5px;
    }

    /* Horizontal Rules */
    hr {
        border: 0;
        height: 1px;
        background: #e2e8f0;
        margin: 14px 0;
    }

    /* Badges */
    p img[src*="shields.io"] {
        display: inline-block;
        margin: 0 2px;
        border: none;
        vertical-align: middle;
    }
    """

    full_html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="utf-8">
    <title>Two-Tower Recommendation Benchmark Report</title>
</head>
<body>
{html_body}
</body>
</html>"""

    temp_html_path = output_pdf_path.with_suffix(".html")
    temp_html_path.write_text(full_html, encoding="utf-8")

    print(f"Generating PDF -> {output_pdf_path} ...")
    html_doc = HTML(filename=str(temp_html_path), base_url=str(base_dir))
    html_doc.write_pdf(target=str(output_pdf_path), stylesheets=[CSS(string=css_style)])
    print(f"Successfully generated PDF: {output_pdf_path} (Size: {output_pdf_path.stat().st_size:,} bytes)")


def main():
    readme_path = Path("README.md")
    output_pdf = Path("README.pdf")
    render_math_and_emojis_to_pdf(readme_path, output_pdf)


if __name__ == "__main__":
    main()
