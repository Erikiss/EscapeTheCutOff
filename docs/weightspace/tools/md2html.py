"""Markdown dossier -> standalone print HTML (with a table of contents). Usage: md2html.py IN.md OUT.html"""
import html
import re
import sys

from markdown_it import MarkdownIt

src = open(sys.argv[1], encoding='utf-8').read()
md = MarkdownIt('commonmark', {'html': False, 'linkify': False, 'typographer': False}).enable('table')

tokens = md.parse(src)
toc, used = [], set()


def slug(text):
    s = re.sub(r'[^\w\- ]', '', text.lower()).strip().replace(' ', '-')
    base, n = s or 'section', 1
    while s in used:
        n += 1
        s = f'{base}-{n}'
    used.add(s)
    return s


for i, t in enumerate(tokens):
    if t.type == 'heading_open' and t.tag in ('h1', 'h2', 'h3'):
        text = tokens[i + 1].content
        plain = re.sub(r'[`*]', '', text)
        sid = slug(plain)
        t.attrSet('id', sid)
        if t.tag in ('h2', 'h3'):
            toc.append((t.tag, plain, sid))

body = md.renderer.render(tokens, md.options, {})
# short code spans inside table cells must not wrap (exact identifiers such as H_RESID_NOFOLD)
def _nw(m):
    cell = m.group(0)
    return re.sub(r'<code>([^<\s]{1,32})</code>', r'<code class="nw">\1</code>', cell)
body = re.sub(r'<td>.*?</td>|<th>.*?</th>', _nw, body, flags=re.S)
# keep the first h1 + German summary before the TOC
first_hr = body.find('<hr />')
toc_html = '<nav class="toc"><h2>Inhalt / Contents</h2><ul>' + ''.join(
    f'<li class="{tag}"><a href="#{sid}">{html.escape(text)}</a></li>' for tag, text, sid in toc) + '</ul></nav>'
body = body[:first_hr] + toc_html + body[first_hr:] if first_hr > 0 else toc_html + body

css = r'''
@page { size: A4; margin: 16mm 14mm 18mm 14mm; }
html { font-family: "DejaVu Sans", "Liberation Sans", sans-serif; font-size: 9.6pt; line-height: 1.42; color: #111; background: #fff; }
body { margin: 0; }
h1 { font-size: 19pt; margin: 0 0 8pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; border-bottom: 1px solid #999; padding-bottom: 2pt; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 13pt 0 4pt; break-after: avoid; }
h4 { font-size: 10pt; margin: 10pt 0 3pt; break-after: avoid; }
p, li { orphans: 2; widows: 2; }
ul, ol { padding-left: 18pt; margin: 4pt 0; }
li { margin: 1.5pt 0; }
hr { border: 0; border-top: 1px solid #bbb; margin: 12pt 0; }
code { font-family: "DejaVu Sans Mono", "Liberation Mono", monospace; font-size: 8.4pt; background: #f2f2f2; padding: 0 2px; border-radius: 2px; overflow-wrap: anywhere; }
pre { background: #f6f6f6; border: 1px solid #ddd; border-radius: 3px; padding: 5pt 6pt; margin: 5pt 0;
      white-space: pre-wrap; overflow-wrap: anywhere; word-break: normal; }
pre code { background: none; padding: 0; font-size: 7.6pt; line-height: 1.32; }
table { border-collapse: collapse; width: 100%; margin: 5pt 0 8pt; font-size: 8.4pt; table-layout: auto; }
th, td { border: 1px solid #bbb; padding: 2.5pt 4pt; vertical-align: top; text-align: left; overflow-wrap: break-word; word-break: normal; hyphens: manual; }
th { background: #ececec; }
tr { break-inside: avoid; }
td code, th code { font-size: 7.8pt; overflow-wrap: break-word; }
code.nw { white-space: nowrap; overflow-wrap: normal; }
nav.toc { break-after: page; margin-top: 10pt; }
nav.toc h2 { border: 0; }
nav.toc ul { list-style: none; padding-left: 0; columns: 2; column-gap: 18pt; font-size: 8.6pt; }
nav.toc li.h3 { padding-left: 12pt; }
nav.toc a { color: #1a4f9c; text-decoration: none; }
a { color: #1a4f9c; }
'''
out = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>WSBench Build Dossier</title>
<style>{css}</style></head><body>{body}</body></html>'''
open(sys.argv[2], 'w', encoding='utf-8').write(out)
print('headings in toc:', len(toc))
