"""Render the local Markdown review draft to styled HTML for LibreOffice export."""
import argparse
from pathlib import Path
import markdown
from bs4 import BeautifulSoup
from PIL import Image


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    soup=BeautifulSoup(markdown.markdown(a.input.read_text(),extensions=['tables','fenced_code']),'html.parser')
    for img in soup.find_all('img'):
        path=(a.input.parent/img['src']).resolve()
        with Image.open(path) as image:
            width=650;height=round(image.height*width/image.width)
        img['src']=path.as_uri();img['width']=str(width);img['height']=str(height)
    style='''@page {size:Letter;margin:.75in;} body {font-family:"Liberation Sans",sans-serif;font-size:11pt;line-height:1.35;color:#182033;} h1 {font-size:22pt;color:#383283;} h2 {font-size:15pt;color:#383283;page-break-after:avoid;} h3 {font-size:12pt;} p {margin:0 0 9pt;} table {border-collapse:collapse;width:100%;font-size:9pt;} td,th {border:1px solid #bbb;padding:5px;} pre {font-size:9pt;white-space:pre-wrap;} img {display:block;} a {color:#383283;}'''
    html='<!doctype html><html><head><meta charset="utf-8"><title>EMCH585 Report Draft</title><style>'+style+'</style></head><body>'+str(soup)+'</body></html>'
    a.output.write_text(html);print(a.output)


if __name__=='__main__':main()
