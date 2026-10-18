"""Check and render the manuscript PDFs without scientific computation."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw
paper = Path('artifacts/vldb2027/paper')
out = Path('build/paper-review')
out.mkdir(parents=True, exist_ok=True)
report = {'documents': {}, 'scientific_runs': 0}
for name in ('main', 'supplement'):
    pdf = paper / (name + '.pdf')
    info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
    count = int(re.search(r'^Pages:\s+(\d+)', info, re.M).group(1))
    text = subprocess.check_output(['pdftotext', '-layout', str(pdf), '-'], text=True)
    normal = [' '.join(p.split()) for p in text.split('\f')[:count]]
    (out / (name + '.txt')).write_text(text)
    log = (paper / (name + '.log')).read_text(errors='replace')
    errors = [s for s in ('There were undefined references', 'There were undefined citations', 'LaTeX Error:', 'Missing character:') if s in log]
    stale = [s for s in ('Evidence draft', 'internal evidence supplement', 'no completed public release is claimed', 'remain explicit completion items', 'public checkpoint access and native-source') if s in text]
    if errors or stale:
        raise RuntimeError(str({'document': name, 'errors': errors, 'stale': stale}))
    fonts = subprocess.check_output(['pdffonts', str(pdf)], text=True)
    (out / (name + '-fonts.txt')).write_text(fonts)
    unembedded = [s for s in fonts.splitlines() if re.search(r'\sno\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$', s)]
    if unembedded:
        raise RuntimeError('Unembedded fonts: ' + str(unembedded))
    urls = subprocess.check_output(['pdfinfo', '-url', str(pdf)], text=True)
    (out / (name + '-urls.txt')).write_text(urls)
    if 'github.com/xingxiaoyang-888/ACL-HCT' not in urls:
        raise RuntimeError('Public artifact link missing from ' + name)
    doc = {'pages': count, 'bytes': pdf.stat().st_size, 'sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(), 'all_fonts_embedded': True,
           'overfull_warnings': [s for s in log.splitlines() if 'Overfull' in s],
           'author_metadata_pending': 'Author information to be supplied' in text}
    if name == 'main':
        for required in ('PVLDB Reference Format', 'PVLDB Artifact Availability', 'Creative Commons'):
            if required not in ' '.join(normal):
                raise RuntimeError('Missing official block: ' + required)
        endings = [i+1 for i,p in enumerate(normal) if 'an actionable choice of inference route.' in p]
        if len(endings) != 1 or endings[0] > 12:
            raise RuntimeError('Body-page boundary requires review: ' + str(endings))
        doc['body_ends_on_page'] = endings[0]
        doc['reference_count'] = (paper / 'main.bbl').read_text().count('\\bibitem')
    report['documents'][name] = doc
    subprocess.run(['pdftoppm', '-png', '-r', '110', str(pdf), str(out / (name+'-page'))], check=True)
    rendered = sorted(out.glob(name+'-page-*.png'))
    for start in range(0,len(rendered),9):
        batch=rendered[start:start+9]
        sheet=Image.new('RGB',(960,420*((len(batch)+2)//3)),'#cccccc')
        for j,page in enumerate(batch):
            im=Image.open(page).convert('RGB'); im.thumbnail((300,390))
            tile=Image.new('RGB',(320,420),'white');tile.paste(im,((320-im.width)//2,24))
            ImageDraw.Draw(tile).text((10,5),page.stem,fill='black')
            sheet.paste(tile,((j%3)*320,(j//3)*420))
        sheet.save(out/(name+'-contact-'+str(start//9+1)+'.png'))
(out/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
