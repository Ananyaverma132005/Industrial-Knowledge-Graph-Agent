"""Build the static GitHub Pages demo into _site/.

Copies the real UI (static/) and adds demo/demo-shim.js, which replays the
responses saved by demo/record_demo.py. Re-run record_demo.py first if the
data or agent changes.

Usage:
    python demo/build_static.py
"""
import hashlib
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, '_site')


def file_hash(name):
    with open(os.path.join(OUT, name), 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()[:10]


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.copytree(os.path.join(ROOT, 'static'), OUT)
    for name in ('demo-shim.js', 'recorded.json'):
        shutil.copy(os.path.join(ROOT, 'demo', name), os.path.join(OUT, name))

    index = os.path.join(OUT, 'index.html')
    with open(index, encoding='utf-8') as f:
        html = f.read()
    tag = '<script src="script.js"></script>'
    if tag not in html:
        raise SystemExit('script.js tag not found in index.html')
    html = html.replace(tag, '<script src="demo-shim.js"></script>\n  ' + tag)

    # Version asset URLs by content so browsers pick up new builds immediately.
    shim_path = os.path.join(OUT, 'demo-shim.js')
    with open(shim_path, encoding='utf-8') as f:
        shim = f.read()
    shim = shim.replace("var DATA_URL = 'recorded.json';",
                        f"var DATA_URL = 'recorded.json?v={file_hash('recorded.json')}';")
    with open(shim_path, 'w', encoding='utf-8') as f:
        f.write(shim)
    for name in ('style.css', 'demo-shim.js', 'script.js'):
        html = html.replace(f'"{name}"', f'"{name}?v={file_hash(name)}"')
    with open(index, 'w', encoding='utf-8') as f:
        f.write(html)

    open(os.path.join(OUT, '.nojekyll'), 'w').close()
    print(f'Built static demo in {OUT}')


if __name__ == '__main__':
    main()
