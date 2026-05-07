import io
import re
import tarfile
import urllib.request

url = 'https://arxiv.org/src/1904.03000'
print('Downloading source...')
raw = urllib.request.urlopen(url, timeout=60).read()
print('Bytes:', len(raw))

tf = tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz')

texts = []
for m in tf.getmembers():
    if not m.isfile():
        continue
    name = m.name.lower()
    if name.endswith('.tex') or name.endswith('.txt') or name.endswith('.bib'):
        try:
            data = tf.extractfile(m).read().decode('utf-8', errors='ignore')
            texts.append((m.name, data))
        except Exception:
            pass

print('Files:', [n for n,_ in texts])

for n,t in texts:
    if 'serve wine' in t.lower() or 'task' in t.lower():
        print('\n===', n, '===')
        for pat in [r'serve wine', r'\\caption\{[^}]*\}', r'task[s]?\\?\s*[:=].*']:
            m = re.search(pat, t, flags=re.IGNORECASE)
            if m:
                print('match:', m.group(0)[:400])

# brute: print lines containing 'would a human choose to'
for n,t in texts:
    lines = [ln.strip() for ln in t.splitlines() if 'would a human choose to' in ln.lower()]
    if lines:
        print('\nTask-like lines in', n)
        for ln in lines:
            print(ln)
