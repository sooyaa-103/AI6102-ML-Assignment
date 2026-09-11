"""Download official a9a files and verify hashes from the recorded experiment."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    expected = json.loads((ROOT / 'results/metadata.json').read_text())['sha256']
    folder = ROOT / 'data'
    folder.mkdir(exist_ok=True)
    for name in ('a9a', 'a9a.t'):
        path = folder / name
        if not path.exists():
            temporary = folder / (name + '.download')
            subprocess.run(['curl', '--fail', '--location', '--retry', '2',
                '--max-time', '60',
                'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/' + name,
                '-o', str(temporary)], check=True)
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if digest != expected[name]:
                raise ValueError(f'Unexpected SHA-256 for downloaded {name}')
            temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected[name]:
            raise ValueError(f'Existing {name} differs from the recorded dataset')
        print(f'PASS {name}: SHA-256 {digest}')

if __name__ == '__main__':
    main()
