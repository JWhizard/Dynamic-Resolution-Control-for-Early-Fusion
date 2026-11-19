"""Verify captured frame bytes against the hashes recorded with a manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('manifest',type=Path);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());checked=0
    for row in manifest['frames']:
        for sensor in ['rgb','thermal']:
            expected=row.get(sensor+'_sha256')
            if not expected:raise ValueError(f'missing recorded SHA-256: frame {row["id"]} {sensor}')
            actual=hashlib.sha256((a.manifest.parent/row[sensor]).read_bytes()).hexdigest()
            if actual!=expected:raise ValueError(f'frame bytes changed: {row["id"]} {sensor}')
            checked+=1
    print(f'Verified {checked} frame files against captured SHA-256 values')


if __name__=='__main__':main()
