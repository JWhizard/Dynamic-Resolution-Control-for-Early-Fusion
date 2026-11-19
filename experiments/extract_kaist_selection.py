"""Read selected JPEG/XML/metadata files from a checksum-verified official tar.

Never extracts symlinks or archive code. All destination names come from the
predeclared image selection, not arbitrary archive paths.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ['archive','selection','verification','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();selection=json.loads(a.selection.read_text());proof=json.loads(a.verification.read_text())
    stat=a.archive.stat()
    if proof['actual_md5']!=proof['publisher_md5'] or proof['bytes']!=stat.st_size or proof['mtime_ns']!=stat.st_mtime_ns:
        raise ValueError('archive verification mismatch or source changed')
    wanted={}
    for seq in selection['sequences']:
        for name in seq['frames']:
            path=Path(name)
            for modality in ['visible','lwir']:
                r=Path('images')/path.parent/modality/(path.name+'.jpg');wanted['kaist-cvpr15/'+str(r)]=r
            r=Path('annotations-xml-new-sanitized')/path.with_suffix('.xml');wanted['kaist-cvpr15/'+str(r)]=r
    for split in ['train-all-01','test-all-01','test-day-01','test-night-01','test-all-20']:
        r=Path('imageSets')/(split+'.txt');wanted['kaist-cvpr15/'+str(r)]=r
    a.output.mkdir(parents=True,exist_ok=False);records=[]
    with tarfile.open(a.archive,'r:') as archive:
        for member in archive:
            if member.name not in wanted:continue
            if not member.isfile():raise ValueError('selected archive item is not a regular file')
            relative=wanted[member.name];destination=a.output/relative
            if '..' in relative.parts or relative.is_absolute():raise ValueError('unsafe destination')
            destination.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256()
            with archive.extractfile(member) as src,destination.open('xb') as dst:
                for block in iter(lambda:src.read(1024*1024),b''):h.update(block);dst.write(block)
            records.append(dict(path=str(relative),bytes=member.size,sha256=h.hexdigest()))
            if len(records)%1000==0:print('Extracted',len(records),'/',len(wanted),flush=True)
    if len(records)!=len(wanted):raise ValueError('missing selected archive members')
    if a.archive.stat().st_mtime_ns!=stat.st_mtime_ns:raise ValueError('archive changed during extraction')
    (a.output/'EXTRACTION.json').write_text(json.dumps(dict(archive=str(a.archive),verification=proof,
        selection_sha256=hashlib.sha256(a.selection.read_bytes()).hexdigest(),files=records),indent=2))
    (a.output/'selection.json').write_text(a.selection.read_text());print('Extracted',len(records),'verified regular files',flush=True)


if __name__=='__main__':main()
