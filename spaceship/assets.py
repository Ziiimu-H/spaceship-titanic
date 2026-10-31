"""Download the reviewed public probability assets without executing source code."""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen
import zipfile

ROOT=Path(__file__).resolve().parents[1]
API='https://www.kaggle.com/api/v1/kernels/pull/kalidenaiarys/spaceship-titanic-tabpfn-catboost-lb-0-82534'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'.cache/tabpfn')
    args=parser.parse_args()
    expected=json.loads((ROOT/'reference/asset_hashes.json').read_text())
    with urlopen(Request(API,headers={'User-Agent':'spaceship-titanic-replay/1.0'}),timeout=60) as response:
        payload=json.load(response)
    source=payload['blob']['source']
    notebook=json.loads(source)
    archive=None
    for cell in notebook['cells']:
        if cell['cell_type']!='code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MODEL_PROBABILITY_ASSETS' for t in node.targets):
                archive=ast.literal_eval(node.value)
    if not isinstance(archive,str):
        raise ValueError('The source no longer contains the reviewed probability archive. Do not skip integrity checks.')
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(archive,validate=True))) as bundle:
        if set(bundle.namelist())!=set(expected):
            raise ValueError('Source asset list changed; review the upstream release before proceeding.')
        contents={name:bundle.read(name) for name in expected}
        for name,checksum in expected.items():
            path=Path(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe archive path')
            if hashlib.sha256(contents[name]).hexdigest()!=checksum:
                raise ValueError(f'Public asset changed: {name}')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,data in contents.items():
        dest=args.output_dir/name
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    for name in ['asset_hashes.json','ATTRIBUTION.md']:
        shutil.copyfile(ROOT/'reference'/name,args.output_dir/name)
    provenance=json.loads((ROOT/'reference/provenance.json').read_text())
    provenance['reference_template_source_sha256']=provenance['source_notebook_sha256']
    provenance['source_notebook_sha256']=hashlib.sha256(source.encode()).hexdigest()
    provenance['source_notebook_version']=payload['metadata']['currentVersionNumber']
    (args.output_dir/'source_notebook.ipynb').write_text(source)
    (args.output_dir/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'Downloaded and verified {len(expected)} assets to {args.output_dir}. Source code was parsed as data, not executed.')

if __name__=='__main__':
    main()
