"""USBF local command line. No scheduler, daemon or automatic background work."""
import argparse,json,sys,time
from pathlib import Path

def doctor():
    import platform,importlib.metadata as m
    import rasterio,pyproj,shapely,torch,cv2
    packages={n:m.version(n) for n in ('usbf-reproducible','torch','torchvision','numpy','scipy','rasterio','pyproj',
        'shapely','geopandas','pyogrio','pandas','pyarrow','scikit-image','opencv-python-headless')}
    return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=packages,
        gdal=rasterio.__gdal_version__,proj=pyproj.proj_version_str,geos=shapely.geos_version_string,
        cuda_available=torch.cuda.is_available(),opencv=cv2.__version__)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor')
    f=sub.add_parser('fetch');f.add_argument('--data-dir',type=Path,required=True);f.add_argument('--manifest');f.add_argument('--offline',action='store_true');f.add_argument('--full-inputs',action='store_true')
    v=sub.add_parser('verify');v.add_argument('--data-dir',type=Path,required=True)
    for name in ('generate','example'):
        c=sub.add_parser(name);c.add_argument('--data-dir',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
        c.add_argument('--device',choices=('cpu','cuda'),default='cpu');c.add_argument('--threads',type=int,default=2)
        c.add_argument('--historical-probabilities',action='store_true',help='Regression only; not a model end-to-end run')
        c.add_argument('--simulate-missing-height',action='store_true',help='Explicit robustness control; not a new reference evaluation')
        c.add_argument('--dataset',choices=('example','full'),default='example')
    e=sub.add_parser('evaluate');e.add_argument('--data-dir',type=Path,required=True);e.add_argument('--output',type=Path,required=True);e.add_argument('--example-product',type=Path);e.add_argument('--generated-product',type=Path)
    r=sub.add_parser('register');r.add_argument('--data-dir',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    b=sub.add_parser('regression');b.add_argument('--data-dir',type=Path,required=True);b.add_argument('--product',type=Path,required=True);b.add_argument('--output',type=Path,required=True)
    b=sub.add_parser('preview');b.add_argument('--data-dir',type=Path,required=True);b.add_argument('--product',type=Path,required=True);b.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('prepare');a.add_argument('--rgb',type=Path,required=True);a.add_argument('--dsm',type=Path);a.add_argument('--dtm',type=Path);a.add_argument('--output',type=Path,required=True)
    t=sub.add_parser('train');t.add_argument('--data-dir',type=Path,required=True);t.add_argument('--output',type=Path,required=True)
    t.add_argument('--manifest',type=Path);t.add_argument('--parent',type=Path);t.add_argument('--arm',choices=('B0_control','B1_shape'),default='B1_shape')
    t.add_argument('--device',choices=('cpu','cuda'),default='cpu');t.add_argument('--epochs',type=int,default=60);t.add_argument('--max-updates',type=int,default=100)
    t.add_argument('--batch-size',type=int,default=8);t.add_argument('--software-smoke',action='store_true')
    args=p.parse_args(argv)
    if args.command=='doctor':result=doctor()
    elif args.command=='fetch':
        from .data.download import fetch
        result=fetch(args.data_dir,args.manifest,args.offline,args.full_inputs)
    elif args.command=='verify':
        from .io.common import verify_files
        result=verify_files(args.data_dir)
    elif args.command in ('generate','example'):
        from .refinement.pipeline import generate
        if args.command=='example' and args.dataset!='example':p.error('Use generate --dataset full, followed by evaluate --generated-product')
        result=generate(args.data_dir,args.output,args.device,args.threads,args.historical_probabilities,args.simulate_missing_height,args.dataset)
        if args.command=='example':
            from .evaluation.replay import replay
            result['evaluation']=replay(args.data_dir,args.output/'evaluation',args.output/'USBF.parquet')
            from .io.common import save
            save(args.output/'END_TO_END.json',result)
    elif args.command=='evaluate':
        from .evaluation.replay import replay
        result=replay(args.data_dir,args.output,args.example_product,args.generated_product)
    elif args.command=='prepare':
        from .data.sensors import prepare
        result=prepare(args.rgb,args.output,args.dsm,args.dtm)
    elif args.command=='register':
        from .evaluation.registration import run
        result=run(args.data_dir,args.output)
    elif args.command=='regression':
        from .evaluation.regression import compare
        result=compare(args.data_dir,args.product,args.output)
    elif args.command=='preview':
        from .io.preview import plot
        result=plot(args.data_dir,args.product,args.output)
    elif args.command=='train':
        from .training.train import train,smoke
        if args.software_smoke:result=smoke(args.data_dir,args.output,args.arm)
        else:
            if not args.manifest or not args.parent:p.error('Training requires --manifest and the exact --parent checkpoint; see docs/training.md')
            result=train(args.manifest,args.data_dir,args.parent,args.output,args.arm,args.device,args.epochs,args.max_updates,args.batch_size)
    from .io.common import native
    print(json.dumps(result,indent=2,default=native,allow_nan=False))
    return 1 if result.get('status')=='geometry_regression_difference' else 0
