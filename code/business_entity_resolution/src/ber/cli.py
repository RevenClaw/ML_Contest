import argparse
import json
import logging
from .common import load_config


def main(argv=None):
    parser=argparse.ArgumentParser(description='Offline entity resolution. Training is an explicit separate command.')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('index'); p.add_argument('--data',required=True); p.add_argument('--split',choices=['train','test'],required=True); p.add_argument('--index',required=True); p.add_argument('--config',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--index',required=True); p.add_argument('--out',required=True); p.add_argument('--config',required=True)
    p=sub.add_parser('train'); p.add_argument('--prepared',required=True); p.add_argument('--out',required=True); p.add_argument('--config',required=True)
    p=sub.add_parser('predict'); p.add_argument('--index',required=True); p.add_argument('--model',required=True); p.add_argument('--out',required=True)
    p=sub.add_parser('validate'); p.add_argument('--index',required=True); p.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.command in ('index','prepare','predict'):
        from .text import require_fast_text
        require_fast_text()
    if args.command=='index':
        from .store import build_index
        build_index(args.data,args.split,args.index,load_config(args.config))
    elif args.command=='prepare':
        from .prepare import prepare
        prepare(args.index,args.out,load_config(args.config))
    elif args.command=='train':
        from .train import train
        train(args.prepared,args.out,load_config(args.config))
    elif args.command=='predict':
        from .predict import predict
        predict(args.index,args.model,args.out)
    else:
        from .validate import validate
        print(json.dumps(validate(args.index,args.output)))


if __name__=='__main__': main()
