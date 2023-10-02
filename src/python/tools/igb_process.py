import argparse
import os
import shutil

from pathlib import Path

from marius.tools.preprocess import custom
from preprocess.datasets import *
from marius.tools.configuration.constants import PathConstants


def set_args():
    parser = argparse.ArgumentParser(
                description='Preprocess Datasets', prog='preprocess')

    parser.add_argument('--output_directory',
                        metavar='output_directory',
                        type=str,
                        default="/scratch/yw8143/dataset/ogbn_arxiv",
                        help='Directory to put graph data')

    parser.add_argument('--edges',
                        metavar='edges',
                        nargs='+',
                        type=str,
                        help='File(s) containing the edge list(s) for a custom dataset')

    parser.add_argument('--nodes',
                        metavar='nodes',
                        nargs='+',
                        type=str,
                        help='File(s) containing the node ids for a custom dataset')

    parser.add_argument('--features',
                        metavar='features',
                        nargs='+',
                        type=str,
                        help='File(s) containing node features for a custom dataset')

    parser.add_argument('--labels',
                        metavar='labels',
                        nargs='+',
                        type=str,
                        help='File(s) containing node labels for a custom node classification dataset')

    parser.add_argument('--dataset',
                        metavar='dataset',
                        type=str,
                        default="custom",
                        help='Name of dataset to preprocess')

    parser.add_argument('--num_partitions',
                        metavar='num_partitions',
                        required=False,
                        type=int,
                        default=1,
                        help='Number of node partitions')

    parser.add_argument('--delim',
                        '-d',
                        metavar='delim',
                        type=str,
                        default="\t",
                        help='Delimiter to use for delimited file inputs')

    parser.add_argument('--dataset_split',
                        '-ds',
                        metavar='dataset_split',
                        nargs='+',
                        type=float,
                        default=None,
                        help='Split dataset into specified fractions')

    parser.add_argument('--overwrite',
                        metavar='overwrite',
                        type=bool,
                        default=False,
                        help='If true, the preprocessed dataset will be overwritten if it already exists')

    parser.add_argument('--spark',
                        metavar='spark',
                        type=bool,
                        default=False,
                        help='If true, pyspark will be used to perform the preprocessing')

    parser.add_argument('--no_remap_ids',
                        action='store_true',
                        default=False,
                        help='If true, the node ids of the input dataset will not be remapped to random integer ids')

    parser.add_argument('--sequential_train_nodes',
                        action='store_true',
                        default=False,
                        help='If true, the train nodes will be given ids 0 to num train nodes')
    parser.add_argument('--dataset_type', type=str, default='homogeneous',
        choices=['homogeneous', 'heterogeneous'], 
        help='dataset type')
    parser.add_argument('--dataset_size', type=str, default='tiny',
        choices=['tiny', 'small', 'medium'], 
        help='size of the datasets')
    ## add train/val/test split as a list
    parser.add_argument('--train_split', type=float, default=0.1,
        help='train split')
    parser.add_argument('--val_split', type=float, default=0.1,
        help='val split')
    parser.add_argument('--test_split', type=float, default=0.8,
        help='test split')

    return parser


def main():
    parser = set_args()
    args = parser.parse_args()
    args.dataset="IGB"
    args.sequential_train_nodes=True
    args.num_partitions=100
    args.overwrite=False
    
    # args.output_directory=f"/scratch/yw8143/mariusdataset/IGB_{args.dataset_type}_{args.dataset_size}_{args.num_partitions}"
    args.output_directory=f"/home/yw8143/marius_artifact/datasets/IGB_{args.dataset_type}_{args.dataset_size}_{args.num_partitions}"
    if args.output_directory is "":
        args.output_directory = args.dataset

    if args.overwrite and Path(args.output_directory).exists():
        shutil.rmtree(args.output_directory)
    dataset_dict = {
        "FB15K": fb15k.FB15K,
        "FB15K_237": fb15k_237.FB15K237,
        "LIVEJOURNAL": livejournal.Livejournal,
        "TWITTER": twitter.Twitter,
        "FREEBASE86M": freebase86m.Freebase86m,
        "OGBL_WIKIKG2": ogbl_wikikg2.OGBLWikiKG2,
        "OGBL_CITATION2": ogbl_citation2.OGBLCitation2,
        "OGBL_PPA": ogbl_ppa.OGBLPpa,
        "OGBN_ARXIV": ogbn_arxiv.OGBNArxiv,
        "OGBN_PRODUCTS": ogbn_products.OGBNProducts,
        # "OGBN_MAG": ogbn_mag.OGBNMag,
        "OGBN_PAPERS100M": ogbn_papers100m.OGBNPapers100M,
        "OGB_WIKIKG90MV2": ogb_wikikg90mv2.OGBWikiKG90Mv2,
        "OGB_MAG240M": ogb_mag240m.OGBMag240M,
        'IGB': igb.IGB
    }

    dataset = dataset_dict.get(args.dataset.upper())
    if args.dataset=="IGB":
        dataset.dataset_type=args.dataset_type
        dataset.dataset_size=args.dataset_size
        dataset.train_percentage=0.1
        dataset.valid_percentage=0.1
        dataset.test_percentage=0.8
    if dataset is not None:
        dataset = dataset(args.output_directory)
        dataset.download(args.overwrite)
        dataset.preprocess(num_partitions=args.num_partitions,
                           remap_ids=not args.no_remap_ids,
                           splits=args.dataset_split,
                           sequential_train_nodes=args.sequential_train_nodes)
    else:
        print("Preprocess custom dataset")

        # custom link prediction dataset
        dataset = custom.CustomLinkPredictionDataset(
            output_directory=args.output_directory,
            files=args.edges,
            delim=args.delim,
            dataset_name=args.dataset
        )
        dataset.preprocess(num_partitions=args.num_partitions,
                           remap_ids=not args.no_remap_ids,
                           splits=args.dataset_split)


if __name__ == "__main__":
    main()
