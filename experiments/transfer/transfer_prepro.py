import os
import argparse
import random
from sys import path

path.append(os.getcwd())
from experiments.common_utils import dump_rows
from data_utils.task_def import DataFormat
from data_utils.log_wrapper import create_logger
from experiments.transfer.transfer_utils import *

logger = create_logger(__name__, to_disk=True, log_file='glue_prepro.log')


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing GLUE/SNLI/SciTail dataset.')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--root_dir', type=str, default='data')
    args = parser.parse_args()
    return args


def main(args):
    
    root = args.root_dir
    assert os.path.exists(root)

    ######################################
    # Transfer Tasks
    ######################################
    # scitail_train_path = os.path.join(root, 'SciTail/tsv_format/scitail_1.0_train.tsv')
    # scitail_dev_path = os.path.join(root, 'SciTail/tsv_format/scitail_1.0_dev.tsv')
    # scitail_test_path = os.path.join(root, 'SciTail/tsv_format/scitail_1.0_test.tsv')

    # snli_train_path = os.path.join(root, 'SNLI/train.tsv')
    # snli_dev_path = os.path.join(root, 'SNLI/dev.tsv')
    # snli_test_path = os.path.join(root, 'SNLI/test.tsv')

    # winogrande_xl_train_path = os.path.join(root, 'winogrande/train_xl.jsonl')
    # winogrande_l_train_path = os.path.join(root, 'winogrande/train_l.jsonl')
    # winogrande_m_train_path = os.path.join(root, 'winogrande/train_m.jsonl')
    # winogrande_s_train_path = os.path.join(root, 'winogrande/train_s.jsonl')
    # winogrande_xs_train_path = os.path.join(root, 'winogrande/train_xs.jsonl')
    # winogrande_dev_path = os.path.join(root, 'winogrande/dev.jsonl')
    # winogrande_test_path = os.path.join(root, 'winogrande/test.jsonl')

    # amazon_train_path = os.path.join(root, 'amazon/train.csv')
    # amazon_dev_path = os.path.join(root, 'amazon/test.csv')

    # yelp_train_path = os.path.join(root, 'yelp/train.csv')
    # yelp_dev_path = os.path.join(root, 'yelp/test.csv')

    # sicke_train_path = os.path.join(root, 'SICK/SICK.txt')
    # sicke_dev_path = os.path.join(root, 'SICK/SICK.txt')

    # sickr_train_path = os.path.join(root, 'SICK/SICK.txt')
    # sickr_dev_path = os.path.join(root, 'SICK/SICK.txt')

    # imdb_train_path = os.path.join(root, 'IMDB/imdb_master.csv')
    # imdb_dev_path = os.path.join(root, 'IMDB/imdb_master.csv')

    # wikiqa_train_path = os.path.join(root, 'wikiqa/WikiQA-train.tsv')
    # wikiqa_dev_path = os.path.join(root, 'wikiqa/WikiQA-dev.tsv')
    # wikiqa_test_path = os.path.join(root, 'wikiqa/WikiQA-test.tsv')

    cr_dev_path = os.path.join(root, 'CR/cr.txt')
    mr_dev_path = os.path.join(root, 'MR/mr.txt')

    ######################################
    # Loading DATA
    ######################################
    # scitail_train_data = load_scitail(scitail_train_path)
    # scitail_dev_data = load_scitail(scitail_dev_path)
    # scitail_test_data = load_scitail(scitail_test_path)
    # logger.info('Loaded {} SciTail train samples'.format(len(scitail_train_data)))
    # logger.info('Loaded {} SciTail dev samples'.format(len(scitail_dev_data)))
    # logger.info('Loaded {} SciTail test samples'.format(len(scitail_test_data)))

    # snli_train_data = load_snli(snli_train_path)
    # snli_dev_data = load_snli(snli_dev_path)
    # snli_test_data = load_snli(snli_test_path)
    # logger.info('Loaded {} SNLI train samples'.format(len(snli_train_data)))
    # logger.info('Loaded {} SNLI dev samples'.format(len(snli_dev_data)))
    # logger.info('Loaded {} SNLI test samples'.format(len(snli_test_data)))

    # winogrande_xl_train_data = load_winogrande(winogrande_xl_train_path)
    # winogrande_l_train_data = load_winogrande(winogrande_l_train_path)
    # winogrande_m_train_data = load_winogrande(winogrande_m_train_path)
    # winogrande_s_train_data = load_winogrande(winogrande_s_train_path)
    # winogrande_xs_train_data = load_winogrande(winogrande_xs_train_path)
    # winogrande_dev_data = load_winogrande(winogrande_dev_path)
    # winogrande_test_data = load_winogrande(winogrande_test_path, is_train=False)

    # logger.info('Loaded {} Winogrande train xl samples'.format(len(winogrande_xl_train_data)))
    # logger.info('Loaded {} Winogrande train l samples'.format(len(winogrande_l_train_data)))
    # logger.info('Loaded {} Winogrande train m samples'.format(len(winogrande_m_train_data)))
    # logger.info('Loaded {} Winogrande train s samples'.format(len(winogrande_s_train_data)))
    # logger.info('Loaded {} Winogrande train xs samples'.format(len(winogrande_xs_train_data)))
    # logger.info('Loaded {} Winogrande dev samples'.format(len(winogrande_dev_data)))
    # logger.info('Loaded {} Winogrande test samples'.format(len(winogrande_test_data)))


    # sicke_train_data = load_sicke(sicke_train_path)
    # sicke_dev_data = load_sicke(sicke_dev_path, is_train=False)
    # logger.info('Loaded {} SICK-E train samples'.format(len(sicke_train_data)))
    # logger.info('Loaded {} SICK-E dev samples'.format(len(sicke_dev_data)))


    # sickr_train_data = load_sickr(sickr_train_path)
    # sickr_dev_data = load_sickr(sickr_dev_path, is_train=False)
    # logger.info('Loaded {} SICK-R train samples'.format(len(sickr_train_data)))
    # logger.info('Loaded {} SICK-R dev samples'.format(len(sickr_dev_data)))

    # imdb_train_data = load_imdb(imdb_train_path)
    # imdb_dev_data = load_imdb(imdb_dev_path, is_train=False)
    # logger.info('Loaded {} IMDB train samples'.format(len(imdb_train_data)))
    # logger.info('Loaded {} IMDB dev samples'.format(len(imdb_dev_data)))

    # wikiqa_train_data = load_wikiqa(wikiqa_train_path)
    # wikiqa_dev_data = load_wikiqa(wikiqa_dev_path)
    # wikiqa_test_data = load_wikiqa(wikiqa_test_path)
    # logger.info('Loaded {} wikiqa train samples'.format(len(wikiqa_train_data)))
    # logger.info('Loaded {} wikiqa dev samples'.format(len(wikiqa_dev_data)))
    # logger.info('Loaded {} wikiqa test samples'.format(len(wikiqa_test_data)))


    # yelp_train_data = load_yelp(yelp_train_path)
    # yelp_dev_data = load_yelp(yelp_dev_path)
    # logger.info('Loaded {} yelp train samples'.format(len(yelp_train_data)))
    # logger.info('Loaded {} yelp test samples'.format(len(yelp_dev_data)))

    # amazon_train_data = load_amazon(amazon_train_path)
    # amazon_dev_data = load_amazon(amazon_dev_path)
    # logger.info('Loaded {} amazon train samples'.format(len(amazon_train_data)))
    # logger.info('Loaded {} amazon test samples'.format(len(amazon_dev_data)))

    cr_dev_data = load_cr(cr_dev_path)
    logger.info('Loaded {} CR test samples'.format(len(cr_dev_data)))

    mr_dev_data = load_cr(mr_dev_path)
    logger.info('Loaded {} MR test samples'.format(len(mr_dev_data)))

    ######################################
    # dump rows
    ######################################

    transfer_data_suffix = "transfer_data"
    transfer_data_root = os.path.join(root, transfer_data_suffix)
    if not os.path.isdir(transfer_data_root):
        os.mkdir(transfer_data_root)

    
    # scitail_train_fout = os.path.join(transfer_data_root, 'scitail_train.tsv')
    # scitail_dev_fout = os.path.join(transfer_data_root, 'scitail_dev.tsv')
    # scitail_test_fout = os.path.join(transfer_data_root, 'scitail_test.tsv')
    # dump_rows(scitail_train_data, scitail_train_fout, DataFormat.PremiseAndOneHypothesis)
    # dump_rows(scitail_dev_data, scitail_dev_fout, DataFormat.PremiseAndOneHypothesis)
    # dump_rows(scitail_test_data, scitail_test_fout, DataFormat.PremiseAndOneHypothesis)
    # logger.info('done with scitail')


    # snli_train_fout = os.path.join(transfer_data_root, 'snli_train.tsv')
    # snli_dev_fout = os.path.join(transfer_data_root, 'snli_dev.tsv')
    # snli_test_fout = os.path.join(transfer_data_root, 'snli_test.tsv')
    # dump_rows(snli_train_data, snli_train_fout, DataFormat.PremiseAndOneHypothesis)
    # dump_rows(snli_dev_data, snli_dev_fout, DataFormat.PremiseAndOneHypothesis)
    # dump_rows(snli_test_data, snli_test_fout, DataFormat.PremiseAndOneHypothesis)
    # logger.info('done with snli')

    # winogrande_xl_train_fout = os.path.join(transfer_data_root, 'winogrande_xl_train.tsv')
    # winogrande_l_train_fout = os.path.join(transfer_data_root, 'winogrande_l_train.tsv')
    # winogrande_m_train_fout = os.path.join(transfer_data_root, 'winogrande_m_train.tsv')
    # winogrande_s_train_fout = os.path.join(transfer_data_root, 'winogrande_s_train.tsv')
    # winogrande_xs_train_fout = os.path.join(transfer_data_root, 'winogrande_xs_train.tsv')
    # winogrande_dev_fout = os.path.join(transfer_data_root, 'winogrande_dev.tsv')
    # winogrande_test_fout = os.path.join(transfer_data_root, 'winogrande_test.tsv')

    # dump_rows(winogrande_xl_train_data, winogrande_xl_train_fout, DataFormat.PremiseAndOneHypothesis)
    # dump_rows(winogrande_l_train_data, winogrande_l_train_fout, DataFormat.PremiseAndOneHypothesis)
    # dump_rows(winogrande_m_train_data, winogrande_m_train_fout, DataFormat.PremiseAndOneHypothesis)
    # dump_rows(winogrande_s_train_data, winogrande_s_train_fout, DataFormat.PremiseAndOneHypothesis)
    # dump_rows(winogrande_xs_train_data, winogrande_xs_train_fout, DataFormat.PremiseAndOneHypothesis)
    # dump_rows(winogrande_dev_data, winogrande_dev_fout, DataFormat.PremiseAndOneHypothesis)
    # dump_rows(winogrande_test_data, winogrande_test_fout, DataFormat.PremiseAndOneHypothesis)
    # logger.info('done with winogrande')


    # yelp_train_fout = os.path.join(transfer_data_root, 'yelp_train.tsv')
    # yelp_dev_fout = os.path.join(transfer_data_root, 'yelp_dev.tsv')
    # dump_rows(yelp_train_data, yelp_train_fout, DataFormat.PremiseOnly)
    # dump_rows(yelp_dev_data, yelp_dev_fout, DataFormat.PremiseOnly)
    # logger.info('done with yelp')

    # sicke_train_fout = os.path.join(transfer_data_root, 'sicke_train.tsv')
    # sicke_dev_fout = os.path.join(transfer_data_root, 'sicke_dev.tsv')
    # dump_rows(sicke_train_data, sicke_train_fout, DataFormat.PremiseAndOneHypothesis)
    # dump_rows(sicke_dev_data, sicke_dev_fout, DataFormat.PremiseAndOneHypothesis)
    # logger.info('done with SICK-E')

    # sickr_train_fout = os.path.join(transfer_data_root, 'sickr_train.tsv')
    # sickr_dev_fout = os.path.join(transfer_data_root, 'sickr_dev.tsv')
    # dump_rows(sickr_train_data, sickr_train_fout, DataFormat.PremiseAndOneHypothesis)
    # dump_rows(sickr_dev_data, sickr_dev_fout, DataFormat.PremiseAndOneHypothesis)
    # logger.info('done with SICK-R')

    # imdb_train_fout = os.path.join(transfer_data_root, 'imdb_train.tsv')
    # imdb_dev_fout = os.path.join(transfer_data_root, 'imdb_dev.tsv')
    # dump_rows(imdb_train_data, imdb_train_fout, DataFormat.PremiseOnly)
    # dump_rows(imdb_dev_data, imdb_dev_fout, DataFormat.PremiseOnly)
    # logger.info('done with IMDB')

    # wikiqa_train_fout = os.path.join(transfer_data_root, 'wikiqa_train.tsv')
    # wikiqa_dev_fout = os.path.join(transfer_data_root, 'wikiqa_dev.tsv')
    # wikiqa_test_fout = os.path.join(transfer_data_root, 'wikiqa_test.tsv')
    # dump_rows(wikiqa_train_data, wikiqa_train_fout, DataFormat.PremiseAndOneHypothesis)
    # dump_rows(wikiqa_dev_data, wikiqa_dev_fout, DataFormat.PremiseAndOneHypothesis)
    # dump_rows(wikiqa_test_data, wikiqa_test_fout, DataFormat.PremiseAndOneHypothesis)
    # logger.info('done with WikiQA')

    # amazon_train_fout = os.path.join(transfer_data_root, 'amazon_train.tsv')
    # amazon_dev_fout = os.path.join(transfer_data_root, 'amazon_dev.tsv')
    # dump_rows(amazon_train_data, amazon_train_fout, DataFormat.PremiseOnly)
    # dump_rows(amazon_dev_data, amazon_dev_fout, DataFormat.PremiseOnly)
    # logger.info('done with amazon')

    cr_dev_fout = os.path.join(transfer_data_root, 'cr_dev.tsv')
    dump_rows(cr_dev_data, cr_dev_fout, DataFormat.PremiseOnly)
    mr_dev_fout = os.path.join(transfer_data_root, 'mr_dev.tsv')
    dump_rows(mr_dev_data, mr_dev_fout, DataFormat.PremiseOnly)

    logger.info('done with cr & mr')

if __name__ == '__main__':
    args = parse_args()
    main(args)
