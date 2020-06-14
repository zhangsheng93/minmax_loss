#! /bin/sh
python experiments/transfer/transfer_prepro.py
python prepro_std.py --model bert-base-uncased --root_dir data/transfer_data --task_def experiments/transfer/transfer_task_def.yml --do_lower_case
