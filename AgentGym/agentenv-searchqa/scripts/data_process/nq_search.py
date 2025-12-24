# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the nq dataset to parquet format
"""

import os
import datasets
import json
import argparse


def make_prefix(dp, template_type):
    question = dp['question']
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'nq'

    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            example['question'] = example['question'].strip()
            if example['question'][-1] != '?':
                example['question'] += '?'
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target": example['golden_answers'],
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir

    os.makedirs(local_dir, exist_ok=True)

    # train_dataset.to_json(os.path.join(local_dir, 'train.json'),lines=False, indent=2)
    # test_dataset.to_json(os.path.join(local_dir, 'test.json'), orient='records', lines=False, indent=2)
    
    test_data = test_dataset.to_list()
    with open(os.path.join(local_dir, 'searchqa_test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
        
    train_data = train_dataset.to_list()
    with open(os.path.join(local_dir, 'searchqa_train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

