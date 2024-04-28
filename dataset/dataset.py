from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
import nusacrowd as nc
import torch
import json
import pdb



class NERDataset(Dataset):
  def __init__(self, texts, labels, max_length, tokenizer):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = self.texts[idx]
    word_labels = self.labels[idx]
    encoded = {}
    labels = []
    tokens = []
    input_ids = []
    attention_mask = []

    bos_token_id = self.tokenizer.bos_token_id
    eos_token_id = self.tokenizer.eos_token_id

    bos_token = self.tokenizer.bos_token
    eos_token = self.tokenizer.eos_token
    pad_token = self.tokenizer.pad_token

    for word_idx, word in enumerate(text):
        word_encoded = self.tokenizer.tokenize(word)
        total_sub_words = len(word_encoded)
        tokens.extend(word_encoded)
        input_ids.extend(self.tokenizer.convert_tokens_to_ids(word_encoded))
        attention_mask.extend([1] * total_sub_words)
        labels.extend([word_labels[word_idx]]*total_sub_words)


    # add padding to right tail of the sentence
    if len(input_ids) < self.max_length - 2:
        input_ids = (
            [bos_token_id] + input_ids + [eos_token_id] + [1] * (self.max_length - len(input_ids) - 2)
        )
        padded_labels = [-100] + labels + [-100] * (self.max_length - len(labels) - 1)

        attention_mask = [1] + attention_mask + [1] + [0] * (self.max_length - len(attention_mask) - 2)
        tokens = [bos_token]+ tokens + [eos_token] + [pad_token] * (self.max_length - len(tokens) - 2)
    else:
        input_ids = [bos_token_id] + input_ids[: self.max_length - 2] + [eos_token_id]
        padded_labels = [-100] + labels[: self.max_length - 2] + [-100]
        attention_mask = [1] + attention_mask[: self.max_length - 2] + [1]
        tokens = [bos_token] + tokens[: self.max_length - 2] + [eos_token]
    encoded["input_ids"] = torch.tensor(input_ids)
    encoded["attention_mask"] = torch.tensor(attention_mask)
    encoded["labels"] = torch.tensor(padded_labels)
    return encoded
    

class NERT5Dataset(Dataset):
    def __init__(self, texts, labels, max_length, tokenizer):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        word_labels = self.labels[idx]
        sentinel_tags_tokens = []
        sentinel_tags_labels = []
        sentinel_id_format = "<extra_id_{idx}>"
        for word_idx, word in enumerate(text):
            sentinel_id = sentinel_id_format.format(idx=word_idx)
            sentinel_tag_token = f"{sentinel_id} {word}"
            sentinel_tag_label = f"{sentinel_id} {word_labels[word_idx]}"
            sentinel_tags_tokens.append(sentinel_tag_token)
            sentinel_tags_labels.append(sentinel_tag_label)
        input_sentence = " ".join(sentinel_tags_tokens)
        output_sentence = " ".join(sentinel_tags_labels)

        encoded_input = self.tokenizer.encode_plus(
            input_sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        with self.tokenizer.as_target_tokenizer(): 
            encoded_output = self.tokenizer.encode_plus(
                output_sentence,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        encoded_input["labels"] = encoded_output["input_ids"]
        
        return {
            'input_ids': encoded_input['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': encoded_input['attention_mask'].squeeze(0),
            'labels': encoded_input['labels'].squeeze(0)
        }
        

def convert_to_idx(labels: list) -> list:
    label_2_idx = {label: idx for idx, label in enumerate(list(set(labels)))}
    return [label_2_idx[label] for label in labels]


def load_dataset_with_certain_loader(loader_type, dataset_name, split, dataset_subset):
    if loader_type == 'hf':
        if dataset_subset == '':
            dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_name, dataset_subset, split=split, trust_remote_code=True)
    else:
        dataset = nc.load_dataset(dataset_name)
        dataset = dataset[split]
    
    return dataset

    

def add_to_dataset(dataset_dict, dataset_name, dataset_subset, loader_type
, text_field, text_values_field, label_field, label_values_field, split, label_mapping_file):
    dataset = load_dataset_with_certain_loader(loader_type, dataset_name, split, dataset_subset)
    dataset_len = len(dataset)
    if label_mapping_file != '':
        # align the label with the labels in test set
        with open(label_mapping_file, 'r') as f:
            id2id = json.load(f)
        #pdb.set_trace()
        label_field_values = []
        for label_instances in dataset[label_values_field]:
            label_instance = [int(id2id[str(label)]) for label in label_instances]
            label_field_values.append(label_instance)
    else:
        label_field_values = dataset[label_values_field]
    text_field_values = dataset[text_values_field]
    dataset_dict[text_field].extend(text_field_values)
    dataset_dict[label_field].extend(label_field_values)
    dataset_dict['len'].append(dataset_len)


def load_just_dataset(dataset_cfg, tokenizer_cfg, tokenizer, task_type):
    text_field = dataset_cfg.test_text_field
    label_field = dataset_cfg.test_label_field
    train_dataset = {
        text_field:[],
        label_field:[],
        'len': []
    }
    val_dataset = {
        text_field: [],
        label_field:[],
        'len': []
    }
    for idx in range(len(dataset_cfg.train_dataset_names)):
        print("Training dataset!")
        add_to_dataset(train_dataset, dataset_cfg.train_dataset_names[idx], dataset_cfg.train_subsets[idx]
        , dataset_cfg.train_loader_types[idx], text_field, dataset_cfg.train_text_fields[idx]
        , label_field, dataset_cfg.train_label_fields[idx], dataset_cfg.train_train_splits[idx], dataset_cfg.train_label_mapping_files[idx])
        print()
        print("Validation dataset!") 
        add_to_dataset(val_dataset, dataset_cfg.train_dataset_names[idx], dataset_cfg.train_subsets[idx]
        , dataset_cfg.train_loader_types[idx], text_field, dataset_cfg.train_text_fields[idx]
        , label_field, dataset_cfg.train_label_fields[idx], dataset_cfg.train_val_splits[idx], dataset_cfg.train_label_mapping_files[idx])
    print("Dataset length")
    print(train_dataset['len'])
    print(val_dataset['len'])
    test_dataset = load_dataset_with_certain_loader(dataset_cfg.test_loader_type, dataset_cfg.test_dataset_name, dataset_cfg.test_split, dataset_cfg.test_subset)

    if task_type == 'seq2seq':
        train_dataset = NERT5Dataset(train_dataset[text_field], train_dataset[label_field], tokenizer_cfg.max_length, tokenizer)
        val_dataset = NERT5Dataset(val_dataset[text_field], val_dataset[label_field], tokenizer_cfg.max_length, tokenizer)
        test_dataset = NERT5Dataset(test_dataset[text_field], test_dataset[label_field], tokenizer_cfg.max_length, tokenizer)
    else:
        train_dataset = NERDataset(train_dataset[text_field], train_dataset[label_field], tokenizer_cfg.max_length, tokenizer)
        val_dataset = NERDataset(val_dataset[text_field], val_dataset[label_field], tokenizer_cfg.max_length, tokenizer)
        test_dataset = NERDataset(test_dataset[text_field], test_dataset[label_field], tokenizer_cfg.max_length, tokenizer)
    return train_dataset, val_dataset, test_dataset

def load_dataloader(dataset_cfg, tokenizer_cfg, tokenizer, task_type='default'):
    train_dataset, val_dataset, test_dataset = load_just_dataset(dataset_cfg, tokenizer_cfg, tokenizer, task_type)
    id2label_file = dataset_cfg.test_id2label_file

    with open(id2label_file, 'r') as f:
        id2label = json.load(f) 

    if task_type == 'seq2seq':
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    else:
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
    train_dl = DataLoader(train_dataset, batch_size=dataset_cfg.batch_size, collate_fn=collator, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=dataset_cfg.batch_size,  collate_fn=collator)
    test_dl = DataLoader(test_dataset, batch_size=dataset_cfg.batch_size, collate_fn=collator)
    return train_dl, val_dl, test_dl, id2label
