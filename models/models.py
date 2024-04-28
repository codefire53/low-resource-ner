from transformers import AutoModelForSeq2SeqLM, AutoModelForTokenClassification
import lightning as L
from torchmetrics import F1Score
from torch import nn
import torch
import evaluate
import re

class Seq2SeqModel(L.LightningModule):
    def _extract_token_labels(self, text: str) -> list:
        pattern = r"\<extra_id_[0-9]+\>\s*(\S+)"
        tokens = re.findall(pattern, text)
        return tokens


    def __init__(self, model: AutoModelForSeq2SeqLM, lr: float):
        super(Seq2SeqModel, self).__init__()
        self.model = model
        self.all_preds = []
        self.all_labels = []
        self.seqeval = evaluate.load('seqeval')

        self.lr = lr

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        logits, loss = output.logits, output.loss
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        # Generate predictions
        generated_tokens = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        # Remove -100 indices (labels for padding tokens)
        decoded_labels = [label.replace(self.tokenizer.pad_token, '') for label in decoded_labels]

        pred_labels = self._extract_token_labels(decoded_preds)
        gt_labels = self._extract_token_labels(decoded_labels)
        
        if len(pred_labels) < len(gt_labels):
            pred_labels = pred_labels +["#"]*(len(gt_labels)-len(pred_labels))
        elif len(pred_labels) > len(gt_labels):
            pred_labels = pred_labels[:len(gt_labels)]
        
        results = self.seqeval.compute(predictions=pred_labels, references=gt_labels)


        val_f1_score = results['overall_f1']
        self.log("val_f1", val_f1_score, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=True, prog_bar=True)
        
    
    def test_step(self, batch, batch_idx):
        # Generate predictions
        generated_tokens = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        # Remove -100 indices (labels for padding tokens)
        decoded_labels = [label.replace(self.tokenizer.pad_token, '') for label in decoded_labels]

        pred_labels = self._extract_token_labels(decoded_preds)
        gt_labels = self._extract_token_labels(decoded_labels)
        
        if len(pred_labels) < len(gt_labels):
            pred_labels = pred_labels +["O"]*(len(gt_labels)-len(pred_labels))
        elif len(pred_labels) > len(gt_labels):
            pred_labels = pred_labels[:len(gt_labels)]
        
        self.all_labels.extend(pred_labels)
        self.all_preds.extend(gt_labels)

    
    def test_step(self, batch, batch_idx):
        labels = batch['labels']
        output = self(**batch)
        logits, _ = output.logits, output.loss
        #log_proba = torch.nn.functional.log_softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        print(preds)
        print(labels)
        print()
        ner_preds, ner_labels = self._get_ner_preds_and_ner_labels(preds, labels)
        self.all_labels.extend(ner_labels)
        self.all_preds.extend(ner_preds)

    def on_test_epoch_end(self):
        results = self.seqeval.compute(predictions=self.all_preds, references=self.all_labels)
        print(results)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
  
class TokenClassificationModel(L.LightningModule):
    def __init__(self, model: AutoModelForTokenClassification, id2label: dict, lr: float):
        super(TokenClassificationModel, self).__init__()
        self.model = model
        self.id2label = id2label
        self.all_preds = []
        self.all_labels = []
        self.seqeval = evaluate.load('seqeval')

        self.num_label = len(list(self.id2label.keys()))
        self.lr = lr

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        logits, loss = output.logits, output.loss
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss

    def _get_ner_preds_and_ner_labels(self, preds, labels):
        batch_preds = []
        batch_labels = []
        
        for i in range(len(labels)):
            batch_label = []
            batch_pred = []
            for j in range(len(labels[i])):
                if labels[i][j] != -100:
                    batch_label.append(self.id2label[str(labels[i][j].detach().item())])
                    batch_pred.append(self.id2label[str(preds[i][j].detach().item())])
            batch_labels.append(batch_label)
            batch_preds.append(batch_pred)
        
        return batch_preds, batch_labels

    def validation_step(self, batch, batch_idx):
        labels = batch['labels']
        output = self(**batch)
        logits, loss = output.logits, output.loss
        preds = torch.argmax(logits, dim=-1)
        print(preds.shape)
        print(labels.shape)
        print()
        ner_preds, ner_labels = self._get_ner_preds_and_ner_labels(preds, labels)
        print(ner_preds)
        print(ner_labels)
        print()

        results = self.seqeval.compute(predictions=ner_preds, references=ner_labels)

        val_f1_score = results['overall_f1']
        self.log("val_f1", val_f1_score, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        labels = batch['labels']
        output = self(**batch)
        logits, _ = output.logits, output.loss
        #log_proba = torch.nn.functional.log_softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        print(preds)
        print(labels)
        print()
        ner_preds, ner_labels = self._get_ner_preds_and_ner_labels(preds, labels)
        self.all_labels.extend(ner_labels)
        self.all_preds.extend(ner_preds)

    def on_test_epoch_end(self):
        results = self.seqeval.compute(predictions=self.all_preds, references=self.all_labels)
        print(results)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer

