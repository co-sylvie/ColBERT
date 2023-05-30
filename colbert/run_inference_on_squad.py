"""
Sample command: python colbert/run_inference_on_squad.py --amp --doc_maxlen 180 --mask-punctuation --checkpoint "/home/sylvie_cohere_ai/ColBERT/colbertv2.0/pytorch_model.bin"
"""
import os
import json
import random
from typing import List, Dict

import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
from colbert.utils.parser import Arguments
from colbert.evaluation.load_model import load_model


class Pcolour:
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def colorize(text, color):
    return f"{color}{text}{Pcolour.END}"


def blue(*args):
    return colorize(" ".join(map(str, list(args))), Pcolour.BLUE)


def bold(*args):
    return colorize(" ".join(map(str, list(args))), Pcolour.BOLD)


def green(*args):
    return colorize(" ".join(map(str, list(args))), Pcolour.GREEN)


def magenta(*args):
    return colorize(" ".join(map(str, list(args))), Pcolour.MAGENTA)


def red(*args):
    return colorize(" ".join(map(str, list(args))), Pcolour.RED)


def underline(*args):
    return colorize(" ".join(map(str, list(args))), Pcolour.UNDERLINE)


def yellow(*args):
    return colorize(" ".join(map(str, list(args))), Pcolour.YELLOW)


RAINBOW = [Pcolour.RED, Pcolour.YELLOW, Pcolour.GREEN, Pcolour.BLUE, Pcolour.MAGENTA]


def rainbow(*args):
    text = " ".join(map(str, list(args)))
    curr_colour = random.randint(0, len(RAINBOW) - 1)
    coloured_text = ""
    for char in text:
        coloured_text += f"{RAINBOW[curr_colour]}{char}"
        curr_colour = (curr_colour + 1) % (len(RAINBOW))
    coloured_text += Pcolour.END
    return coloured_text


def parse_squad_data(data_path: str):
    out_data = []
    with open(data_path, "r") as f:
        raw_data = json.load(f)["data"]
    for article in raw_data:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                out_data.append({"id": qa["id"], "question": qa["question"], "context": paragraph["context"], "answers": [qa["answers"][i]["text"] for i in range(len(qa["answers"]))]})
    return out_data


def parse_nq_data(data_path: str):
    out_data = []
    with open(data_path, "r") as f:
        raw_data = json.load(f)["data"]
    for line in raw_data:
        out_data.append({"id": line["id"], "question": line["question"], "context": line["context"]})
    return out_data


def get_snippet(sim_mtx: np.array, window_size: int, input_ids: List[int], tokenizer: AutoTokenizer):
  """
  Computes colbert scores for each overlapping window of window_size, and return the span with the highest score.
  Args:
    sim_mtx: (query_length, doc_length) matrix of cosine similarities between query and document tokens
    window_size: size of the sliding window
  """
  spans_with_scores = []
  for i in range(1, sim_mtx.shape[1] - window_size):
    curr_mtx = sim_mtx[:, i:i+window_size] # Shape: (32, window_size)
    score = np.sum(np.max(curr_mtx, axis=-1, keepdims=False), axis=-1, keepdims=False) # Shape: (32, window_size) -> (32,) -> (1,)
    spans_with_scores.append(([i, i+window_size], score))
  spans_with_scores.sort(key=lambda x: x[1], reverse=True)
  top_span = spans_with_scores[0]
  top_span_ids = input_ids[top_span[0][0]:top_span[0][1]]
  top_span_string = tokenizer.decode(top_span_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
  return top_span_string


def export_predicted_snippets(
    data: List[Dict[str, str]], 
    answer_strings: List[str],
    window_size: int,
    output_path: str="data"):
    """

    Args:
        input_data (Dict[Dict[str, str]]): List of dictionaries with keys "id", "question" and "context".
    """
    id_lst = [pair["id"] for pair in data]
    out_dict = {id: answer for id, answer in zip(id_lst, answer_strings)}
    json.dump(out_dict, open(os.path.join(output_path, f"colbert_predictions_ws={window_size}.json"), "w"))
    print(f"Wrote predictions to {os.path.join(output_path, f'colbert_predictions_ws={window_size}.json')}!")


if __name__ == "__main__":
    ckpt_path = "/home/sylvie_cohere_ai/ColBERT/colbertv2.0/pytorch_model.bin"
    data_path = "/home/sylvie_cohere_ai/ColBERT/data/dev-v1.1.json"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    top_k_docs = 10
    
    parser = Arguments(description='Exhaustive (slow, not index-based) evaluation of re-ranking with ColBERT.')

    parser.add_model_parameters()
    parser.add_model_inference_parameters()
    args = parser.parse()

    colbert_model, ckpt = load_model(args, do_print=False)
    tokenizer = colbert_model.tokenizer
    eval_data = parse_nq_data(data_path) # parse_squad_data(data_path)
    colbert_model.to(device)
    sample_eval_data = random.sample(eval_data, 5)
    df_lst = []
    answers_list = []
    window_size = 30
    for line in tqdm(eval_data, total=len(eval_data), desc="Getting snippets using ColBERT v1..."):
        tokenized_q = tokenizer(line["question"], return_tensors="pt", truncation=True, padding="max_length", max_length=colbert_model.query_maxlen).to(device)
        tokenized_doc = tokenizer(line["context"], return_tensors="pt", truncation=True, padding="max_length", max_length=colbert_model.doc_maxlen).to(device)
        with torch.no_grad():
            # Get normalized token embeddings
            Q = colbert_model.query(tokenized_q["input_ids"], attention_mask=tokenized_q["attention_mask"]) # (batch, max_length=32, dim=128)
            D = colbert_model.doc(tokenized_doc["input_ids"], attention_mask=tokenized_doc["attention_mask"]) # (batch, max_length=180, dim=128)
            # mtx_sim=torch.einsum("bqd, bpd -> bqp", Q, D)
            mtx_sim = Q @ D.permute(0, 2, 1)  # shape: (batch_size, query_maxlen, doc_maxlen) - NOT batching for now
            answers_list.append(get_snippet(mtx_sim.squeeze().cpu().numpy(), window_size, tokenized_doc["input_ids"].squeeze().cpu().numpy().tolist(), tokenizer))
    
    # export_predicted_snippets(eval_data, answers_list, window_size)
            
        detokenized_q = [tokenizer.decode([t], clean_up_tokenization_spaces=True) for t in tokenized_q["input_ids"].squeeze().cpu().numpy().tolist()]
        detokenized_doc = [tokenizer.decode([t], clean_up_tokenization_spaces=True) for t in tokenized_doc["input_ids"].squeeze().cpu().numpy().tolist()]
        df = pd.DataFrame(mtx_sim.squeeze().cpu().numpy(), index=detokenized_q, columns=detokenized_doc)
        df_lst.append(df)
        query_sim = torch.sum(mtx_sim, axis=1, keepdims=False) # shape: (batch, doc_maxlen)
        top_doc_indices = torch.argsort(query_sim, axis=-1, descending=True)[:, :top_k_docs] # shape: (batch, top_k_docs)
        top_doc_tokens = torch.gather(tokenized_doc["input_ids"], dim=1, index=top_doc_indices).squeeze().cpu().numpy().tolist()
        top_doc_strings = [tokenizer.decode([t], skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in top_doc_tokens]
        if "" in top_doc_strings:
            top_doc_strings.remove("")
        top_doc_strings = list(set(top_doc_strings)) # deduplicate
        ans_spans = []
        
        print(blue(f"\nQuestion: {line['question']}"))
        print(red(f"Answers: {line['answers']}"))
        for s in top_doc_strings:
            try:
                answer_start = line["context"].lower().index(s.lower())
            except ValueError:
                print(red(f"\n{s} not found in context"))
                continue
            ans_spans.append((answer_start, answer_start + len(s)))
        # sort spans
        ans_spans.sort(key=lambda x: x[0])
        start_idx = 0
        print(blue("Context: "), end="")
        for span in ans_spans:
            print(line["context"][start_idx:span[0]], end="")
            print(green(line["context"][span[0]:span[1]]), end="")
            start_idx = span[1]
        print(line["context"][start_idx:])
        
    with pd.ExcelWriter("cossim_colbert_sample1.xlsx") as writer:
        for i, df in enumerate(df_lst):
            df.to_excel(writer, sheet_name=f"pair_{i}")