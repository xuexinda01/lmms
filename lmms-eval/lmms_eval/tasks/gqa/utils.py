from datasets import load_dataset
import jieba

import sacrebleu
# from sacrebleu.metrics import BLEU
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import PIL
import logging
import sys
import jieba
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor
import requests
import base64
import re
import statistics

GQA_RAW_IMAGE_DATASET = None
GQA_ID2IMAGE = None


def gqa_doc_to_visual(doc):
    global GQA_RAW_IMAGE_DATASET
    global GQA_ID2IMAGE
    if GQA_RAW_IMAGE_DATASET is None:
        GQA_RAW_IMAGE_DATASET = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev", token=True)
        GQA_ID2IMAGE = {}
        for row in GQA_RAW_IMAGE_DATASET:
            GQA_ID2IMAGE[row["id"]] = row["image"].convert("RGB")
    image = GQA_ID2IMAGE[doc["imageId"]]
    return [image]


def gqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"

def parse_pred_ans_NY(pred_ans):
    pred_label = None
    if pred_ans in ["yes", "no"]:
        pred_label = pred_ans
    else:
        prefix_pred_ans = pred_ans[:4]

        if "yes" in prefix_pred_ans:
            pred_label = "yes"
        elif "no" in prefix_pred_ans:
            pred_label = "no"
        else:
            pred_label = "other"
    return pred_label


def parse_pred_ans_choice(pred_ans):
    return pred_ans.replace(" ", "")[0]

def metric_gpt4o(doc, pred_ans):
    API_KEY = "869d966045f44db6ae0b8de02f7bf776"

    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }
    
    eval_prompt = """
        [Instruction]\nPlease act as an impartial judge and evaluate the quality 
        of the response provided by an AI assistant to
        the user question displayed below. Your evaluation should 
        consider correctness and helpfulness. You will be given
        a reference answer and the assistant’s answer. Begin 
        your evaluation by comparing the assistant’s answer with the
        reference answer. Identify and correct any mistakes. The 
        assistant has access to an image alongwith questions but
        you will not be given images. Therefore, please consider only 
        how the answer is close to the reference answer. If
        the assistant’s answer is not exactly same as or similar to 
        the answer, then he must be wrong. Be as objective as
        possible. Discourage uninformative answers. Also, 
        equally treat short and long answers and focus on the correctness
        of answers. After providing your explanation, you 
        must rate the response with either 0, 0.5 or 1 by strictly following
        this format: “[[rating]]”, for example: “Rating: [[0.5]]”.
        \n\n[Question]\n{question}\n\n[The Start of Reference
        Answer]\n{refanswer}\n[The End of Reference Answer]
        \n\n[The Start of Assistant’s Answer]\n{answer}\n[The
        End of Assistant’s Answer]
    """
    eval_texts = eval_prompt.format(question=doc["question"].strip(), refanswer=doc["answer"].strip(), answer=pred_ans)
    payload = {
    "messages": [
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are an AI assistant that helps people find information."
            }
        ]
        },
        {
            "role": "user", 
            "content": eval_texts
        }
    ],
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 800
    }

    ENDPOINT = "https://baai-emllm-eastus2.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"

    # Send request
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")
    response = response.json()
    content = response['choices'][0]['message']['content']
    pattern = r"Rating:\s*\[\[(\d+(\.\d+)?)\]\]"

    match = re.search(pattern, content)

    if match:
        rating_value = float(match.group(1))  # 提取第一个捕获组（数值部分）
    else:
        rating_value = -1.0

    return rating_value, content
    
def gqa_process_results(doc, results):
    pred = results[0].lower().strip()
    gt_ans = doc["answer"].lower().strip()
    score = 1.0 if pred == gt_ans else 0.0
    return {
        "pope_accuracy": {"score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_precision": { "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_recall": { "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_f1_score": {"score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_yes_ratio": { "score": score, "prediction": pred, "ground_truth": gt_ans},
    }





def gqa_aggregate_accuracy(results):
    total_score = 0
    for result in results:
        total_score += result["score"]
    avg_score = total_score / len(results)
    return avg_score


def gqa_aggregate_precision(results):
    true_positives = 0
    false_positives = 0
    for result in results:
        pred = result["prediction"]
        gt = result["ground_truth"]
        if gt == "yes" and pred == "yes":
            true_positives += 1
        elif gt == "no" and pred == "yes":
            false_positives += 1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    return precision


def gqa_aggregate_recall(results):
    true_positives = 0
    false_negatives = 0
    for result in results:
        pred = result["prediction"]
        gt = result["ground_truth"]
        if gt == "yes" and pred == "yes":
            true_positives += 1
        elif gt == "yes" and pred == "no":
            false_negatives += 1
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall


def gqa_aggregate_f1_score(results):
    precision = pope_aggregate_precision(results)
    recall = pope_aggregate_recall(results)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def gqa_aggregate_yes_ratio(results):
    yes_count = 0
    no_count = 0
    for result in results:
        gt = result["ground_truth"]
        if gt == "yes":
            yes_count += 1
        elif gt == "no":
            no_count += 1
    yes_ratio = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
    return yes_ratio
