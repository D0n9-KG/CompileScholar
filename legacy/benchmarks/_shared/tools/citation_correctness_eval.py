import argparse
import collections
import json
import re
import string
import torch
import copy
import torch
import json
import re
import string

from nltk import sent_tokenize
import numpy as np
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm
import re
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


from run_utils import load_jsonlines

GOOGLE_AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture"
OSU_AUTOAIS_MODEL="osunlp/attrscore-flan-t5-xl"
input_prompt = "As an Attribution Validator, your task is to verify whether a given reference can support the given claim. A claim can be either a plain sentence or a question followed by its answer. Specifically, your response should clearly indicate the relationship: Attributable, Contradictory or Extrapolatory. A contradictory error occurs when you can infer that the answer contradicts the fact presented in the context, while an extrapolatory error means that you cannot infer the correctness of the answer based on the information provided in the context. \n\nClaim: {claim}\n Reference: {output}"

global autoais_model, autoais_tokenizer
global claim_autoais_model, claim_autoais_tokenizer
autoais_model, autoais_tokenizer = None, None
claim_autoais_model, claim_autoais_tokenizer = None, None

def extract_citations(text):
    # Regular expression to match [number] or [number_1, number_2, number_3]
    citation_pattern = r'\[(\d+(?:,\s*\d+)*)\]'
    # Find all matches in the text
    matches = re.findall(citation_pattern, text)
    # Extract individual numbers and convert them to integers
    citations = []
    for match in matches:
        # Split by commas, strip any extra whitespace, and convert to integers
        citations.extend([int(num.strip()) for num in match.split(',')])
    citations = ["[{}]".format(i) for i in citations]
    return citations

def remove_citations(text):
    # Regular expression to match [number] or [number_1, number_2, number_3]
    citation_pattern = r'\[\d+(?:,\s*\d+)*\]'
    # Remove all citations from the text
    cleaned_text = re.sub(citation_pattern, '', text)
    # Optionally, remove extra spaces that might result from removing citations
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
    cleaned_text = cleaned_text.replace(" .", ".")
    cleaned_text = cleaned_text.replace(" ,", ",")
    return cleaned_text

def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(a_gold, a_pred):
    """Compute F1 score between two strings."""

    def _get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    gold_toks = _get_tokens(a_gold)
    pred_toks = _get_tokens(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_exact(a_gold, a_pred):
    """Check whether two strings are equal up to normalization."""

    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False


def compute_rouge(data):
    """Main function for rouge scoring.
    If two references are provided,
    the best score is chosen for each instance.
    Args:
        data: requires field `output` and `answer` (or `annotations` for ASQA)
        metrics: list of evaluation metrics
    Returns:
        dictionary representation of rouge scores
    """
    def _rouge_calculation(hypotheses,
                        references1,
                        references2=[],
                        metrics=['rougeL', 'rouge1', 'rouge2']):

        if references2 == []:
            references2 = references1

        scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()
        all_scores = []

        for i in range(len(hypotheses)):
            scores1 = scorer.score(references1[i], hypotheses[i])
            scores2 = scorer.score(references2[i], hypotheses[i])
            if scores1['rougeL'].fmeasure > scores2['rougeL'].fmeasure:
                aggregator.add_scores(scores1)
                all_scores.append(scores1['rougeL'].fmeasure)
            else:
                aggregator.add_scores(scores2)
                all_scores.append(scores2['rougeL'].fmeasure)

        scores = {m: [] for m in metrics}

        for m in metrics:
            fmeasure = aggregator.aggregate()[m].mid.fmeasure
            scores[m].append(fmeasure)

        for m in scores:
            scores[m] = 100 * sum(scores[m]) / len(scores[m])

        return scores, all_scores

    hypotheses = {}
    references1 = {}
    references2 = {}

    for idx, item in enumerate(data):
        hypotheses[idx] = item["output"]
        if "annotations" in item and item['annotations'] is not None: # For ASQA
            references1[idx] = item["annotations"][0]["long_answer"]
            references2[idx] = item["annotations"][1]["long_answer"]
        else:
            references1[idx] = item["answer"]
            references2[idx] = item["answer"]

    h, r1, r2 = [], [], []

    for key in references1:
        h.append(hypotheses[key])
        r1.append(references1[key])

        if references2 is not None:
            r2.append(references2[key])

    h = ['\n'.join(sent_tokenize(text.lower())) for text in h]
    r1 = ['\n'.join(sent_tokenize(text.lower())) for text in r1]
    r2 = ['\n'.join(sent_tokenize(text.lower())) for text in r2]
    scores, all_scores = _rouge_calculation(h, r1, r2)

    print(scores['rougeL'])
    return scores, all_scores


def compute_str_em(data):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        return 0, 0

    acc = []
    hit = []

    for item in data:
        loc_acc = []
        for qa_pair in item['qa_pairs']:
            loc_acc.append(exact_presence(qa_pair['short_answers'], item["output"]))
        acc.append(np.mean(loc_acc))
        hit.append( int(np.mean(loc_acc) == 1) )

    return 100 * np.mean(acc), 100 * np.mean(hit)


def compute_len(data):
    """Compute average length of predictions."""

    res, cntr = 0, 0
    for item in data:
        res += len(item["output"].split())
        cntr += 1
    return res / cntr

def compute_single_qa(data):
    """Compute QA-based accuracy.
    Args:
        data: requires filed `qa_pairs/short_answers` and `output`
    Returns:
        QA metrics (QA-EM, QA-F1, QA-Hit)
    """

    # Get prediction
    em, f1 = [], []
    for item in tqdm(data):
        answers = [item["answer"]]
        prediction = item["output"]
        print(answers)
        print(prediction)
        em.append([compute_exact(a, prediction) for a in answers])
        f1.append([compute_f1(a, prediction) for a in answers])
        print(em[-1])


    return {
        'QA-EM': 100 * np.mean(em),
        'QA-F1': 100 * np.mean(f1),
    }
    
def compute_match(data):
    """Compute QA-based accuracy.
    Args:
        data: requires filed `qa_pairs/short_answers` and `output`
    Returns:
        QA metrics (QA-EM, QA-F1, QA-Hit)
    """

    # Get prediction
    match = []
    for item in tqdm(data):
        answers = remove_citations(item["answer"]).lower()
        prediction = remove_citations(item["output"]).lower()
        if answers in prediction:
            match.append(1.0)
        else:
            match.append(0.0)


    return {
        'match': 100 * np.mean(match),
    }
    

def _run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    # global autoais_model, autoais_tokenizer
    # input_text = "premise: {} hypothesis: {}".format(passage, claim)
    # input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    # with torch.inference_mode():
    #     outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    # result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # inference = 1 if result == "1" else 0
    global claim_autoais_model, claim_autoais_tokenizer
    input_text = input_prompt.format_map({"output": passage, "claim": claim})
    input_ids = claim_autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(claim_autoais_model.device)
    with torch.inference_mode():
        outputs = claim_autoais_model.generate(input_ids, max_new_tokens=10)
    result = claim_autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if result == "Attributable":
        inference = 1.0
    else:
        inference = 0.0

    return inference


def compute_autoais(data,
                    at_most_citations=None):
    """
    Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
    """

    global claim_autoais_model, claim_autoais_tokenizer

    if claim_autoais_model is None:
        logger.info("Loading Claims AutoAIS model...")
        claim_autoais_model = AutoModelForSeq2SeqLM.from_pretrained(OSU_AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
        claim_autoais_tokenizer = AutoTokenizer.from_pretrained(OSU_AUTOAIS_MODEL, use_fast=False)
        
    logger.info(f"Running AutoAIS...")

    def _format_document(doc):
        """Format document for AutoAIS."""

        if "sent" in doc:
            # QA-extracted docs
            return "Title: %s\n%s" % (doc['title'], doc['sent'])
        else:
            if "title" in doc:
                return "Title: %s\n%s" % (doc['title'], doc['text'])
            else:
                return doc["text"]

    ais_scores = []
    ais_scores_prec = []

    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0
    autoais_log = []
    cited_paper_total = []
    for item in tqdm(data):
        sents = sent_tokenize(item['output'])
        if len(sents) == 0:
            continue

        target_sents = [remove_citations(sent).strip() for sent in sents]
        

        cited_papers = set(extract_citations(item['output']))
        cited_paper_total.append(len(cited_papers))

        entail = 0
        entail_prec = 0
        total_citations = 0
        total_sents = 0
        previous_citations = None
        citations = item["ctxs"]
        for sent_id, sent in enumerate(sents):
            # add minimum length for citation
            if len(sent) < 50:
                continue
            total_sents += 1

            target_sent = target_sents[sent_id] # Citation removed and (if opted for) decontextualized
            joint_entail = -1 # Undecided

            # Find references
            # ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] # In text citation id starts from 1
            ref = [int(r[1:]) for r in re.findall(r"\[\d+", sent)]
            logger.info(f"For `{sent}`, find citations {ref}")
            if len(ref) == 0 and previous_citations is not None:
                ref = previous_citations
            
            if len(ref) == 0:
                # No citations
                joint_entail = 0
            elif any([ref_id >= len(citations) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
            else:
                previous_citations = ref
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                # print(item['docs'].keys())
                joint_passage = '\n'.join([_format_document(item['docs'][psgs_id]) for psgs_id in ref if psgs_id >=0])

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1:
                joint_entail = _run_nli_autoais(joint_passage, target_sent)
                autoais_log.append({
                    "question": item['question'],
                    "output": item['output'],
                    "claim": sent,
                    "passage": [joint_passage],
                    "model_type": "NLI",
                    "model_output": joint_entail,
                })

            entail += joint_entail
            if len(ref) > 1:
                sent_mcite += 1

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                sent_mcite_support += 1
                # Precision check: did the model cite any unnecessary documents?
                for psgs_id in ref:
                    # condition A
                    passage = _format_document(item['docs'][psgs_id])
                    nli_result = _run_nli_autoais(passage, target_sent)

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage = '\n'.join([_format_document(item['docs'][pid]) for pid in subset_exclude])
                        nli_result = _run_nli_autoais(passage, target_sent)
                        if nli_result: # psgs_id is not necessary
                            flag = 0
                            sent_mcite_overcite += 1
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail

        # sent_total += len(sents)
        # ais_scores.append(entail / len(sents))
        sent_total = total_sents
        if sent_total > 0:
            ais_scores.append(entail / sent_total ) 
        else:
            ais_scores.append(0 ) 

        ais_scores_prec.append(entail_prec / total_citations if total_citations > 0 else 0) # len(sents))

    if sent_mcite > 0 and sent_mcite_support > 0:
        print("Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite." % (
            100 * sent_mcite / sent_total,
            100 * sent_mcite_support / sent_mcite,
            100 * sent_mcite_overcite / sent_mcite_support
        ))

    return {
        "citation_rec": 100 * np.mean(ais_scores),
        "citation_rec_all": ais_scores,
        "citation_prec": 100 * np.mean(ais_scores_prec),
        "citation_prec_all": ais_scores_prec,
        "cited_paper_numbers": np.mean(cited_paper_total)
    }


def compute_autoais_short_form(data,
                    at_most_citations=None):
    """
    Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output
    """

    global claim_autoais_model, claim_autoais_tokenizer
    # if autoais_model is None:
    #     logger.info("Loading AutoAIS model...")
    #     autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
    #     autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    if claim_autoais_model is None:
        logger.info("Loading Claims AutoAIS model...")
        claim_autoais_model = AutoModelForSeq2SeqLM.from_pretrained(OSU_AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
        claim_autoais_tokenizer = AutoTokenizer.from_pretrained(OSU_AUTOAIS_MODEL, use_fast=False)
        
    logger.info(f"Running AutoAIS...")

    def _format_document(doc):
        """Format document for AutoAIS."""

        if "sent" in doc:
            # QA-extracted docs
            return "Title: %s\n%s" % (doc['title'], doc['sent'])
        else:
            if "title" in doc:
                return "Title: %s\n%s" % (doc['title'], doc['text'])
            else:
                return doc["text"]

    ais_scores = []
    ais_scores_prec = []

    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0
    autoais_log = []

    for item in tqdm(data):
        target_sents = [item["input"] + " " + remove_citations(item["output"]).strip()]
        sents = [item["input"] + " " + item["output"]]
        citations = item["ctxs"]
        total_sents = 0

        entail = 0
        entail_prec = 0
        total_citations = 0
        total_sents = 0
        for sent_id, sent in enumerate(sents):
            # add minimum length for citation
            total_sents += 1
            target_sent = target_sents[sent_id] # Citation removed and (if opted for) decontextualized
            joint_entail = -1 # Undecided

            # Find references
            # ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] # In text citation id starts from 1
            ref = [int(r[1:]) for r in re.findall(r"\[\d+", sent)]
            logger.info(f"For `{sent}`, find citations {ref}")
            
            if len(ref) == 0:
                # No citations
                joint_entail = 0
            elif any([ref_id >= len(citations) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
            else:
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = '\n'.join([_format_document(item['docs'][psgs_id]) for psgs_id in ref if psgs_id >=0])

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1:
                joint_entail = _run_nli_autoais(joint_passage, target_sent)
                autoais_log.append({
                    "question": item['question'],
                    "output": item['output'],
                    "claim": sent,
                    "passage": [joint_passage],
                    "model_type": "NLI",
                    "model_output": joint_entail,
                })

            entail += joint_entail
            if len(ref) > 1:
                sent_mcite += 1

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                sent_mcite_support += 1
                # Precision check: did the model cite any unnecessary documents?
                for psgs_id in ref:
                    # condition A
                    passage = _format_document(item['docs'][psgs_id])
                    nli_result = _run_nli_autoais(passage, target_sent)

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage = '\n'.join([_format_document(item['docs'][pid]) for pid in subset_exclude])
                        nli_result = _run_nli_autoais(passage, target_sent)
                        if nli_result: # psgs_id is not necessary
                            flag = 0
                            sent_mcite_overcite += 1
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail

        # sent_total += len(sents)
        # ais_scores.append(entail / len(sents))
        sent_total = total_sents
        ais_scores.append(entail / sent_total )

        ais_scores_prec.append(entail_prec / total_citations if total_citations > 0 else 0) # len(sents))

    if sent_mcite > 0 and sent_mcite_support > 0:
        print("Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite." % (
            100 * sent_mcite / sent_total,
            100 * sent_mcite_support / sent_mcite,
            100 * sent_mcite_overcite / sent_mcite_support
        ))

    return {
        "citation_rec": 100 * np.mean(ais_scores),
        "citation_rec_all": ais_scores,
        "citation_prec": 100 * np.mean(ais_scores_prec),
        "citation_prec_all": ais_scores_prec,
    }

def compute_citation_coverage(data):
    prec = []
    rec = []
    f1 = []

    num_preds = []
    for item in data:
        o = item['output']
        preds = list(set(extract_citations(item["output"])))
        if "id_mapping" in item:
            print(item["id_mapping"])
            preds = [item["id_mapping"][p.split("[")[1].split("]")[0]] for p in preds if p.split("[")[1].split("]")[0] in item["id_mapping"]]
        num_preds.append(len(preds))
        if "gold_ctxs" not in item:
            answers = list(set(extract_citations(item["answer"])))
        else:
            answers = list(set(item["gold_ctxs"]))
        print("answers: {0} preds: {1}".format(answers, preds))
        prec_i = len([p for p in preds if p in answers]) / len(preds) if len(preds) > 0 else 0
        prec.append(prec_i)
        rec_i = len([a for a in answers if a in preds]) / len(answers) if len(answers) > 0 else 0
        rec.append(rec_i)
        print(prec_i, rec_i)
        if (prec[-1] + rec[-1]) == 0:
            f1.append(0)
        else:
            f1.append(2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1]))
    return {
        "num_preds": np.mean(num_preds),
        "evidence_prec": 100 * np.mean(prec),
        "evidence_rec": 100 * np.mean(rec),
        "evidence_f1": 100 * np.mean(f1),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, required=True, help="Output file. Should have field `question`, `output`, (ROUGE) `answer`, \
                        (accuracy) `qa_pairs`, (AIS) `docs`", nargs="+")
    parser.add_argument("--no_rouge", action="store_true", help="Do not evaluate ROUGE score")
    parser.add_argument("--qa", action="store_true", help="Use the QA model")
    parser.add_argument("--single_qa", action="store_true", help="Use the QA model")
    
    parser.add_argument("--citations", action="store_true", help="Evaluation with citation")
    parser.add_argument("--at_most_citations", type=int, default=3, help="At most take this many documents (mostly for precision)")
    parser.add_argument("--claims_nli", action="store_true", help="Use claims for ELI5")
    parser.add_argument("--evidence", action="store_true", help="Compute evidence coverage.")
    parser.add_argument("--match", action="store_true", help="Compute answer matching")
    parser.add_argument("--use_input", action="store_true", help="Use input to compute the auto ais")
    parser.add_argument("--source_data", type=str, default=None)
    parser.add_argument("--max_limit", type=int, default=None)
    
    parser.add_argument("--citations_short", action="store_true", help="Evaluation with citation")

    args = parser.parse_args()
    
    if args.source_data is not None:
        source_data = load_jsonlines(args.source_data)
        input2claims = {item["input"]: item["claims"] for item in source_data}
            
    for file_name in args.f:
        if file_name.endswith(".json"):
            with open(file_name) as f:
                data_with_config = json.load(f)
            data = data_with_config['data'] if type(data_with_config) is dict else data_with_config
        else:
            data = load_jsonlines(file_name)   
        if args.max_limit is not None:
            print("original data num: {0}".format(len(data)))
            data = [item for item in data if len(item["output"].split()) < args.max_limit]
            print("filtered data num: {0}".format(len(data)))

        if(args.citations is True or args.citations_short is True ) and "docs" not in data[0]:
            for item in data:
                item["docs"] = item["ctxs"]
                if type(item["ctxs"]) is not list:
                    item["docs"] = [{"text": ctx_text[0], "title": ctx_text[1]} for ctx_text in list(item["ctxs"].values())]

        if "question" not in data[0]:
            for item in data:
                item["question"] = item["input"] if "input" in item else ""

        # Truncate by newline and remove on the fly search result
        logger.warning("We remove all the pre/appended space/newlines and we truncate the answer by the first newline.")
        logger.warning("We replace any on the fly search result to standard bracket citation format.")
        # for i in range(len(data)):
        #     data[i]['output'] = data[i]['output'].replace("<|im_end|>", "")
        #     data[i]["output"] = data[i]["output"].replace("Here is the revised answer:\n\n", "")
        #     data[i]['output'] = data[i]['output'].replace("[OUTLINE_START]", "")
        #     data[i]['output'] = data[i]['output'].replace("[OUTLINE_END]", "")
        #     data[i]['output'] = data[i]['output'].replace("[CITATION]", "")
        #     data[i]['output'] = data[i]['output'].replace("<cit.>", "")
        #     if "answer" in data[i]:
        #         data[i]['answer'] = data[i]['answer'].replace("[OUTLINE_START]", "")
        #         data[i]['answer'] = data[i]['answer'].replace("[OUTLINE_END]", "")
        #         data[i]['answer'] = data[i]['answer'].replace("[CITATION]", "")
        #         data[i]['answer'] = data[i]['answer'].replace("<cit.>", "")

        # Remove all citations for all non-AutoAIS evaluation
        normalized_data = copy.deepcopy(data)
        for i in range(len(normalized_data)):
            normalized_data[i]['output'] = remove_citations(normalized_data[i]['output'])

        result = {}
        all_scores = {}
        result['length'] = compute_len(normalized_data)
        if args.evidence:
            result["coverage"] = compute_citation_coverage(data)
        result['str_em'], result['str_hit'] = compute_str_em(normalized_data)
        
        if not args.no_rouge and "answer" in normalized_data:
            rouge_results, all_scores["rougeL"] = compute_rouge(normalized_data)
            result['rougeL'] = rouge_results["rougeL"]
            result['rouge1'] = rouge_results["rouge1"]
            result['rouge2'] = rouge_results["rouge2"]

        if args.single_qa:
            result.update(compute_single_qa(normalized_data))
    
        if args.match:
            result.update(compute_match(normalized_data))
            
        if args.citations:
            ais_results = compute_autoais(data,  at_most_citations=args.at_most_citations)
            result["citation_rec"] = ais_results["citation_rec"]
            result["citation_prec"] = ais_results["citation_prec"]
            all_scores["citation_rec_all"] = ais_results["citation_rec_all"]
            all_scores["citation_prec_all"] = ais_results["citation_prec_all"]
            result["cited_paper_numbers"] = ais_results["cited_paper_numbers"]

        if args.citations_short:
            ais_results = compute_autoais_short_form(data, at_most_citations=args.at_most_citations)
            result["citation_rec"] = ais_results["citation_rec"]
            result["citation_prec"] = ais_results["citation_prec"]
            all_scores["citation_rec_all"] = ais_results["citation_rec_all"]
            all_scores["citation_prec_all"] = ais_results["citation_prec_all"]

        print(result)
        with open(file_name + ".score_post_fix", "w") as f:
            json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()
