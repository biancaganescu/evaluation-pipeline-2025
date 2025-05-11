from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from evaluation_pipeline.reading.evaluation_functions import get_p2_mntp, get_p2, get_p2_mlm
from tqdm import tqdm
import pandas as pd
import argparse
import pathlib
import statsmodels.formula.api as smf
from functools import partial
import math
import json
import torch
from tokenizers.processors import TemplateProcessing
import sys
sys.path.append('/home/bmg44/DualStreamTransformer')
from model import DualStreamTransformer

def parse_args():
    parser = argparse.ArgumentParser()

    # Required Parameters
    parser.add_argument("--output_dir", default="results", type=pathlib.Path, help="The output directory where the results will be written.")
    parser.add_argument("--data_path", default="reading/data/reading_data.csv", type=pathlib.Path, help="Path to file containing the lambada dataset, we expect it to be in a JSONL format.")
    parser.add_argument("--model_path_or_name", default="ltg/gpt-bert-babylm-small", type=pathlib.Path, help="The path/name to/of the huggingface folder/repository.")
    parser.add_argument("--backend", default="causal", type=str, help="The evaluation backend strategy.", choices=["mlm", "mntp", "causal", "dst"])
    parser.add_argument("--number_of_mask_tokens_to_append", default=3, type=int, help="When using either mlm or mntp, the number of mask tokens to append to approximate causal generation.")
    parser.add_argument("--revision_name", default=None, type=str, help="Name of the checkpoint/version of the model to test. (If None, the main will be used)")

    args = parser.parse_args()

    args.output_dir /= args.model_path_or_name.stem
    if args.revision_name is None:
        args.output_dir /= "main"
    else:
        args.output_dir /= args.revision_name
    args.output_dir /= "zero_shot"
    args.output_dir /= args.backend
    args.output_dir /= "reading"
    if args.backend in ["mlm", "mntp"]:
        args.output_dir /= f"{args.number_of_mask_tokens_to_append}_mask_tokens"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    return args


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.data_path, dtype={'item': str})
    df["item"] = df["item"].fillna("None")

    if args.backend == "causal":
        model = AutoModelForCausalLM.from_pretrained(args.model_path_or_name, trust_remote_code=True, revision=args.revision_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name, trust_remote_code=True, revision=args.revision_name)

    elif args.backend == "mlm" or args.backend == "mntp":
        model = AutoModelForMaskedLM.from_pretrained(args.model_path_or_name, trust_remote_code=True, revision=args.revision_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name, trust_remote_code=True, revision=args.revision_name)

    elif args.backend == "dst":
        checkpoint = torch.load(args.model_path_or_name)
        model_args = checkpoint.get("model_args", {})
        if not model_args:
                    raise ValueError("Checkpoint does not contain model_args")
        model = DualStreamTransformer(**model_args)
        model.load_state_dict(checkpoint["model_state_dict"])
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.add_special_tokens({'pad_token': '[PAD]', 'eos_token': '[EOS]', 'bos_token': '[BOS]'})
        tokenizer._tokenizer.post_processor = TemplateProcessing(
        tokenizer.bos_token + " $A " + tokenizer.eos_token,
            special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id), (tokenizer.bos_token, tokenizer.bos_token_id)],
            )



    model.eval()

    if args.backend == "causal" or args.backend == "dst":
        p2_function = get_p2
    elif args.backend == "mlm":
        p2_function = partial(get_p2_mlm, num_mask_tokens=args.number_of_mask_tokens_to_append)
    else:
        p2_function = partial(get_p2_mntp, num_mask_tokens=args.number_of_mask_tokens_to_append)

    out = []
    prev_p2 = []
    for index, row in tqdm(df.iterrows(), total=len(df)):

        p, _ = p2_function(row["item"], row["word"], model, tokenizer, args)
        out.append(p)
        if isinstance(row["prev_item"], str):
            try:
                prev_p, _ = p2_function(row["prev_item"], row["prev_word"], model, tokenizer, args)
            except Exception:
                print("exception reached")
                print(row)
                exit()
            prev_p2.append(-math.log(prev_p))
        else:
            prev_p2.append(float("NaN"))

    p2 = [-math.log(p) for p in out]

    df["pred"] = p2
    df["prev_pred"] = prev_p2

    pred_file = args.output_dir / "prediction.jsonl"
    with pred_file.open("w") as fj:
        for index, row in df.iterrows():
            print(json.dumps({"Index": index, "Sentence": row["item"], "Word": row["word"], "Logprob": row["pred"], "Prev_Logprob": row["prev_pred"]}), file=fj)

    variables = ['RTfirstfix', 'RTfirstpass', 'RTgopast', 'RTrightbound', 'self_paced_reading_time',  'ELAN', 'LAN', 'N400', 'P600', 'EPNP', 'PNP']

    correlations = df[["pred"] + variables].corr()["pred"]

    corr_file = args.output_dir / "correlations.txt"
    with corr_file.open("w") as f:
        for index, values in correlations.items():
            if index != "pred":
                print(f"{index}\t{values:.4f}", file=f)

    results = []
    report_values = []

    for dv in variables:
        # baseline model
        temp = df[[dv, "Subtlex_log10", "length", "context_length"]].dropna()
        # first fit baseline model without predictability
        OLS_baseline = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length', data=temp).fit()
        R2_baseline = float(OLS_baseline.rsquared)
        aic_baseline = float(OLS_baseline.aic)
        temp = df[["pred", dv, "Subtlex_log10", "length", "context_length"]].dropna()
        # experimental model with iv
        OLS_model = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length + pred', data=temp).fit()
        is_sig = float(OLS_model.tvalues["pred"])
        the_p = float(OLS_model.pvalues["pred"])
        the_B = float(OLS_model.params["pred"])
        R2_model = float(OLS_model.rsquared)
        aic_model = float(OLS_model.aic)
        results.append({
            "Predicted variable": dv,
            "Coefficient": the_B,
            "Number of standard deviations": is_sig,
            "P-value": the_p,
            "R2": R2_model,
            "Change in R2 from baseline": R2_model-R2_baseline,
            "AIC": aic_model,
            "Change in AIC from baseline": aic_model-aic_baseline,
        })
        if "RT" in dv:
            report_values.append(((R2_model-R2_baseline)/(1-R2_baseline)) * 100)

    predictability_file = args.output_dir / "predictive_power.jsonl"
    with predictability_file.open("w") as fj:
        for res in results:
            print(json.dumps(res), file=fj)

    predictability_file = args.output_dir / "report.txt"
    with predictability_file.open("w") as fj:
        print(f"EYE TRACKING SCORE: {sum(report_values) / len(report_values):.2f}", file=fj)

    results = []

    for dv in variables:
        # baseline model
        temp = df[[dv, "Subtlex_log10", "length", "context_length", "prev_length", "prev_pred"]].dropna()
        # first fit baseline model without predictability
        OLS_baseline = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + prev_length + prev_pred + Subtlex_log10:length + Subtlex_log10:context_length + Subtlex_log10:prev_length + Subtlex_log10:prev_pred + length:context_length + length:prev_length + length:prev_pred + context_length:prev_length + context_length:prev_pred + prev_length:prev_pred', data=temp).fit()
        R2_baseline = float(OLS_baseline.rsquared)
        aic_baseline = float(OLS_baseline.aic)
        temp = df[["pred", dv, "Subtlex_log10", "length", "context_length", "prev_length", "prev_pred"]].dropna()
        # experimental model with iv
        OLS_model = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + prev_length + prev_pred + Subtlex_log10:length + Subtlex_log10:context_length + Subtlex_log10:prev_length + Subtlex_log10:prev_pred + length:context_length + length:prev_length + length:prev_pred + context_length:prev_length + context_length:prev_pred + prev_length:prev_pred + pred', data=temp).fit()
        is_sig = float(OLS_model.tvalues["pred"])
        the_p = float(OLS_model.pvalues["pred"])
        the_B = float(OLS_model.params["pred"])
        R2_model = float(OLS_model.rsquared)
        aic_model = float(OLS_model.aic)
        results.append({
            "Predicted variable": dv,
            "Coefficient": the_B,
            "Number of standard deviations": is_sig,
            "P-value": the_p,
            "R2": R2_model,
            "Change in R2 from baseline": R2_model-R2_baseline,
            "AIC": aic_model,
            "Change in AIC from baseline": aic_model-aic_baseline,
        })

        if "self" in dv:
            report_values = ((R2_model-R2_baseline)/(1-R2_baseline)) * 100

    predictability_file = args.output_dir / "predictive_power_spillover.jsonl"
    with predictability_file.open("w") as fj:
        for res in results:
            print(json.dumps(res), file=fj)

    predictability_file = args.output_dir / "report.txt"
    with predictability_file.open("a") as fj:
        print(f"SELF-PACED READING SCORE: {report_values:.2f}", file=fj)
