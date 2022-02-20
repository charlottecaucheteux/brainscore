REGEX_GENTLE_TOKENIZER = r"(\w|\’\w|\'\w)+"

TRANSFORMER_NAMES = {
    "gpt2": "gpt2",
    "bert": "bert-base-cased",
    "bert-base-cased": "bert-base-cased",
    "bert-large": "bert-large-cased",
}

for model in [
    "albert-base-v1",
    "roberta-base-openai-detector",
    "distilbert-base-uncased",
    "distilgpt2",
    "microsoft/layoutlm-base-uncased",
    "allenai/longformer-base-4096",
    "bert-base-uncased",
    "squeezebert/squeezebert-mnli",
    "xlnet-base-cased",
    "roberta-base",
    "transfo-xl-wt103",
    "allenai/longformer-base-4096",
]:
    label = model.split("/")[-1]
    TRANSFORMER_NAMES[label] = label
