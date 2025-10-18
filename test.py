from datasets import load_dataset
import torch
from imported_code.modeling_sfr import SFR
from transformers import AutoTokenizer

d = load_dataset("brimmann2/squad-v2-sampled")
ds = d["train"].select(range(3))

device = "cuda" if torch.cuda.is_available() else "cpu"

retriever_name = "Salesforce/SFR-Embedding-Mistral"
retriever = SFR.from_pretrained(retriever_name,torch_dtype = torch.bfloat16).eval().to(device)

retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name)

def embed_batch(batch):
    docs = batch["context"]  # list of strings
    inputs = retriever_tokenizer(
        docs,
        max_length=180,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        embeds = retriever.get_doc_embedding(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        # in case model returns a tuple
        if isinstance(embeds, (tuple, list)):
            embeds = embeds[0]

    # store as list-of-floats per row (HF Datasets-friendly)
    return {"embeddings": embeds.detach().cpu().numpy().astype("float32").tolist()}


ds = ds.map(embed_batch, batched=True, batch_size=3)
print(ds)
