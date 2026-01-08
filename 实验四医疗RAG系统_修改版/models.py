import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import CrossEncoder  # 新增导入交叉编码器

@st.cache_resource
def load_embedding_model(model_name):
    """Loads the sentence transformer model."""
    st.write(f"Loading embedding model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        st.success("Embedding model loaded.")
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

@st.cache_resource
def load_generation_model(model_name):
    """Loads the Hugging Face generative model and tokenizer."""
    st.write(f"Loading generation model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Adjust device_map and torch_dtype based on your hardware
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="cpu", # Force CPU to avoid CUDA issues
            torch_dtype=torch.float32
        )
        if tokenizer.pad_token is None:
             tokenizer.pad_token = tokenizer.eos_token
        st.success("Generation model and tokenizer loaded.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load generation model: {e}")
        return None, None


@st.cache_resource
def load_reranking_model(model_name):
    """Loads the cross-encoder reranking model."""
    if not model_name:
        return None
    st.write(f"Loading reranking model: {model_name}...")
    try:
        model = CrossEncoder(model_name)
        # # 设置填充标记（关键步骤！）
        # tokenizer = model.tokenizer
        # if tokenizer.pad_token is None:
        #     # 方法1：使用 eos_token 作为 pad_token
        #     tokenizer.pad_token = tokenizer.eos_token
        #     # 方法2：添加新的 [PAD] 标记
        #     # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #     # reranking_model.model.resize_token_embeddings(len(tokenizer))
        st.success("Reranking model loaded.")
        return model
    except Exception as e:
        st.error(f"Failed to load reranking model: {e}")
        return None

# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_HOME'] = '/root/autodl-tmp/SHU/RAG/hf_cache' 
# # load_embedding_model("thenlper/gte-large")
# # SentenceTransformer("thenlper/gte-large")
# SentenceTransformer("/root/autodl-tmp/SHU/RAG/hf_cache/hub/models--thenlper--gte-large")