# Updated: Modified to ask users for their Hugging Face API token

import streamlit as st
import time
import numpy as np
from huggingface_hub import InferenceClient
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F
from typing import Dict
import plotly.graph_objects as go

# --------- Config ---------
st.set_page_config(page_title="LLM Evolution Visualizer", layout="wide")

# --------- InferenceClient for Mistral (chat completion) ---------
def call_mistral_api(prompt, hf_token, max_new_tokens=20, temperature=0.7):
    try:
        client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=hf_token)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Mistral API Error: {str(e)}")

# --------- Token Cleanup ---------
def clean_token(token: str, model_type: str) -> str:
    if model_type in ["gpt2", "distilgpt2"] and token.startswith("Ġ"):
        return token[1:]
    elif model_type == "mistral7b" and token.startswith("▁"):
        return token[1:]
    return token

# --------- Visualization ---------
def create_probability_chart(token_probs: Dict[str, float], title: str = "Next Token Probabilities"):
    sorted_items = sorted(token_probs.items(), key=lambda x: x[1], reverse=True)
    tokens, probs = zip(*sorted_items)
    fig = go.Figure(data=[
        go.Bar(
            y=tokens,
            x=probs,
            orientation='h',
            marker_color='lightblue',
            text=[f'{p:.3f}' for p in probs],
            textposition='auto'
        )
    ])
    fig.update_layout(title=title, xaxis_title="Probability", yaxis_title="Tokens", height=400)
    return fig

def display_text_with_highlight(text: str, new_token: str = ""):
    if new_token:
        html = f"""
        <div style='font-size:20px;'>
            {text}<span style='background-color: #ffeaa7; font-weight: bold;'>{new_token}</span>
        </div>
        """
    else:
        html = f"<div style='font-size:20px;'>{text}</div>"
    return html

# --------- Utility ---------
def apply_repetition_penalty(logits, input_ids, penalty=1.0):
    if penalty == 1.0:
        return logits
    unique_tokens = torch.unique(input_ids)
    for token in unique_tokens:
        if logits[token] > 0:
            logits[token] /= penalty
        else:
            logits[token] *= penalty
    return logits

def top_k_sampling(logits, k=10, temperature=1.0):
    logits = logits / temperature
    top_k_values, top_k_indices = torch.topk(logits, k)
    mask = torch.full_like(logits, float('-inf'))
    mask[top_k_indices] = top_k_values
    probs = F.softmax(mask, dim=-1)
    sampled_token = torch.multinomial(probs, 1)
    return sampled_token, probs

# --------- App ---------
st.title("🧠 Transformer Text Completion Visualizer")

# API Token Input Section
with st.sidebar:
    st.header("🔑 API Configuration")
    hf_token = st.text_input(
        "Hugging Face API Token:",
        type="password",
        help="Enter your Hugging Face API token. Get one at https://huggingface.co/settings/tokens",
        placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxx"
    )
    
    if not hf_token:
        st.warning("⚠️ Please enter your Hugging Face API token to use Mistral-7B model")
        st.info("💡 You can get a free token at https://huggingface.co/settings/tokens")
    else:
        st.success("✅ API token configured")

col1, col2 = st.columns([1, 2])

SAMPLE_SNIPPETS = {
    "Classic Opening": "It was the best of times, it was the worst of",
    "Sci-Fi": "The spaceship landed on Mars and the crew discovered",
    "Mystery": "The detective examined the crime scene and noticed",
    "Fantasy": "The wizard cast a powerful spell and suddenly",
    "Philosophy": "The nature of consciousness remains one of the most",
    "Technology": "Artificial intelligence will fundamentally change how we",
    "Custom": ""
}

with col1:
    st.subheader("⚙️ Configuration")
    models = {
        "DistilGPT-2 (Fast)": "distilgpt2",
        "GPT-2 (Full)": "gpt2",
        "Mistral-7B-Instruct (API)": "mistral7b"
    }
    selected_model = st.selectbox("🤖 Choose Model:", list(models.keys()))
    
    # Show warning if Mistral is selected but no API token
    if models[selected_model] == "mistral7b" and not hf_token:
        st.error("❌ Mistral-7B requires a Hugging Face API token. Please enter it in the sidebar.")
    
    snippet_choice = st.selectbox("📝 Choose Text Snippet:", list(SAMPLE_SNIPPETS.keys()))
    if snippet_choice == "Custom":
        input_text = st.text_area("Enter your text:", placeholder="Type your text here...")
    else:
        input_text = SAMPLE_SNIPPETS[snippet_choice]
        st.text_area("Selected text:", value=input_text, disabled=True)
    
    max_tokens = st.slider("🔢 Tokens to generate:", 3, 100, 10)
    top_k = st.slider("📊 Top-k sampling:", 3, 50, 10)
    temperature = st.slider("🌡️ Temperature:", 0.1, 2.0, 0.8, 0.1)
    repetition_penalty = st.slider("🔄 Repetition penalty:", 1.0, 2.0, 1.1, 0.1)
    animation_speed = st.slider("⏱️ Animation speed (seconds):", 0.3, 3.0, 1.0, 0.1)

with col2:
    st.subheader("🎬 Live Generation")
    text_display = st.empty()
    prob_display = st.empty()
    progress_bar = st.empty()

# Validate inputs before allowing generation
can_generate = True
if not input_text.strip():
    can_generate = False
    st.error("❌ Please enter some text to complete")
elif models[selected_model] == "mistral7b" and not hf_token:
    can_generate = False
    st.error("❌ Please enter your Hugging Face API token to use Mistral-7B")

if st.button("🚀 Start Text Completion", type="primary", disabled=not can_generate):
    current_text = input_text
    text_display.markdown(display_text_with_highlight(current_text), unsafe_allow_html=True)

    if models[selected_model] == "mistral7b":
        try:
            with st.spinner("🔄 Generating text with Mistral-7B..."):
                result = call_mistral_api(
                    prompt=input_text,
                    hf_token=hf_token,
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
            
            tokens = result.split()
            for i, word in enumerate(tokens):
                progress_bar.progress((i + 1) / len(tokens))
                current_text += " " + word
                text_display.markdown(display_text_with_highlight(current_text), unsafe_allow_html=True)
                time.sleep(animation_speed)
            prob_display.info("ℹ️ Token probability visualization is not available for Mistral-7B (API mode).")
            st.success("✅ Completed using Mistral API")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("💡 Make sure your API token is valid and has sufficient quota")

    else:
        try:
            with st.spinner(f"🔄 Loading {selected_model}..."):
                if models[selected_model] == "distilgpt2":
                    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
                    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
                    model_type = "distilgpt2"
                else:
                    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                    model = GPT2LMHeadModel.from_pretrained("gpt2")
                    model_type = "gpt2"

            tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
            input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False)
            input_ids = input_ids.to(next(model.parameters()).device)
            progress = progress_bar.progress(0)

            for step in range(max_tokens):
                progress.progress((step + 1) / max_tokens)
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits[0, -1, :].clone()
                    logits = apply_repetition_penalty(logits, input_ids.flatten(), repetition_penalty)
                    sampled_token, sampling_probs = top_k_sampling(logits, top_k, temperature)
                    top_k_probs, top_k_indices = torch.topk(sampling_probs, min(top_k, len(sampling_probs)))
                    top_tokens = tokenizer.convert_ids_to_tokens(top_k_indices.tolist())
                    top_values = top_k_probs.cpu().tolist()
                    cleaned_tokens = [clean_token(token, model_type) for token in top_tokens]
                    prob_dict = {token: prob for token, prob in zip(cleaned_tokens, top_values)}
                    fig = create_probability_chart(prob_dict, f"Step {step + 1}: Top-k Sampling (k={top_k})")
                    prob_display.plotly_chart(fig, use_container_width=True)
                    next_token = tokenizer.decode(sampled_token, skip_special_tokens=True)
                    current_text += next_token
                    text_display.markdown(display_text_with_highlight(current_text, next_token), unsafe_allow_html=True)
                    time.sleep(animation_speed)
                    sampled_token_expanded = sampled_token.unsqueeze(0).to(input_ids.device)
                    input_ids = torch.cat([input_ids, sampled_token_expanded], dim=1)
                    if sampled_token.item() == tokenizer.eos_token_id or input_ids.shape[1] > 1024:
                        break
            progress.empty()
            st.success(f"✅ Generated {step + 1} tokens successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")

# Instructions section
with st.expander("📖 How to use"):
    st.markdown("""
    **Getting Started:**
    1. **Get a Hugging Face API token** (free): Visit https://huggingface.co/settings/tokens
    2. **Enter your token** in the sidebar (it's kept secure with password masking)
    3. **Choose a model**: Local models (GPT-2, DistilGPT-2) or API-based (Mistral-7B)
    4. **Select or enter text** to complete
    5. **Adjust parameters** like temperature and top-k sampling
    6. **Click "Start Text Completion"** to see the magic happen!
    
    **Models:**
    - **DistilGPT-2**: Fast, lightweight, runs locally
    - **GPT-2**: Full model, better quality, runs locally  
    - **Mistral-7B**: Most advanced, requires API token, runs on Hugging Face servers
    
    **Parameters:**
    - **Temperature**: Higher = more creative, Lower = more predictable
    - **Top-k**: Number of most likely tokens to sample from
    - **Repetition penalty**: Reduces repetitive text
    """)