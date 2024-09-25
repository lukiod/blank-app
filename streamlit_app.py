import streamlit as st
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from byaldi import RAGMultiModalModel
from qwen_vl_utils import process_vision_info

# Model and processor names
RAG_MODEL = "vidore/colpali"
QWN_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

@st.cache_resource
def load_models():
    RAG = RAGMultiModalModel.from_pretrained(RAG_MODEL)
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        QWN_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    ).eval()
    
    processor = AutoProcessor.from_pretrained(QWN_MODEL, trust_remote_code=True)
    
    return RAG, model, processor

RAG, model, processor = load_models()

def document_rag(text_query, image):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": text_query},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

st.title("Document Processor")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
text_query = st.text_input("Enter your text query")

if uploaded_file is not None and text_query:
    image = Image.open(uploaded_file)
    
    if st.button("Process Document"):
        with st.spinner("Processing..."):
            result = document_rag(text_query, image)
        st.success("Processing complete!")
        st.write("Result:", result)