"""
📚 Study Card Generator
Fine-tuned Llama 3.2 1B - KTH ID2223 Lab 2
Generates flashcards from topics or text for studying
"""

import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import json
import re
import os

# Model configuration
HF_REPO_ID = "dnagard/PEFT-optimization"
GGUF_FILENAME = "llama-3.2-1b-finetuned-q8_0_v2.gguf"

# Download and load model
print(f"Downloading model from {HF_REPO_ID}...")
model_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=GGUF_FILENAME,
)
print(f"Model downloaded to: {model_path}")

print("Loading model...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=2,
    n_gpu_layers=0,
    verbose=False
)
print("Model loaded!")


SYSTEM_PROMPT = """You are a helpful study assistant that creates flashcards for students. 
When asked to create study cards, generate clear question-answer pairs that help with memorization and understanding.
Format each card as:
Q: [question]
A: [answer]

Keep answers concise but informative."""


def format_prompt(message: str) -> str:
    """Format prompt for Llama 3.2 Instruct."""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def parse_cards(text: str) -> list:
    """Parse Q/A pairs from model output."""
    cards = []
    
    # Split by Q: pattern
    parts = re.split(r'\n(?=Q:)', text)
    
    for part in parts:
        part = part.strip()
        if not part.startswith('Q:'):
            continue
            
        # Extract question and answer
        q_match = re.search(r'Q:\s*(.+?)(?=\nA:|\Z)', part, re.DOTALL)
        a_match = re.search(r'A:\s*(.+?)(?=\nQ:|\Z)', part, re.DOTALL)
        
        if q_match and a_match:
            question = q_match.group(1).strip()
            answer = a_match.group(1).strip()
            if question and answer:
                cards.append({"question": question, "answer": answer})
    
    return cards


def generate_cards(topic: str, source_text: str, num_cards: int, difficulty: str) -> tuple:
    """Generate study cards from topic or source text."""
    
    if not topic and not source_text:
        return "⚠️ Please enter a topic or paste some text to study.", ""
    
    # Build the prompt
    if source_text:
        user_prompt = f"""Create {num_cards} study flashcards based on this text:

"{source_text}"

Difficulty level: {difficulty}
Generate {num_cards} question-answer pairs covering the key concepts."""
    else:
        user_prompt = f"""Create {num_cards} study flashcards about: {topic}

Difficulty level: {difficulty}
Generate {num_cards} question-answer pairs covering important concepts about this topic."""

    prompt = format_prompt(user_prompt)
    
    # Generate
    output = llm(
        prompt,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        echo=False
    )
    
    raw_output = output["choices"][0]["text"].strip()
    
    # Parse into cards
    cards = parse_cards(raw_output)
    
    if not cards:
        # If parsing failed, show raw output
        return f"Generated content:\n\n{raw_output}", raw_output
    
    # Format cards as HTML
    html_cards = ""
    for i, card in enumerate(cards, 1):
        html_cards += f"""
        <div style="border: 2px solid #4a5568; border-radius: 12px; padding: 16px; margin: 12px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <div style="color: #fff; font-weight: bold; margin-bottom: 8px;">Card {i}</div>
            <div style="background: white; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                <div style="color: #4a5568; font-size: 12px; text-transform: uppercase; margin-bottom: 4px;">Question</div>
                <div style="color: #1a202c; font-size: 16px;">{card['question']}</div>
            </div>
            <div style="background: #f7fafc; border-radius: 8px; padding: 12px;">
                <div style="color: #4a5568; font-size: 12px; text-transform: uppercase; margin-bottom: 4px;">Answer</div>
                <div style="color: #2d3748; font-size: 15px;">{card['answer']}</div>
            </div>
        </div>
        """
    
    return html_cards, raw_output


# Build Gradio UI
with gr.Blocks(
    title="📚 Study Card Generator",
    theme=gr.themes.Soft(primary_hue="purple"),
    css=".gradio-container {max-width: 900px !important}"
) as demo:
    
    gr.Markdown(
        """
        # 📚 Study Card Generator
        
        Generate flashcards to help you study any topic! Powered by a fine-tuned Llama 3.2 1B model.
        
        **KTH ID2223 - Scalable Machine Learning** | Fine-tuned with LoRA on FineTome-100k
        
        ---
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            topic_input = gr.Textbox(
                label="📌 Topic",
                placeholder="e.g., Photosynthesis, World War 2, Python loops...",
                lines=1
            )
            
            source_text = gr.Textbox(
                label="📄 Or paste text to study (optional)",
                placeholder="Paste lecture notes, textbook excerpt, or any text you want to create cards from...",
                lines=5
            )
            
            with gr.Row():
                num_cards = gr.Slider(
                    minimum=2,
                    maximum=6,
                    value=3,
                    step=1,
                    label="Number of cards"
                )
                difficulty = gr.Dropdown(
                    choices=["Beginner", "Intermediate", "Advanced"],
                    value="Intermediate",
                    label="Difficulty"
                )
            
            generate_btn = gr.Button("🎴 Generate Study Cards", variant="primary", size="lg")
            
            gr.Markdown("---")
            gr.Markdown("**⏱️ Note:** Running on CPU - generation takes 15-45 seconds")
    
    with gr.Row():
        with gr.Column():
            cards_output = gr.HTML(label="Your Study Cards")
    
    with gr.Accordion("🔍 Raw Model Output", open=False):
        raw_output = gr.Textbox(label="Raw output", lines=10, interactive=False)
    
    # Examples
    gr.Examples(
        examples=[
            ["Machine Learning basics", "", 3, "Beginner"],
            ["The French Revolution", "", 4, "Intermediate"],
            ["Quantum Computing", "", 3, "Advanced"],
            ["", "Mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells. They are often referred to as the powerhouses of the cell because they generate most of the cell's supply of ATP, which is used as a source of chemical energy.", 3, "Intermediate"],
        ],
        inputs=[topic_input, source_text, num_cards, difficulty],
        label="Try these examples:"
    )
    
    # Event handler
    generate_btn.click(
        fn=generate_cards,
        inputs=[topic_input, source_text, num_cards, difficulty],
        outputs=[cards_output, raw_output]
    )

if __name__ == "__main__":
    demo.launch()