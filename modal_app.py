
import modal
import torch


app = modal.App(name="CascaAI")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    
    "Pillow==10.3.0",
    "transformers==4.41.2",
    "torch==2.2.0",
    "torchvision==0.16.0",
    "torchaudio==2.2.0",
    "fastapi[standard]==0.115.4",
    "uvicorn[standard]==0.29.0",
    "pdfplumber==0.11.0",
    "langchain==0.2.0",
    "langchain_community==0.2.0",
    "langchain_core==0.2.0",
    "langchain_mistralai==0.1.0",
    "langchain_text_splitters==0.2.0",
    "langchain_huggingface==0.2.0",
    "langchain_chroma==0.2.0",
    "langgraph==0.1.0",
    "modal==0.62.0"
)

@app.function(gpu="A100", image=image)
def run():
    print(torch.cuda.is_available())
    print("*****")



