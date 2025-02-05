# api.py
import uvicorn
import logging
from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the remote inference from modal_app and the RAG inference from lang.
#from modal_app import remote_inference
from lang import rag_inference
from plumber import inference
from langchain_core.prompts import PromptTemplate


app = FastAPI(title="Bank Statement Analysis API")

# Enable CORS to allow requests from your Vercel Next.js front end.
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["https://cascadingfrontend.vercel.app/"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# @app.get("/inference")
# def inference_endpoint():
#     result = remote_inference()
#     return result

# @app.post("/rag")
# def rag_endpoint():

#     result = rag_inference(inference())
#     return result
    
class InferenceResponse(BaseModel):
    decision: str


@app.post("/analyze", response_model=InferenceResponse)
async def analyze_endpoint(pdf: UploadFile = File(...)):
    # Save the uploaded PDF file to a temporary location.
    temp_path = f"/tmp/{pdf.filename}"
    with open(temp_path, "wb") as f:
        f.write(await pdf.read())
    
    # Run the image-to-text inference (using your plumber logic) with the uploaded PDF.
    #local_result = inference(temp_path)
    local_result = "Based on the bank statement informaiton, does this person deserve a loan? Use the context from the documents."
    
    # Pass the result to your RAG chain for final decision.
    final_decision = rag_inference(local_result)
    
    # Return the decision text to the front end.
    return InferenceResponse(decision=final_decision)


# class AnalysisResponse(BaseModel):
#     decision: str
#     reasoning: str
#     confidence: float

# def analyze_statement(text: str) -> dict:
#     """Mock ML/RAG analysis function"""
#     # Replace with actual ML/RAG logic
#     return {
#         "decision": "Approved" if "salary" in text.lower() else "Denied",
#         "reasoning": "Regular income detected" if "salary" in text.lower() else "Insufficient income history",
#         "confidence": 0.85
#     }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
