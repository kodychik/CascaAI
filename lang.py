import os
import getpass
import bs4
from typing import List, TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import START, StateGraph
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from modal import App, Image, Volume, enter, method

#app = App("bank-statement-analyzer")

# Set your Mistral API key if not already set


# Define prompt for question-answering
#prompt = hub.pull("rlm/rag-prompt")


# # Set up LangChain vectorstore (reuse embeddings from above if possible)
# lc_embeddings = LC_HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# vector_store = LC_Chroma(embedding_function=lc_embeddings)

# # Load and chunk the PDF document (reuse the same bank statement PDF)
# lc_loader = LC_PyPDFLoader("bank_statements/loanportmanage.pdf")
# lc_documents = lc_loader.load()
# lc_text_splitter = LC_RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# all_splits = lc_text_splitter.split_documents(lc_documents)
# vector_store.add_documents(documents=all_splits)



prompt_template = PromptTemplate.from_template("""
Based on the bank statement informaiton, does this person deserve a loan? Use the context from the documents.
Assistant: ```json
{{
    "date": "30 November 2019",
    "sort_code": "11-10-51",
    "bic": "LOYDGB21033",
    "account_number": "45201526",
    "iban": "GB17LOYD30135525281061",
    "customer": {{
        "name": "Oleh Beshleu",
        "address": "293 STRONE ROAD\nMANOR PARK(BARKING AND DAGENHAM)\nLONDON\nE12 6TR"
    }},
    "statement_period": "01 September 2019 to 30 November 2019",
    "transactions": [
        {{
            "date": "04 Sep19",
            "description": "LINK COOPERATIVE SW CD 4821 08DEC19",
            "type": "CPT",
            "in": 300.00,
            "out": 873.56,
            "balance": 1173.56
        }},
        {{
            "date": "09 Sep19",
            "description": "P ADAMCZUK KASA 7420",
            "type": "TFR",
            "in": 50.00,
            "out": 923.56,
            "balance": 921.06
        }},
        {{
            "date": "11 Sep19",
            "description": "BARCLAYCARD CD 7420",
            "type": "DEB",
            "in": 2.50,
            "out": 921.06,
            "balance": 821.06
        }},
        {{
            "date": "15 Sep19",
            "description": "WESTERN VILLA CD 7420",
            "type": "DEB",
            "in": 240.00,
            "out": 681.06,
            "balance": 681.06
        }},
        {{
            "date": "16 Sep19",
            "description": "WINELEAF CD 7420",
            "type": "DEB",
            "in": 26.69,
            "out": 654.37,
            "balance": 654.37
        }},
        {{
            "date": "17 Sep19",
            "description": "LINK COOPERATIVE SW CD 7420 06DEC19",
            "type": "CPT",
            "in": 200.00,
            "out": 454.37,
            "balance": 454.37
        }},
        {{
            "date": "18 Sep19",
            "description": "LV LIFE 03592291015W",
            "type": "DD",
            "in": 33.03,
            "out": 421.34,
            "balance": 421.34
        }},
        {{
            "date": "22 Sep19",
            "description": "PARK FOOD AND WINE CD 7420",
            "type": "DEB",
            "in": 30.46,
            "out": 390.88,
            "balance": 390.88
        }},
        {{
            "date": "22 Sep19",
            "description": "LINK COOPERATIVE SW CD 7420 05DEC19",
            "type": "CPT",
            "in": 20.00,
            "out": 370.88,
            "balance": 370.88
        }},
        {{
            "date": "25 Sep19",
            "description": "LINK COOPERATIVE SW CD 7420 03DEC19",
            "type": "CPT",
            "in": 200.00,
            "out": 170.88,
            "balance": 170.88
        }},
        {{
            "date": "26 Sep19",
            "description": "UBER *TRIP LJSAM",
            "type": "DEB",
            "in": 17.43,
            "out": 153.45,
            "balance": 153.45
        }},
        {{
            "date": "29 Sep19",
            "description": "D ROBERTSON",
            "type": "DEB",
            "in": 960.00,
            "out": 1113.45,
            "balance": 1113.45
        }},
        {{
            "date": "03 Oct 19",
            "description": "WINELEAF CD 7420",
            "type": "DEB",
            "in": 17.00,
            "out": 1096.45,
            "balance": 1096.45
        }},
        {{
            "date": "05 Oct 19",
            "description": "KATE EYG D LTD KEGD",
            "type": "FPI",
            "in": 790.00,
            "out": 1886.45,
            "balance": 1886.45
        }}
    ]
}}
```json

Context: {context}

Question: {question}

Answer:
""")


if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(embedding_function=embeddings)

pdf_path = "bank_statements/loanportmanage.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(documents)

# Index the document chunks
_ = vector_store.add_documents(documents=all_splits)

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Helper function to extract a snippet (around 200 characters) from a document
def extract_snippet(doc: Document, query: str, snippet_length: int = 200) -> str:
    content = doc.page_content
    lower_content = content.lower()
    lower_query = query.lower()
    pos = lower_content.find(lower_query)
    if pos != -1:
        start = max(0, pos - snippet_length // 2)
        end = start + snippet_length
        return content[start:end]
    else:
        return content[:snippet_length]


# Define application steps
def retrieve(state: State):
    # Retrieve the top 5 documents most similar to the query
    retrieved_docs = vector_store.similarity_search(state["question"], k=5)
    # Replace each document's content with a 200-character snippet around the query
    snippet_docs = []
    for doc in retrieved_docs:
        snippet = extract_snippet(doc, state["question"])
        snippet_docs.append(Document(page_content=snippet, metadata=doc.metadata))
    return {"context": snippet_docs}

def generate(state: State):
    llm = ChatMistralAI(model="mistral-large-latest")
    # Concatenate the snippets from the retrieved documents
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    # Invoke the prompt using the question and the context snippets
    messages = prompt_template.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# class State(TypedDict):
#     question: str
#     context: List[]
#     answer: str

def rag_inference(query: str) -> str:
    state: State = {"question": query, "context": [], "answer": ""}
    retrieve_out = retrieve(state)
    state["context"] = retrieve_out["context"]
    generate_out = generate(state)
    state["answer"] = generate_out["answer"]
    print(state["answer"])
    return state["answer"]
# # Compile application and test
# graph_builder = StateGraph(State).add_sequence([retrieve, generate])
# graph_builder.add_edge(START, "retrieve")
# graph = graph_builder.compile()

# # Test query: determine if the person deserves a loan based on the bank statement
# response = graph.invoke({"question": "Based on the bank statement, does this person deserve a loan?"})
# print(response["answer"])
