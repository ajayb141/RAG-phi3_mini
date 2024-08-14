from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import VLLM
from langchain.chains import RetrievalQA
import gradio as gr
import torch

embeddings = FakeEmbeddings(size=1324)
llm = VLLM(
    model="microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=True,
    dtype="half",
    temperature=0.1,
    gpu_memory_utilization=0.9,
    vllm_kwargs={
        "max_model_len": 512,
    },
    max_new_tokens=128,
)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
torch.cuda.empty_cache()


def chatbot(file, query):
    loader = UnstructuredFileLoader(file)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embeddings)
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(), input_key="question")
    sys_prompt = "You are an AI language model developed by IBM Research. Only return Answer for the user query, Answer in only few words, Do not return any extra information or notes. Stop the answer once the user's query is addressed."
    prompt = f"<|system|>\n{sys_prompt}<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>"
    # stop_token = "<|endoftext|>"
    final = chain.invoke(prompt)
    return final["result"]


iface = gr.Interface(fn=chatbot, inputs=["file", "text"], outputs="text", title="PDF chatbot", allow_flagging="never").launch(server_name="0.0.0.0")