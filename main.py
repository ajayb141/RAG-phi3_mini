import torch 
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

# torch.random.manual_seed(0) 
model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3-mini-128k-instruct",  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
) 

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct") 

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 
def run_conversation(prompt):
    messages = [ 
        {"role": "system", "content": "You are a helpful assistant who answers user queries with long context."}, 
        {"role": "user", "content": prompt}, 
    ] 
    generation_args = { 
        "max_new_tokens": 4096, 
        "return_full_text": False, 
        "temperature": 0.3, 
        "do_sample": False, 
    } 

    output = pipe(messages, **generation_args) 
    gen_text = output[0]['generated_text']
    return str(gen_text)

# iface = gr.ChatInterface(fn = run_conversation, additional_inputs=gr.Textbox(label="System Prompt"))
iface = gr.Interface(fn= run_conversation, inputs= gr.Text(label="Enter your prompt :"), outputs= gr.Text(label="Generated Text"), title="Phi-3-mini", allow_flagging="never")
iface.launch(server_name="0.0.0.0")