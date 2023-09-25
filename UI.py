import tkinter as tk
from tkinter import ttk
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai

# Load pretrained GPT-2 model and tokenizer from Hugging Face's Model Hub
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

# Initialize OpenAI API
openai.api_key = "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"

# Load pretrained LangChain (LLM) model and tokenizer from Hugging Face's Model Hub
llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
llm_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

def generate_responses():
    prompt = text_input.get("1.0", 'end-1c')  # Extract input from tkinter Text widget
    
    # GPT-2 response
    input_ids_gpt2 = gpt2_tokenizer.encode(prompt, return_tensors='pt')
    output_gpt2 = gpt2_model.generate(input_ids_gpt2, max_length=100, num_return_sequences=1)
    response_gpt2 = gpt2_tokenizer.decode(output_gpt2[0], skip_special_tokens=True)
    
    # GPT-3 response
    response_gpt3 = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=100).choices[0].text.strip()

    # LangChain (LLM) response
    input_ids_llm = llm_tokenizer.encode(prompt, return_tensors='pt')
    output_llm = llm_model.generate(input_ids_llm, max_length=100, num_return_sequences=1)
    response_llm = llm_tokenizer.decode(output_llm[0], skip_special_tokens=True)
    
    # Display results in the tkinter Text widgets
    gpt2_response.delete("1.0", tk.END)
    gpt2_response.insert(tk.END, response_gpt2)

    gpt3_response.delete("1.0", tk.END)
    gpt3_response.insert(tk.END, response_gpt3)

    llm_response.delete("1.0", tk.END)
    llm_response.insert(tk.END, response_llm)

# Create the tkinter window
root = tk.Tk()
root.title("Response Generator")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Input prompt widget
ttk.Label(frame, text="Enter your prompt:").grid(row=0, column=0, sticky=tk.W, pady=5)
text_input = tk.Text(frame, height=5, width=40)
text_input.grid(row=1, column=0, columnspan=2, pady=5)

# Button to generate responses
btn_generate = ttk.Button(frame, text="Generate Responses", command=generate_responses)
btn_generate.grid(row=2, column=0, columnspan=2, pady=10)

# GPT-2 response widget
ttk.Label(frame, text="GPT-2 Response:").grid(row=3, column=0, sticky=tk.W, pady=5)
gpt2_response = tk.Text(frame, height=5, width=40)
gpt2_response.grid(row=4, column=0, columnspan=2, pady=5)

# GPT-3 response widget
ttk.Label(frame, text="GPT-3 Response:").grid(row=5, column=0, sticky=tk.W, pady=5)
gpt3_response = tk.Text(frame, height=5, width=40)
gpt3_response.grid(row=6, column=0, columnspan=2, pady=5)

# LangChain (LLM) response widget
ttk.Label(frame, text="LangChain (LLM) Response:").grid(row=7, column=0, sticky=tk.W, pady=5)
llm_response = tk.Text(frame, height=5, width=40)
llm_response.grid(row=8, column=0, columnspan=2, pady=5)

root.mainloop()
