from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import pandas as pd

# Define the BLEU computation function
def compute_bleu(reference, candidate):
    reference = [word_tokenize(reference)]
    candidate = word_tokenize(candidate)
    bleu_score = sentence_bleu(reference, candidate)
    return bleu_score

# Load the responses
df1 = pd.read_csv('./data/gpt2_responses.csv')  
df2 = pd.read_csv('./data/gpt3_responses.csv')  
df3 = pd.read_csv('./data/langchain_responses.csv')  

# Assuming that df1, df2, and df3 have the same test cases in the same order
bleu_scores_12 = []
bleu_scores_13 = []
bleu_scores_23 = []

for idx, row in df1.iterrows():
    gpt2_response = row['gpt2_response']
    gpt3_response = df2.loc[idx, 'gpt3_response']
    llm_response = df3.loc[idx, 'langchain_response']

    bleu_scores_12.append(compute_bleu(gpt2_response, gpt3_response))
    bleu_scores_13.append(compute_bleu(gpt2_response, llm_response))
    bleu_scores_23.append(compute_bleu(gpt3_response, llm_response))

# Print average BLEU scores
print(f"Average BLEU score between GPT-2 and GPT-3: {sum(bleu_scores_12) / len(bleu_scores_12):.4f}")
print(f"Average BLEU score between GPT-2 and LLM: {sum(bleu_scores_13) / len(bleu_scores_13):.4f}")
print(f"Average BLEU score between GPT-3 and LLM: {sum(bleu_scores_23) / len(bleu_scores_23):.4f}")
