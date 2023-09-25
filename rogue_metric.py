from rouge import Rouge
import pandas as pd

# Load the CSV files
df1 = pd.read_csv('./data/gpt2_responses.csv')
df2 = pd.read_csv('./data/gpt3_responses.csv')
df3 = pd.read_csv('./data/langchain_responses.csv')

# Merge the dataframes on the 'test_case' column
merged_df = df1.merge(df2, on="test_case").merge(df3, on="test_case")
merged_df = merged_df.rename(columns={"gpt3_response": "gpt3_response", "langchain_response": "llm_response"})

rouge = Rouge()

# Calculate average scores
def average_rouge(scores_list):
    avg_scores = {
        'rouge-1': {'r': 0, 'p': 0, 'f': 0},
        'rouge-2': {'r': 0, 'p': 0, 'f': 0},
        'rouge-l': {'r': 0, 'p': 0, 'f': 0}
    }
    for scores in scores_list:
        for key in scores:
            avg_scores[key]['r'] += scores[key]['r']
            avg_scores[key]['p'] += scores[key]['p']
            avg_scores[key]['f'] += scores[key]['f']
    n = len(scores_list)
    for key in avg_scores:
        avg_scores[key]['r'] /= n
        avg_scores[key]['p'] /= n
        avg_scores[key]['f'] /= n
    return avg_scores

# Calculate and store the ROUGE scores in a DataFrame
data = {
    "Models Compared": ["GPT-2 vs GPT-3", "GPT-2 vs LLM", "GPT-3 vs LLM"],
    "ROUGE-1 Recall": [
        average_rouge(rouge.get_scores(merged_df['gpt2_response'], merged_df['gpt3_response']))['rouge-1']['r'],
        average_rouge(rouge.get_scores(merged_df['gpt2_response'], merged_df['llm_response']))['rouge-1']['r'],
        average_rouge(rouge.get_scores(merged_df['gpt3_response'], merged_df['llm_response']))['rouge-1']['r'],
    ],
    "ROUGE-1 Precision": [
        average_rouge(rouge.get_scores(merged_df['gpt2_response'], merged_df['gpt3_response']))['rouge-1']['p'],
        average_rouge(rouge.get_scores(merged_df['gpt2_response'], merged_df['llm_response']))['rouge-1']['p'],
        average_rouge(rouge.get_scores(merged_df['gpt3_response'], merged_df['llm_response']))['rouge-1']['p'],
    ],
    "ROUGE-1 F1 Score": [
        average_rouge(rouge.get_scores(merged_df['gpt2_response'], merged_df['gpt3_response']))['rouge-1']['f'],
        average_rouge(rouge.get_scores(merged_df['gpt2_response'], merged_df['llm_response']))['rouge-1']['f'],
        average_rouge(rouge.get_scores(merged_df['gpt3_response'], merged_df['llm_response']))['rouge-1']['f'],
    ]
    # You can extend this for ROUGE-2 and ROUGE-L similarly
}

# Convert the data to a DataFrame
scores_df = pd.DataFrame(data)
print(scores_df)
