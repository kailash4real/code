import openai
import pandas as pd
import os

# Paths
TEST_CASES_FILE = "./data/test_cases.csv"
RESPONSES_FILE = "./data/gpt3_responses.csv"

# Initialize OpenAI API
openai.api_key = "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"

def generate_responses_gpt3():
    # Check if test_cases.csv exists
    if not os.path.exists(TEST_CASES_FILE):
        print(f"{TEST_CASES_FILE} does not exist. Please provide this file with your test cases.")
        return

    # Load test cases
    test_cases = pd.read_csv(TEST_CASES_FILE)
    
    # Initialize an empty list to store responses
    responses = []
    
    for index, row in test_cases.iterrows():
        test_case = row['test_case']
        print(f"Generating response for test case {index+1}: {test_case}")
        
        # Generate response using GPT-3
        prompt = test_case
        response = openai.Completion.create(
          engine="text-davinci-002",
          prompt=prompt,
          max_tokens=100
        )
        
        # Extract and store the generated text
        generated_text = response.choices[0].text.strip()
        responses.append(generated_text)

    # Save the responses to a CSV file
    responses_df = pd.DataFrame({'test_case': test_cases['test_case'], 'gpt3_response': responses})
    responses_df.to_csv(RESPONSES_FILE, index=False)
    print(f"Responses saved to {RESPONSES_FILE}")

if __name__ == "__main__":
    generate_responses_gpt3()
