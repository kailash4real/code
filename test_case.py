import pandas as pd

# Sample test queries for the CSV
test_queries = [
    "What are the symptoms of COVID-19?",
    "How does the coronavirus spread?",
    "What treatments are available for the virus?",
    "How long does the virus survive on surfaces?",
    "What are the preventive measures for COVID-19?",
    "How effective are masks in preventing the spread?",
    "What's the difference between COVID-19 and the flu?",
    "How does the vaccine work?",
    "What are the side effects of the vaccine?",
    "Is it safe to travel during the pandemic?",
    "How many cases have been reported globally?",
    "What is herd immunity?",
    "How does testing for the virus work?",
    "How long should one quarantine if exposed?",
    "Are there any travel restrictions in place?",
    "What is the difference between isolation and quarantine?",
    "How is the pandemic affecting the economy?",
    "What research is being done on the virus?",
    "Are animals susceptible to the virus?",
    "How do I protect myself from the virus?",
    "What age group is most affected by the virus?",
    "How is the virus mutating?",
    "What are the long-term effects of the virus?",
    "Are there any home remedies for the virus?",
    "How is the virus impacting mental health?",
    "What precautions should pregnant women take?",
    "How is the vaccine distributed?",
    "How many doses of the vaccine are required?",
    "Can I get infected after vaccination?",
    "How do I report side effects from the vaccine?",
    "Is there a digital passport for vaccination?",
    "How is the pandemic affecting children?",
    "What's the difference between different vaccines?",
    "How are hospitals coping with the pandemic?",
    "What's the role of WHO in the pandemic?",
    "How do I get tested for the virus?",
    "What are antibody tests?",
    "How reliable are the COVID-19 tests?",
    "How is contact tracing done?",
    "What is the reproduction number of the virus?",
    "How are recovered patients monitored?",
    "Are there any post-recovery complications?",
    "How do variants of the virus emerge?",
    "Is reinfection possible with COVID-19?",
    "How long does immunity last after infection?",
    "Are children vectors for the virus?",
    "How is data being used to combat the pandemic?",
    "What role do super-spreaders play in the spread?",
    "How do health departments track the virus spread?",
    "Should children get vaccinated?"
]

# Convert the list to a DataFrame
df_test_queries = pd.DataFrame(test_queries, columns=["test_case"])

# Save the DataFrame to a CSV file
file_path = "data/test_cases.csv"
df_test_queries.to_csv(file_path, index=False)

file_path
