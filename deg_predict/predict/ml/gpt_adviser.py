# predict/ml/gpt_adviser.py

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Initialize OpenRouter client
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def generate_advice(gpa, sentiment, dropout_factors):
    prompt = f"""
        Take note on GPA: 
        1.00	Passed	Excellent
        1.25	Passed	Superior
        1.50	Passed	Very Good
        1.75	Passed	Good
        2.00	Passed	Above Average
        2.25	Passed	Average
        2.50	Passed	Satisfactory
        2.75	Passed	Fair
        3.00	Passed	Lowest Passing Grade
        Above 3.00		Failed	Failing Grade
You are an academic adviser assistant. Based on the following student information:
- Dropout Factors: {', '.join(dropout_factors) if dropout_factors else 'None'}
- GPA: {gpa}
- Sentiment: {sentiment}

Please generate personalized academic advice to help the student succeed. 
Present your suggestions in **clear bullet points**, focusing on practical and supportive steps.
"""

    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",  # OpenRouter free model
            messages=[
                {"role": "system", "content": "You are a helpful academic adviser."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating advice: {str(e)}"
