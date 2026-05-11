import getpass
from langchain import PromptTemplate
from langchain.llms import Cohere

COHERE_API_KEY = getpass.getpass("Enter your Cohere API Key: ")

with open("sample_text.txt", "r", encoding="utf-8") as file:
    text_content = file.read()

cohere_llm = Cohere(cohere_api_key=COHERE_API_KEY, model="command")

template = """
You are an AI assistant helping to analyze a text document.
Here is the document:
{text}

Summary:
- Provide a concise summary.

Key Takeaways:
- List 3 important points.

Sentiment:
- Is the document Positive, Negative, or Neutral?
"""

prompt_template = PromptTemplate(input_variables=["text"], template=template)
formatted_prompt = prompt_template.format(text=text_content)
response = cohere_llm.predict(formatted_prompt)

print(response)
