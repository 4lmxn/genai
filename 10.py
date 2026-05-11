import getpass
import wikipediaapi
from langchain import PromptTemplate
from langchain.llms import Cohere
from pydantic import BaseModel
from typing import Optional

COHERE_API_KEY = getpass.getpass("Enter your Cohere API Key: ")
cohere_llm = Cohere(cohere_api_key=COHERE_API_KEY, model="command")

def fetch_ipc_summary():
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent="IPCChatbot/1.0 (contact: myemail@example.com)",
        language='en'
    )
    page = wiki_wiki.page("Indian Penal Code")
    if not page.exists():
        raise ValueError("IPC page not found on Wikipedia.")
    return page.text[:5000]

ipc_content = fetch_ipc_summary()

class IPCResponse(BaseModel):
    section: Optional[str]
    explanation: Optional[str]

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=f"""
You are a legal assistant chatbot for the Indian Penal Code (IPC).
Use the content below to answer the user's question.

IPC Content:
{ipc_content}

User Question: {{question}}

Give a detailed answer and mention the relevant section if applicable.
"""
)

def get_ipc_response(question: str) -> IPCResponse:
    formatted_prompt = prompt_template.format(question=question)
    response = cohere_llm.predict(formatted_prompt)

    if "Section" in response:
        section = response.split('Section')[1].split(':')[0].strip()
        explanation = response.split(':', 1)[-1].strip()
    else:
        section = None
        explanation = response.strip()

    return IPCResponse(section=section, explanation=explanation)

print("IPC CHATBOT READY. Type 'bye' to exit.")
while True:
    question = input("You: ")
    if question.lower() == "bye":
        print("Bot: Goodbye!")
        break
    response = get_ipc_response(question)
    if response.section:
        print(f"Bot: [Section {response.section}] {response.explanation}\n")
    else:
        print(f"Bot: {response.explanation}\n")
