import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
import openai
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
import random

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Responda com uma receita para a solicitação abaixo:

{context}

---

Responda com base no contexto acima: {question}
"""

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])



# Prepare the DB.
embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def get_response_from_gpt3(query_text, context):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=query_text)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Você é um assistente de receitas saudáveis que fornece um cardápio amplo de opções de café da manhã, almoço e jantar."},
            {"role": "user", "content": f"{prompt}"}
        ]
    )

    return response.choices[0].message.content.strip()


def get_response_from_model(query_text):
    results = db.similarity_search_with_relevance_scores(query_text, k=4)
    if len(results) == 0 or results[0][1] < 0.7:
        return "Não foi possível encontrar resultados correspondentes.", []
    
    random.shuffle(results)


    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    response_text = get_response_from_gpt3(query_text, context_text)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    response_with_sources = f"{response_text}\n\nFontes: {sources[0]}"
    return response_with_sources, sources

def on_input_change():
    user_input = st.session_state.user_input
    response_text, sources = get_response_from_model(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append({"type": "normal", "data": response_text})

def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]

st.session_state.setdefault('past', [])
st.session_state.setdefault('generated', [])

st.title("Chat de Receitas Saudáveis")

chat_placeholder = st.empty()

with chat_placeholder.container():    
    for i in range(len(st.session_state['generated'])):                
        message(st.session_state['past'][i], is_user=True, key=f"{i}_user")
        message(
            st.session_state['generated'][i]['data'], 
            key=f"{i}", 
            allow_html=True,
            is_table=True if st.session_state['generated'][i].get('type') == 'table' else False
        )
    
    st.button("Limpar histórico", on_click=on_btn_click)

with st.container():
    st.text_input("Mensagem:", on_change=on_input_change, key="user_input")
