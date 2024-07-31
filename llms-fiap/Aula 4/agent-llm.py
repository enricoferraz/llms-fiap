import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import matplotlib.pyplot as plt
import yfinance as yf
from dotenv import load_dotenv
import os

#python -m streamlit run "agent-llm.py"

load_dotenv()

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

#ativos = [“ITUB3.SA”, “VALE3.SA”, “PETR3.SA”, “^BVSP”]

def get_preco_acao(ticker):
    print(ticker)
    return str(yf.Ticker(ticker).history(period='1y').iloc[-1].Close)


def plot_preco_acao(ticker):

    data = yf.Ticker(ticker).history(period='1y')
    plt.figure(figsize=(10,5))
    plt.plot(data.index, data.Close)
    plt.title(f"{ticker} - Cotação da Ação no Último Ano")
    plt.xlabel("Data")
    plt.ylabel("Preço na Ação em BRL")
    plt.grid(True)
    plt.savefig(f'./imagens/{ticker}.png')
    plt.close()


funcoes = [
    {
       "type": "function",
        "function": {
            "name": "get_preco_acao",
            "description": "Gets the latest stock price given the ticker symbol of that stock along with the date of the price formatted in the brazilian date system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The ticker symbol for a company stock in Brazil (Common shares have a 3 and preferred shares have a 4) in the format of yahoo finance library (e.g. VALE3.SA for VALE DO RIO DOCE)."
                        }
                },
                "required": ["ticker"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_preco_acao",
            "description": "Plot a graph with the price over time for a given company's ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The ticker symbol for a company stock in Brazil (Common shares have a 3 and preferred shares have a 4) in the format of yahoo finance library (e.g. VALE3.SA for VALE DO RIO DOCE)"
                        }
                },
                "required": ["ticker"]
            },
        }
    }
]

available_functions = {
    "get_preco_acao" : get_preco_acao,
    "plot_preco_acao" : plot_preco_acao
}

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {
            'role' : 'system', 
            'content' : 'Você é um assistente de investimento que auxilia no fornecimento de informações de ações brasileiras.'}
    ]

st.title("Assistente de Investimentos")





user_input = st.text_input("Pergunte: ")

if user_input:
    try:
        st.session_state['messages'].append({
            'role' : 'user', 
            'content' : f'{user_input}'})

        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=st.session_state['messages'],
        tools=funcoes,
        tool_choice='auto'
    )   
        response_message = response.choices[0].message


        if response_message.tool_calls:

            # print(response_message)
            # print(type(response_message))


            function_name = response_message.tool_calls[0].function.name
            function_agrs = json.loads(response_message.tool_calls[0].function.arguments)
            tool_call_id =  response_message.tool_calls[0].id

            print( function_name, function_agrs)

            if function_name in [ "get_preco_acao" ,"plot_preco_acao"] :
                args_dict = {'ticker' : function_agrs.get('ticker')}
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**args_dict)

            print(function_agrs.get('ticker'))
        
            if function_name == 'plot_preco_acao':
                st.image(f"./imagens/{function_agrs.get('ticker')}.png")
            else:
                st.session_state['messages'].append(response_message)
                st.session_state['messages'].append(
                    {
                        'role' : 'tool',
                        "tool_call_id":tool_call_id,
                        'name' : function_name,
                        'content' : function_response
                    }
                )

                second_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state['messages'],
                )
                print(second_response.choices[0].message.content)

                st.text(second_response.choices[0].message.content.strip())
                st.session_state['messages'].append(
                    {
                        'role' : 'assistant',
                        'content' : f'{second_response.choices[0].message.content.strip()}'
                    }
                )
        else:
            st.text(response_message['content'])
            st.session_state['messages'].append(
                    {
                        'role' : 'assistant',
                        'content' : f"{response_message['content']}"
                    }
                )
    except:
        st.text('Tente novamente...')