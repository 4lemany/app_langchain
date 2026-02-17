import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

# Configuración de la página
st.set_page_config(page_title="Agente Investigador LangChain", page_icon="🕵️‍♂️")

st.title("🕵️‍♂️ Agente Investigador con LangChain")
st.markdown("Este agente puede buscar en internet y recordar nuestra conversación.")

# --- SIDEBAR: CONFIGURACIÓN ---
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Introduce tu Google API Key:", type="password")
    st.markdown("[Consigue tu API Key aquí](https://aistudio.google.com/app/apikey)")

# --- LÓGICA DEL AGENTE ---
if not api_key:
    st.warning("⚠️ Por favor, introduce tu Google API Key en la barra lateral para continuar.")
    st.stop()

# Configurar la API Key en el entorno
os.environ['GOOGLE_API_KEY'] = api_key

# Inicializar Memoria en Session State si no existe
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Función para obtener el historial (necesaria para RunnableWithMessageHistory)
def get_session_history(session_id: str):
    return st.session_state.chat_history

# Inicializar y Cacher el Agente (para no recargarlo en cada interacción)
@st.cache_resource
def setup_agent(google_api_key):
    # 1. Modelo
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=google_api_key)
    
    # 2. Herramientas
    search = DuckDuckGoSearchRun()
    tools = [search]
    
    # 3. Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil. Usa DuckDuckGo para buscar información actual si es necesario."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # 4. Agente y Ejecutor
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # 5. Limpiar salida (Arregla el error de "Message dict must contain 'role'")
    def ensure_string_output(agent_result: dict) -> dict:
        output_value = agent_result.get('output')
        if isinstance(output_value, list):
            concatenated_text = ""
            for item in output_value:
                if isinstance(item, dict) and item.get('type') == 'text':
                    concatenated_text += item.get('text', '')
                elif isinstance(item, str):
                    concatenated_text += item
            agent_result['output'] = concatenated_text
        elif not isinstance(output_value, str):
            agent_result['output'] = str(output_value)
        return agent_result

    # Conectar el formateador al ejecutor usando un RunnableLambda
    agent_executor_with_formatted_output = agent_executor | RunnableLambda(ensure_string_output)
    
    # 6. Agente con Memoria
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor_with_formatted_output,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent_with_chat_history

try:
    agent_chain = setup_agent(api_key)
except Exception as e:
    st.error(f"Error al configurar el agente: {e}")
    st.stop()

# --- INTERFAZ DE CHAT ---

# Mostrar mensajes anteriores
for msg in st.session_state.chat_history.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# Input del usuario
user_input = st.chat_input("Pregunta algo (ej: ¿Cuándo es el próximo partido del Real Madrid?)")

if user_input:
    # 1. Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.write(user_input)
    
    # 2. Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Investigando en internet..."):
            try:
                response = agent_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": "session_unica"}}
                )
                st.write(response["output"])
            except Exception as e:
                st.error(f"Ocurrió un error en la ejecución: {e}")

