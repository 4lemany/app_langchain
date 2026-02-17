import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

st.set_page_config(page_title="Agente Investigador", page_icon="🕵️‍♂️")

st.title("🕵️‍♂️ Agente Investigador con Memoria")
st.markdown("Este agente puede buscar información en internet usando DuckDuckGo.")

# --- BARRA LATERAL: CONFIGURACIÓN ---
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Introduce tu Google API Key:", type="password")
    st.markdown("[Obtén tu API Key de Google aquí](https://aistudio.google.com/app/apikey)")

# --- INICIALIZAR MEMORIA EN STREAMLIT ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- MOSTRAR HISTORIAL EN LA INTERFAZ ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- INPUT DEL USUARIO ---
if prompt := st.chat_input("Pregúntame lo que quieras..."):
    # Mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not api_key:
        st.error("⚠️ Por favor, introduce tu Google API Key en la barra lateral para continuar.")
    else:
        # Configurar variable de entorno para que Langchain la detecte
        os.environ['GOOGLE_API_KEY'] = api_key
        
        with st.spinner("Pensando e investigando..."):
            try:
                # 1. Configurar LLM y Herramientas
                llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
                tools = [DuckDuckGoSearchRun()]

                # 2. Configurar Prompt y Agente
                agent_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Eres un asistente útil. Usa DuckDuckGo para buscar información actual si la necesitas para responder."),
                    ("placeholder", "{history}"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])
                
                agent = create_tool_calling_agent(llm, tools, agent_prompt)
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

                # 3. Conectar la memoria
                store = {"sesion_unica": st.session_state.chat_history}
                def get_session_history(session_id: str):
                    return store[session_id]

                app_langchain = RunnableWithMessageHistory(
                    agent_executor,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="history",
                )

                # 4. Ejecutar agente
                respuesta = app_langchain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": "sesion_unica"}}
                )
                
                output_text = respuesta['output']

                # 5. Mostrar y guardar respuesta
                with st.chat_message("assistant"):
                    st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})

            except Exception as e:
                st.error(f"Ha ocurrido un error: {e}")