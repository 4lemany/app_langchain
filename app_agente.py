from langchain_core.runnables import RunnableLambda

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
    
    # 4. Agente
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # 5. Función para extraer siempre texto (¡La clave para evitar el error!)
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

    # Conectar el formateador al ejecutor
    agent_executor_with_formatted_output = agent_executor | RunnableLambda(ensure_string_output)
    
    # 6. Agente con Memoria
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor_with_formatted_output,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent_with_chat_history
