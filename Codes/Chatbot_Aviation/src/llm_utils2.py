# import os
# import re
# import openai
# from dotenv import load_dotenv, find_dotenv
# import streamlit as st
# from langchain_community.chat_models import ChatOpenAI
# from langchain.agents import AgentType
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain.schema.output_parser import OutputParserException

# load_dotenv()  # Load environment variables from .env file
# client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def chat_api(
#     messages, model="gpt-4", temperature=0.0, max_tokens=256, top_p=0.5
# ):
#     plot_flag = False
#     plot_keywords = ["plot", "graph", "chart", "visualize"]
#     if any(keyword in messages[-1]["content"].lower() for keyword in plot_keywords):
#         plot_flag = True
#     else:
#         plot_flag= False

#     client = openai.Client()

#     response = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=temperature,
#         max_tokens=max_tokens,
#         top_p=top_p,
#         frequency_penalty=0,
#         presence_penalty=0,
#         stop=None
#     )

#     if plot_flag and "```python" in response.choices[0].message.content:
#         code = extract_python_code(response.choices[0].message.content)
#         if code:
#             try:
#                 code = code.replace("fig.show()", "st.plotly_chart(fig, use_container_width=True)")
#                 exec(code)
#             except Exception as e:
#                 st.error(f"Error executing the plot: {e}")

#         plot_flag = False

#     return response.choices[0].message.content

# def extract_python_code(text):
#     pattern = r'```python\s(.*?)```'
#     matches = re.findall(pattern, text, re.DOTALL)
#     return matches[0] if matches else None

# def chat_with_data_api(df, model="gpt-4", temperature=0.0, max_tokens=256, top_p=0.5):
#     plot_flag = False
#     plot_keywords = ["plot", "graph", "chart"]

#     # ✅ Reset plot_flag before processing user input
#     if any(keyword in st.session_state.messages[-1]["content"].lower() for keyword in plot_keywords):
#         plot_flag = True
#     else:
#         plot_flag = False
    
    
#     if plot_flag:
#         code_prompt = """
#             Generate the code <code> for plotting the previous data in plotly,
#             in the format requested. The solution should be given using plotly
#             and only plotly. Do not use matplotlib.
#             Return the code <code> in the following
#             format ```python <code>```
#         """
#         st.session_state.messages.append({"role": "assistant", "content": code_prompt})
#         st.session_state.messages.append({
#             "role": "assistant",
#             "content": code_prompt
#         })
#         client = openai.Client()

#         response = client.chat.completions.create(
#                model=model,
#                messages=st.session_state.messages,
#                temperature=temperature,
#                max_tokens=max_tokens,
#                top_p=top_p
#         )

#         code = extract_python_code(response.choices[0].message.content)

#         if code:
#             try:
#                 code = code.replace("fig.show()", "st.plotly_chart(fig, use_container_width=True)")
#                 exec(code)
#             except Exception as e:
#                 st.error(f"Error executing the plot: {e}")
#             return response.choices[0].message.content
#     else:
#         llm = ChatOpenAI(
#             model=model,
#             temperature=temperature,
#             max_tokens=max_tokens,
#             top_p=top_p,
#         )

#         pandas_df_agent = create_pandas_dataframe_agent(
#             llm,
#             df,
#             verbose=True,
#             return_intermediate_steps=True,
#             agent_type=AgentType.OPENAI_FUNCTIONS,
#             handle_parsing_errors=False,
#             allow_dangerous_code=True
#         )

#         try:
#             answer = pandas_df_agent(st.session_state.messages)
#             return answer["output"]
#         except OutputParserException:
#             return "OutputParserException error occurred in LangChain agent. Refine your query."
#         except Exception as e:
#             return f"Unknown error occurred in LangChain agent: {str(e)}"


import os
import re
import openai
import streamlit as st
from dotenv import load_dotenv
import uuid

load_dotenv()  # Load environment variables from .env file
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_api(messages, model="gpt-4", temperature=0.0, max_tokens=256, top_p=0.5):
    """
    Handles general chat interactions and checks if a plot is requested.
    """
    plot_flag = False
    plot_keywords = ["plot", "graph", "chart", "visualize"]

    # ✅ Detect if the user asked for a plot
    if any(keyword in messages[-1]["content"].lower() for keyword in plot_keywords):
        plot_flag = True

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )

    if plot_flag:
        code = extract_python_code(response.choices[0].message.content)
        if code:
            try:
                exec(code.replace("fig.show()", "st.plotly_chart(fig, use_container_width=True)"))
            except Exception as e:
                st.error(f"Error executing the plot: {e}")

    return response.choices[0].message.content

def extract_python_code(text):
    pattern = r'```python\s(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0] if matches else None

def chat_with_data_api(df, model="gpt-4", temperature=0.0, max_tokens=256, top_p=0.5):
    """
    Processes dataframe-related queries, distinguishing between plots and normal responses.
    """
    plot_flag = False
    plot_keywords = ["plot", "graph", "chart"]

    user_query = st.session_state.messages[-1]["content"].lower()

    # ✅ Reset plot_flag before processing user input
    if any(keyword in user_query for keyword in plot_keywords):
        plot_flag = True
    else:
        plot_flag = False

    if plot_flag:
        code_prompt = """
            Generate a Plotly visualization using the given data. 
            Only return the Python code block in the following format:
            ```python
            <code>
            ```
        """
        st.session_state.messages.append({"role": "assistant", "content": code_prompt})

        response = client.chat.completions.create(
            model=model,
            messages=st.session_state.messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        code = extract_python_code(response.choices[0].message.content)

        if code:
            try:
                unique_key = str(uuid.uuid4())
                exec(code.replace("fig.show()", "st.plotly_chart(fig, use_container_width=True)"))
            except Exception as e:
                st.error(f"Error executing the plot: {e}")

        return response.choices[0].message.content

    else:
        # Handle general dataframe queries
        return f"The dataset contains {len(df)} rows and {len(df.columns)} columns."

