# import os
# import re
# import openai
# from dotenv import load_dotenv, find_dotenv
# import streamlit as st
# from langchain_community.chat_models import ChatOpenAI
# from langchain.agents import AgentType
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain.schema.output_parser import OutputParserException
# import matplotlib.pyplot as plt
# import seaborn as  sns
# import pandas as pd
# import uuid

# # if os.environ.get('OPENAI_API_KEY') is not None:
# #     openai.api_key = os.environ['OPENAI_API_KEY']
# # else:
# #     _ = load_dotenv(find_dotenv())  # read local .env file
# #     openai.api_key = os.environ['OPENAI_API_KEY']


# ##--------------------------LOAD THE OPEN AI KEY--------------------------------##  q1
# load_dotenv()  # Load environment variables from .env file
# client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ##------------------------------------------------------------------------------##
# def chat_api(
#     messages, model="gpt-4", temperature=0.0, max_tokens=256, top_p=0.5
# ):
#     """
#     The chat API endpoint of the ChatGPT

#     Args:
#         messages (str): The input messages to the chat API
#         model (str): The model, i.e. the LLM
#         temperature (float): The temperature parameter
#         max_tokens (int): Max number of tokens parameters
#         top_p (float): Top P parameter

#     Returns:
#         str: The LLM response
#     """

#     # response = client.chat.completions.create(
#     #         model=model,
#     #         messages=messages,
#     #         temperature=temperature,
#     #         max_tokens=max_tokens,
#     #         top_p=top_p

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
#         # ✅ Ensure it replaces fig.show() with Streamlit's built-in function
#             unique_key = str(uuid.uuid4())  # Generate a unique key for each chart
#             code = code.replace("fig.show()", f"st.plotly_chart(fig, use_container_width=True, key='{unique_key}')")
#             local_env = {"st": st, "plt": plt, "pd": pd, "sns": sns, "fig": None}
#             try:
#                 exec(code, local_env)  # Execute in local environment
#                 fig = local_env.get("fig")  # Get the generated figure if exists
#                 if fig:
#                    st.plotly_chart(fig, use_container_width=True, key=unique_key)  # Display plot only once
#             except Exception as e:
#                 st.error(f"Error executing the plot: {e}")

#     # ✅ Reset plot_flag so next questions don't trigger plotting again
#         plot_flag = False



#     return response.choices[0].message.content


# def extract_python_code(text):
#     pattern = r'```python\s(.*?)```'
#     matches = re.findall(pattern, text, re.DOTALL)
#     if not matches:
#         return None
#     else:
#         return matches[0]


# def chat_with_data_api(df, model="gpt-4", temperature=0.0, max_tokens=256, top_p=0.5):
#     """
#     A function that answers data questions from a dataframe.
#     """
#     plot_flag = False
#     plot_keywords = ["plot", "graph", "chart", "visualize"]
    
#     if any(keyword in st.session_state.messages[-1]["content"].lower() for keyword in plot_keywords):
#         if not st.session_state.get("plot_generated", False):  # ✅ Prevents multiple plots
#             plot_flag = True
#             st.session_state["plot_generated"] = True  # ✅ Mark plot as generated
            
#             code_prompt = """
#             Generate the code <code> for plotting the previous data in plotly,
#             in the format requested. The solution should be given using plotly
#             and only plotly. Do not use matplotlib.
#             Return the code <code> in the following
#             format ```python <code>```
#             """
            
#             st.session_state.messages.append({
#                 "role": "assistant",
#                 "content": code_prompt
#             })

#         else:
#             plot_flag = False  # ✅ Prevents duplicate plots
        
#         client = openai.Client()

#         response = client.chat.completions.create(
#             model=model,
#             messages=st.session_state.messages,
#             temperature=temperature,
#             max_tokens=max_tokens,
#             top_p=top_p
#         )

#         code = extract_python_code(response.choices[0].message.content)

#         if plot_flag and "```python" in response.choices[0].message.content:
#            code = extract_python_code(response.choices[0].message.content)
        
#            if code:
#             unique_key = str(uuid.uuid4())  # ✅ Ensure unique key for plot display
#             code = code.replace("fig.show()", f"st.plotly_chart(fig, use_container_width=True, key='{unique_key}')")

#             # ✅ Execute safely in a separate environment
#             local_env = {"st": st, "plt": plt, "pd": pd, "sns": sns, "fig": None}
#             try:
#                 exec(code, local_env)
#                 fig = local_env.get("fig")  # ✅ Ensure fig is extracted properly
#                 if fig:
#                     st.plotly_chart(fig, use_container_width=True, key=unique_key)  # ✅ Display only one plot
#             except Exception as e:
#                 st.error(f"Error executing the plot: {e}")

#         return response.choices[0].message.content

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
#             allow_dangerous_code=True  # Add this line
#         )

#         try:
#             answer = pandas_df_agent(st.session_state.messages)
#             if answer["intermediate_steps"]:
#                 action = answer["intermediate_steps"][-1][0].tool_input["query"]
#                 st.write(f"Executed the code ```{action}```")
#             return answer["output"]
#         except OutputParserException:
#             error_msg = """OutputParserException error occurred in LangChain agent.
#                 Refine your query."""
#             return error_msg
#         except Exception as e:  # ✅ Corrected Exception Handling
#             error_msg = f"Unknown error occurred in LangChain agent: {str(e)}"
#             return error_msg  # ✅ Now 'error_msg' is always defined

import os
import re
import openai
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.schema.output_parser import OutputParserException

# if os.environ.get('OPENAI_API_KEY') is not None:
#     openai.api_key = os.environ['OPENAI_API_KEY']
# else:
#     _ = load_dotenv(find_dotenv())  # read local .env file
#     openai.api_key = os.environ['OPENAI_API_KEY']


##--------------------------LOAD THE OPEN AI KEY--------------------------------##  q1
load_dotenv()  # Load environment variables from .env file
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


##------------------------------------------------------------------------------##
def chat_api(
    messages, model="gpt-4", temperature=0.0, max_tokens=256, top_p=0.5
):
    """
    The chat API endpoint of the ChatGPT

    Args:
        messages (str): The input messages to the chat API
        model (str): The model, i.e. the LLM
        temperature (float): The temperature parameter
        max_tokens (int): Max number of tokens parameters
        top_p (float): Top P parameter

    Returns:
        str: The LLM response
    """
    plot_flag = False

    if "plot" in messages[-1]["content"].lower():
        plot_flag = True
        code_prompt = """
            Generate the code <code> for plotting the previous data in plotly,
            in the format requested. The solution should be given using plotly
            and only plotly. Do not use matplotlib.
            Return the code <code> in the following
            format ```python <code>```
        """
        messages.append({
            "role": "assistant",
            "content": code_prompt
        })

    client = openai.Client()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    if plot_flag and "```python" in response.choices[0].message.content:
        code = extract_python_code(response.choices[0].message.content)
        if code is None:
            st.warning(
                "Couldn't find data to plot in the chat. "
                "Check if the number of tokens is too low for the data at hand. "
                "I.e. if the generated code is cut off, this might be the case.",
                icon="🚨"
            )
        else:
            code = code.replace("fig.show()", "")
            code += """st.plotly_chart(fig, theme='streamlit', use_container_width=True)"""  # noqa: E501
            st.write(f"\n{code}")
            # Ensure no ellipsis in the data before executing
            exec("import pandas as pd\n" + code.replace("...", "None"))

            try:
                exec("import pandas as pd\n" + code.replace("...", "None"))
            except Exception as e:
                st.error(f"Error executing the plot: {e}")

        # ✅ Reset plot_flag so next questions don't trigger plotting again
        plot_flag = False

    return response.choices[0].message.content


def extract_python_code(text):
    pattern = r'```python\s(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    else:
        return matches[0]


def chat_with_data_api(df, model="gpt-4", temperature=0.0, max_tokens=256, top_p=0.5):
    """
    A function that answers data questions from a dataframe.
    """

    if "plot" in st.session_state.messages[-1]["content"].lower():
        code_prompt = """
            Generate the code <code> for plotting the previous data in plotly,
            in the format requested. The solution should be given using plotly
            and only plotly. Do not use matplotlib.
            Return the code <code> in the following
            format ```python <code>```
        """
        st.session_state.messages.append({
            "role": "assistant",
            "content": code_prompt
        })
        client = openai.Client()

        response = client.chat.completions.create(
            model=model,
            messages=st.session_state.messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )

        code = extract_python_code(response.choices[0].message.content)

        if code is None:
            st.warning(
                "Couldn't find data to plot in the chat. "
                "Check if the number of tokens is too low for the data at hand. "
                "I.e. if the generated code is cut off, this might be the case.",
                icon="🚨"
            )
            return "Couldn't plot the data"
        else:
            code = code.replace("fig.show()", "")
            code += """st.plotly_chart(fig, theme='streamlit', use_container_width=True)"""  # noqa: E501
            st.write(f"\n{code}")
            exec(code)
            return response.choices[0].message.content

    else:
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            return_intermediate_steps=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors= True,
            allow_dangerous_code=True  # Add this line
        )

        try:
            answer = pandas_df_agent(st.session_state.messages)
            if answer["intermediate_steps"]:
                action = answer["intermediate_steps"][-1][0].tool_input["query"]
                st.write(f"Executed the code\n{action}\n")
            return answer["output"]
        except OutputParserException:
            error_msg = """OutputParserException error occurred in LangChain agent.
                Refine your query."""
            return error_msg
        except Exception as e:  # ✅ Corrected Exception Handling
            error_msg = f"Unknown error occurred in LangChain agent: {str(e)}"
            return error_msg  # ✅ Now 'error_msg' is always defined