import chainlit as cl
import os
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN']
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.6, "max_new_tokens":2000})

template = """
You are an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

{question}

"""

@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain synchronously in a different thread
    res = await cl.make_async(llm_chain)(
        message, callbacks=[cl.LangchainCallbackHandler()]
    )

    # Do any post processing here

    # Send the response
    await cl.Message(content=res["text"]).send()
    return llm_chain