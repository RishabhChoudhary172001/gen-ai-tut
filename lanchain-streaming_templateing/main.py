from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain

# Define the template
template = """
You are a customer support agent.
Answer the customer's question based only on the information provided.

The name of the customer is {customer}.
Greet the customer.

Question: {question}
"""

prompt = PromptTemplate(
    input_variables=["customer", "question"],
    template=template
)

# Initialize Groq LLM with streaming enabled
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

# Create an LLMChain
chain = LLMChain(prompt=prompt, llm=llm)

# Invoke the chain with input
response = chain.invoke({
    "customer": "Rishabh",
    "question": "What is the return policy? give me about 500 words response"
})

print("\n\nFinal response:")
print(response['text'])  # LLMChain returns a dict with 'text' key
