#############################################################
#
#                 BASIC AI AGENT WITH MEMORY
#
#############################################################



from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Load AI Model
llm = OllamaLLM(model="mistral") # Change model to "llama3" or another if needed

# Initialize Memory
chat_history = ChatMessageHistory() # Stores user-AI conversation history

# Define AI Chat Prompt
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous conversation: {chat_history}\nUser: {question}\nAI:"
)

# Function to run AI Chat with memory
def run_chain(question):
    # Retrieve chat history manually
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages])

    # Run the AI response generation
    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))

    # Store new user input and AI response in memory
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)

    return response


# Interactive CLI chatbot
print("\n AI Chatbot with memory")
print("Type 'exit' to stop.")

while True:
    user_input = input("\n You:")
    if user_input.lower() == "exit":
        print("\n Goodbye!")
        break

    ai_response = run_chain(user_input)
    print(f"\n AI: {ai_response}")