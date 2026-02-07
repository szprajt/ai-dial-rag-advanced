import os
from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


SYSTEM_PROMPT = """
You are a helpful AI assistant powered by RAG (Retrieval-Augmented Generation).
Your goal is to assist users with questions about a microwave manual.
You will be provided with relevant context from the manual and the user's question.
You must answer the user's question based ONLY on the provided context.
If the answer is not found in the context, or if the question is unrelated to the microwave manual, 
you should politely inform the user that you cannot answer the question based on the available information.
Do not make up information.
"""

USER_PROMPT = """
Context:
{context}

User Question:
{question}

Please answer the user question using the context above.
"""

def main():
    # Configuration
    embedding_model = 'text-embedding-3-small-1'
    chat_model = 'gpt-4o' # Assuming a standard chat model name, adjust if needed based on available deployments
    
    db_config = {
        'host': 'localhost',
        'port': 5433,
        'database': 'vectordb',
        'user': 'postgres',
        'password': 'postgres'
    }

    # Initialize clients
    if not API_KEY:
        print("Error: DIAL_API_KEY is not set in _constants.py or environment variables.")
        return

    embeddings_client = DialEmbeddingsClient(deployment_name=embedding_model, api_key=API_KEY)
    chat_client = DialChatCompletionClient(deployment_name=chat_model, api_key=API_KEY)
    text_processor = TextProcessor(embeddings_client=embeddings_client, db_config=db_config)

    # Process document (optional: check if already processed or force update)
    # For this task, we'll process it once at startup or if requested. 
    # Let's assume we process it if the DB is empty or just process it to be safe as per instructions.
    # The instructions say "Implement document processing workflow".
    
    manual_path = os.path.join(os.path.dirname(__file__), 'embeddings', 'microwave_manual.txt')
    print("Processing document...")
    try:
        text_processor.process_text_file(
            file_path=manual_path,
            chunk_size=300,
            overlap=40,
            truncate=True # Truncate to ensure clean state
        )
        print("Document processed and embeddings stored.")
    except Exception as e:
        print(f"Error processing document: {e}")
        return

    # Chat loop
    conversation = Conversation()
    # Add system message
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))

    print("\n--- Microwave Manual Assistant ---")
    print("Ask me anything about the microwave manual. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ('exit', 'quit'):
            break
        
        if not user_input:
            continue

        # 1. Retrieval
        try:
            search_results = text_processor.search(
                query=user_input,
                mode=SearchMode.COSINE_DISTANCE,
                top_k=5,
                min_score=0.5
            )
        except Exception as e:
            print(f"Error during retrieval: {e}")
            continue

        # 2. Augmentation
        context_text = ""
        if search_results:
            context_text = "\n\n".join([res['text'] for res in search_results])
        else:
            context_text = "No relevant context found."

        augmented_prompt = USER_PROMPT.format(context=context_text, question=user_input)
        
        # We don't add the full augmented prompt to conversation history as 'User' message usually,
        # but for RAG often we send the augmented prompt as the last message.
        # However, to keep history clean, we might want to store the original user message in history,
        # but send the augmented one to LLM.
        # Let's create a temporary list of messages for the API call.
        
        current_messages = conversation.get_messages().copy()
        current_messages.append(Message(Role.USER, augmented_prompt))

        # 3. Generation
        try:
            response_message = chat_client.get_completion(current_messages)
            print(f"AI: {response_message.content}\n")
            
            # Update conversation history
            conversation.add_message(Message(Role.USER, user_input)) # Store original input
            conversation.add_message(response_message)
            
        except Exception as e:
            print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
