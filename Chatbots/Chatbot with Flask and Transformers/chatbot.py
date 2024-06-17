# utf-8

"""
chatbot.py

Precursor script for implementing a chatbot through console conversation,
before implementing the graphical interface.

Implements a chatbot using the 'facebook/blenderbot-400M-distill' model from Facebook AI.
Allows interaction with a chatbot through the console, and manages the conversation history to
improve coherence in the dialogue.

The script consists of:
   1. Imports: Import the necessary libraries and load both the model and the tokenizer from the 
      transformers library.
   2. History: Create a list to store the conversation history.
   3. Chat Management: Define the 'chat_with_bot' function that handles interaction with the user, including
      input tokenization, response generation, and history updating.
   4. Execution: Call the 'chat_with_bot' function to start the conversation.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer
model_name = "facebook/blenderbot-400M-distill"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the conversation history
conversation_history = []

# Function to manage the conversation
def chat_with_bot():
   """
   Allows interaction with the chatbot through the console.
   - The conversation continues until the user types 'exit'.
   - User inputs and chatbot responses are stored in the conversation history.
   """
   print("You can start chatting with the chatbot (type 'exit' to end the conversation).")
   while True:
      # Create the conversation history string
      history_string = "\n".join(conversation_history)
      # Get user input
      input_text = input("> ")
      if input_text.lower() == 'exit':
         print("Chatbot: Goodbye!")
         break
         
      # Tokenize the input text and history
      inputs = tokenizer.encode_plus(history_string + "\n" + input_text, return_tensors="pt")
      
      # Generate the model response
      outputs = model.generate(**inputs, max_new_tokens=50)
      
      # Decode the response
      response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
      print(response)
      
      # Add the interaction to the conversation history
      conversation_history.append(input_text)
      conversation_history.append(response)

if __name__ == "__main__":
   chat_with_bot()