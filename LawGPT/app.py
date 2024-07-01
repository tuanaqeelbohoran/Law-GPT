import gradio as gr
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Function to load the tokenizer and model from the given path
def load_model(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)  # Load the tokenizer from the pretrained model path
    model = GPT2LMHeadModel.from_pretrained(model_path)    # Load the model from the pretrained model path
    tokenizer.pad_token = tokenizer.eos_token              # Set the pad token to be the same as the end-of-sequence token
    return tokenizer, model                                # Return the tokenizer and model

# Function to generate a response from the model given a prompt
def generate_response(prompt, model, tokenizer, max_length=150, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')  # Encode the input prompt to tensor format
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)  # Create attention mask
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=len(prompt),  # Use the length of the prompt as the maximum length
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        num_beams=5,
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode the output to text
    return response  # Return the generated response

# Load the model and tokenizer
model_path = './trained_model'
tokenizer, model = load_model(model_path)

# Function to handle the chatbot conversation
def chatbot(user_input, history=None):
    if history is None:  # Initialize history if it's None
        history = []

    # Initial conversation history context
    conversation_history = "The following is a conversation with a chatbot. The chatbot is helpful, creative, clever, and very friendly.\n"
    for h in history:  # Append past conversation to the history
        conversation_history += f"Human: {h[0]}\nChatbot: {h[1]}\n"
    conversation_history += f"Human: {user_input}\nChatbot:"  # Add the current user input to the conversation history
    response = generate_response(conversation_history, model, tokenizer)  # Generate the response
    bot_response = response.split("Human:")[-1].split("Chatbot:")[-1].strip()  # Extract the bot's response
    history.append((user_input, bot_response))  # Append the current interaction to history
    return bot_response, history  # Return the bot's response and updated history

# Create the Gradio interface
iface = gr.Interface(fn=chatbot, 
                     inputs=[gr.components.Textbox(lines=7, label="Human"), gr.components.State()],
                     outputs=[gr.components.Textbox(label="Chatbot"), gr.components.State()],
                     title="Chatbot",
                     description="A chatbot based on GPT-2 model.")

# Launch the interface with sharing enabled
iface.launch(share=True)
