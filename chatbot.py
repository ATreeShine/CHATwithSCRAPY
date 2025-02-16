import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

GREETINGS = ["hi", "hello", "hey", "greetings"]
FEELINGS_INQUIRIES = ["how are you", "how are you doing", "how's it going", "what's up"]

class Chatbot:
    def __init__(self, model_dir='./trained_model'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model.eval()  # Set model to evaluation mode

    def generate_response(self, prompt, max_length=150):
        """
        Generate a response using the fine-tuned model with sampling for diversity.
        """
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,       # Enable sampling for more diverse outputs
            top_k=50,             # Use top-k sampling
            top_p=0.95            # Use nucleus sampling
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def predefined_response(self, prompt):
        """
        Returns predefined responses for simple greetings and feelings.
        """
        lower_prompt = prompt.lower()
        if any(greet == lower_prompt.strip() for greet in GREETINGS):
            return "Hello! How can I help you today?"
        if any(inquiry in lower_prompt for inquiry in FEELINGS_INQUIRIES):
            return "I'm doing well, thank you for asking! How can I assist you?"
        # Add more custom interactions if needed.
        return None

    def chat(self, prompt):
        """
        Processes user input and returns an appropriate chatbot response.
        Uses predefined responses for common interactions; otherwise, falls back to model generation.
        """
        # Check for predefined responses first.
        response = self.predefined_response(prompt)
        if response:
            return response
        # Otherwise, generate a response with the model.
        return self.generate_response(prompt)

if __name__ == "__main__":
    bot = Chatbot()
    print("Chatbot is ready! You can say things like 'hi', ask about feelings, or ask questions. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        answer = bot.chat(user_input)
        print("Bot:", answer)
