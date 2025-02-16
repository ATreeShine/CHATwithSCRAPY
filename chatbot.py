import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class Chatbot:
    def __init__(self, model_dir='./trained_model'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model.eval()

    def chat(self, prompt, max_length=150):
        """
        Generates a response using the fine-tuned model based on the given prompt.
        """
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,        
            top_k=50,              
            top_p=0.95             
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

if __name__ == "__main__":
    bot = Chatbot()
    print("Chatbot is ready. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        answer = bot.chat(user_input)
        print("Bot:", answer)
