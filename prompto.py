from flask import Flask, render_template, request
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use when GPU is available

model_name = "facebook/blenderbot-1B-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name).to(device)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_input = request.args.get('msg')
    
    inputs = tokenizer([user_input], return_tensors="pt").to(device)
    
    reply_ids = model.generate(**inputs)
    response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    
    return response

if __name__ == "__main__":
    app.run(debug=True)
