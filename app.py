import random
import json

import torch

from model import NeuralNet
from helpers import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./datasets/intents.json', 'r') as json_data:
    intents = json.load(json_data)

with open('./datasets/products.json', 'r', encoding="utf-8") as json_data:
    products = json.load(json_data)

FILE = "./models/intent.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words_intents = data['all_words']
intents_tags = data['tags']
model_state = data["model_state"]

intent_model = NeuralNet(input_size, hidden_size, output_size).to(device)
intent_model.load_state_dict(model_state)
intent_model.eval()

FILE = "./models/product.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words_products_titles = data['all_words']
products_tags = data['tags']
model_state = data["model_state"]

product_model = NeuralNet(input_size, hidden_size, output_size).to(device)
product_model.load_state_dict(model_state)
product_model.eval()

bot_name = "AI"
print("Let's chat! (type 'quit' to exit)")


def predict_product(tokenized_sentence):
    X = bag_of_words(tokenized_sentence, all_words_products_titles)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = product_model(X)
    _, predicted = torch.max(output, dim=1)

    tag = products_tags[predicted.item()]
    # print(tag)

    probs = torch.softmax(output, dim=1)
    # print(probs)
    prob = probs[0][predicted.item()]
    # print("p:", tag, prob.item())
    if prob.item() > 0.7:
        return tag
    raise "Product not found"


while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words_intents)
    X = X.reshape(1, X.shape[0])
    # print(X)
    # print(all_words)
    X = torch.from_numpy(X).to(device)

    output = intent_model(X)
    _, predicted = torch.max(output, dim=1)

    tag = intents_tags[predicted.item()]
    # print(tag)

    probs = torch.softmax(output, dim=1)
    # print(probs)
    prob = probs[0][predicted.item()]
    # print(prob.item())
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["intent"]:
                if tag == "ProductQuery":
                    try:
                        product_title = predict_product(sentence)
                        product = next(
                            (product for product in products if product['title'] == product_title), None)
                        if not product:
                            raise "Product Not Found"
                        if product['on_sale']:
                            message = intent['responses'][1].replace(
                                "<PRICE>", str(product['price'])
                            )
                            print(f"{bot_name}: {message}")
                        else:
                            message = intent['responses'][0].replace(
                                "<PRICE>", str(product['price'])
                            ).replace(
                                "<PRODUCT>", product['title']
                            )
                            print(f"{bot_name}: {message}")
                    except:
                        print(f"{bot_name}: {intent['responses'][-1]}")
                else:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
