from transformers import pipeline
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = 0 if torch.cuda.is_available() else -1
    print("Active Device:{}".format(device))
    model = pipeline('zero-shot-classification', model='facebook/bart-large-mnli', device=device)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    inputs = model_inputs.get('inputs', None)
    labels = model_inputs.get('labels', None)
    if inputs == None:
        return {'message': "No inputs provided"}
    if labels == None:
        return {'message': "No labels provided"}
    
    # Run the model
    result = model(inputs, labels)

    # Return the results as a dictionary
    return result
