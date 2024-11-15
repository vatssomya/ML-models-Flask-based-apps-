import pickle

def load_model():
    # Load the model and scaler
    with open('best_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open('data_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    return model, scaler
