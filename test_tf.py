from tensorflow.keras.models import load_model

# Load the model
model = load_model('fruit_veg_model.h5')

# Print model summary to verify output shape
print(model.summary())
