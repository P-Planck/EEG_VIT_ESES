from PIL import Image

# Open the image file
image = Image.open("EEG_data/EEG_IMG/n/1_image_26.jpg")

# Resize the image to 224x224
resized_image = image.resize((224, 224))

# Save or display the resized image
resized_image.save("Test/resized_image_n.jpg")
resized_image.show()