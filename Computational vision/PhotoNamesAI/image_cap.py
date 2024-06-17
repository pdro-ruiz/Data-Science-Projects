import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def main():
    """
    Main function to generate automatic captions for images using the BLIP model from Hugging Face.
    
    The script loads an image, processes it, and uses the BLIP model to generate a textual description of the image.
    """
    
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")                  # Load the BLIP processor and model
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        img_path = "PhotoNamesAI.jpg" # <-------------------------------------------------------------------------- Change the path to the image you want to generate a caption for
        image = Image.open(img_path).convert('RGB')                                                         # Load the image and convert it to RGB format

        inputs = processor(images=image, return_tensors="pt")                                               # Process the image to generate appropriate inputs for the model
        #print("Image processed successfully.")

        outputs = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)                 # Generate the caption for the image using the BLIP model
        #print("Caption generated successfully.")

        caption = processor.decode(outputs[0], skip_special_tokens=True)                                    # Decode the generated tokens to readable text
        #print("Caption decoded successfully.")

        print(f"The caption for this image is: {caption}")

    except Exception as e:
        print(f"An error occurred: {e}")                                                                    # Catch and display any errors that occur during the process

if __name__ == "__main__":
    main()
