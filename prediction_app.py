import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import time

# Load the YOLO model
model = YOLO('model.pt')  # Update with your model file name

# Define tumor classes
TUMOR_CLASSES = {'Glioma', 'Meningioma', 'Pituitary'}

def process_image(image_bytes):
    # Convert BytesIO object to PIL Image
    image = Image.open(image_bytes)
    
    # Convert PIL image to numpy array
    img_array = np.array(image)

    # Perform prediction
    results = model.predict(img_array)

    # Draw bounding boxes on the image
    result_image_pil = None
    output_data = []
    tumor_detected = False

    for result in results:
        # Get the annotated image
        result_image = result.plot()
        result_image_pil = Image.fromarray(result_image)
        
        # Collect data about detected objects
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = result.names[class_id]
            
            if class_name in TUMOR_CLASSES:
                tumor_detected = True
            
            output_data.append(class_name)
    
    # Determine result status
    result_status = "Failed" if tumor_detected else "Success"
    
    return result_image_pil, output_data, result_status

def pil_image_to_bytes(image):
    """Convert PIL Image to byte stream."""
    byte_stream = io.BytesIO()
    image.save(byte_stream, format='JPEG')
    byte_stream.seek(0)
    return byte_stream

# Streamlit app
def main():
    # Sidebar with author info and project description
    st.sidebar.header("About the Author")
    st.sidebar.write("**Name:** Dishant Bothra")
    st.sidebar.write("**GitHub:** [Your GitHub Profile](https://github.com/your-profile)")
    
    st.sidebar.header("About the Project")
    st.sidebar.write("This is a self-made project showcasing the application of computer vision in the domain of healthcare. "
                     "The results should not be considered as true. It is designed to demonstrate the use of YOLO "
                     "for object detection in MRI scans for educational purposes.")
    st.sidebar.write("Check out how this project is made on my GitHub repository:")
    st.sidebar.write("[GitHub Repository](https://github.com/your-profile/your-repo)")
    
    st.title("Brain Tumor Prediction")

    # File uploader widget for multiple files
    uploaded_files = st.file_uploader("Choose MRI scan images...", type="jpg", accept_multiple_files=True)

    if uploaded_files:
        # Create tabs for each uploaded file
        tabs = st.tabs([f"Image {i+1}" for i in range(len(uploaded_files))])
        
        for i, uploaded_file in enumerate(uploaded_files):
            with tabs[i]:
                # Display the uploaded image
                st.image(uploaded_file, caption=f'Uploaded Image: {uploaded_file.name}', width=300)  # Adjust width as necessary
                
                # Display a progress bar and text
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text(f"Processing {uploaded_file.name}...")

                # Simulate a delay for processing
                for percent_complete in range(100):
                    time.sleep(0.02)  # Adjust the sleep time as necessary
                    progress_bar.progress(percent_complete + 1)
                
                # Process the image
                result_image, output_data, result_status = process_image(uploaded_file)
                
                # Update progress bar to 100%
                progress_bar.progress(100)
                status_text.text(f"Prediction complete for {uploaded_file.name}!")

                # Display the result image
                st.image(result_image, caption=f'Processed Image: {uploaded_file.name}', width=300)  # Adjust width as necessary

                # Display output data
                st.info(f"Output Data for {uploaded_file.name}: {output_data}")
                
                # Display result status
                if result_status == "Failed":
                    st.error("Prediction Result: Tumor detected")
                else:
                    st.success("Prediction Result: No tumor detected")

                # Provide a download button for the processed image
                result_image_bytes = pil_image_to_bytes(result_image)
                st.download_button(
                    label="Download Processed Image",
                    data=result_image_bytes,
                    file_name=f"processed_{uploaded_file.name}",
                    mime="image/jpeg"
                )

if __name__ == "__main__":
    main()
