import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import numpy as np
import io
from skimage.measure import label, regionprops

# Load the Excel file into a DataFrame
file_path = "Synesthesia (6).xlsx"  # Ensure this file is in the same directory in your repo
df = pd.read_excel(file_path)

# Column names in the Excel file
letter_column = 'Letter'
hex_column = 'HEX'

# Create the color dictionary
color_dict = pd.Series(df[hex_column].values, index=df[letter_column].astype(str).str.upper()).to_dict()
# Create reverse color dictionary for decoding
reverse_color_dict = {v: k for k, v in color_dict.items()}

# Define a function for checking color tolerance
def is_color_similar(color1, color2, tolerance=30):
    return all(abs(c1 - c2) <= tolerance for c1, c2 in zip(color1, color2))

# Streamlit interface
st.title("Text to Marcus Language")
text = st.text_area("Enter text:", height=200)  # Use text_area for multi-line input

# Generate Image Section
if st.button("Generate Image"):
    # Parameters
    block_size = 150  # Diameter of each color circle
    spacing = 10  # Space between circles
    max_chars_per_line = 40  # Maximum characters per line
    line_height = block_size + spacing  # Space required for each line

    # Wrap text into lines based on max_chars_per_line
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        # Check if adding the next word would exceed the line length
        if len(" ".join(current_line + [word])) <= max_chars_per_line:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]  # Start a new line with the current word

    # Add the last line if there's any remaining text
    if current_line:
        lines.append(" ".join(current_line))

    num_lines = len(lines)

    # Calculate image dimensions based on the number of lines and actual content
    max_line_length = max(len(line) for line in lines)
    width = (block_size + spacing) * max_line_length - spacing  # Adjust width to actual content
    height = line_height * num_lines - spacing  # Adjust height to actual content

    # Create a blank image
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # Load a font for rendering text
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Update with path to a font file if needed
    except IOError:
        font = ImageFont.load_default()

    # Draw each character on the image
    y_offset = 0
    for line in lines:
        x_offset = 0
        for char in line:
            if char.isalnum():  # Check if character is alphanumeric
                hex_color = color_dict.get(char.upper(), "#FFFFFF")
                if len(hex_color) == 7 and hex_color.startswith("#"):
                    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))  # Adjust index for '#'
                else:
                    rgb_color = (255, 255, 255)  # Default to white if hex_color is invalid

                draw.ellipse(
                    [x_offset, y_offset, x_offset + block_size, y_offset + block_size],
                    fill=rgb_color
                )
            else:
                # Draw punctuation as regular black text centered in the block
                bbox = draw.textbbox((x_offset, y_offset), char, font=font)
                text_x = x_offset + (block_size - (bbox[2] - bbox[0])) // 2
                text_y = y_offset + (block_size - (bbox[3] - bbox[1])) // 2
                draw.text((text_x, text_y), char, fill="black", font=font)

            x_offset += block_size + spacing  # Move to the next circle position
        y_offset += line_height  # Move down for the next line

    # Display the generated image
    st.image(image, caption='Generated Synesthetic Image')

    # Create a BytesIO object to save the image in memory
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_buffer.seek(0)  # Move to the beginning of the BytesIO buffer

    # Add a download button
    st.download_button(
        label="Download Image",
        data=img_buffer,
        file_name="output_image.png",
        mime="image/png"
    )

# Image Upload Section
uploaded_file = st.file_uploader("Upload an image with colors", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    st.image(image, caption='Uploaded Image')

    # Convert the image to a NumPy array for processing
    image_array = np.array(image)
    height, width, _ = image_array.shape

    # Create a mask for colored areas
    mask = np.zeros((height, width), dtype=np.uint8)

    # Detect blobs by applying a threshold based on color similarity
    for y in range(height):
        for x in range(width):
            r, g, b = image_array[y, x]
            hex_color = f'#{r:02x}{g:02x}{b:02x}'  # Convert to hex format

            # Check for matches within the color tolerance
            for hex_key, letter in reverse_color_dict.items():
                if is_color_similar(
                    (r, g, b), 
                    tuple(int(hex_key[i:i+2], 16) for i in (1, 3, 5)), 
                    tolerance=30
                ):
                    mask[y, x] = 1  # Mark this pixel as part of a blob
                    break

    # Label connected regions in the mask
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)

    detected_text = []

    # Process each region to determine the detected text
    for region in regions:
        if region.area > 10:  # Consider only large enough regions
            # Get the centroid of the region
            y, x = region.centroid

            # Get the color of the pixel at the centroid
            centroid_color = image_array[int(y), int(x)]
            hex_color = f'#{centroid_color[0]:02x}{centroid_color[1]:02x}{centroid_color[2]:02x}'

            # Map the color back to the corresponding letter
            matched_letter = None
            for hex_key, letter in reverse_color_dict.items():
                if is_color_similar(
                    centroid_color, 
                    tuple(int(hex_key[i:i+2], 16) for i in (1, 3, 5)), 
                    tolerance=30
                ):
                    matched_letter = letter
                    break

            if matched_letter:
                detected_text.append(matched_letter)

    # Generate detected text ensuring letters are not repeated
    output_text = ''.join(detected_text)

    # Display the detected text
    st.subheader("Detected Text:")
    st.write(output_text)
