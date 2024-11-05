import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import io

# Load the Excel file into a DataFrame
file_path = "Synesthesia (6).xlsx"  # Ensure this file is in the same directory in your repo
df = pd.read_excel(file_path)

# Column names in the Excel file
letter_column = 'Letter'
hex_column = 'HEX'

# Create the color dictionary
color_dict = pd.Series(df[hex_column].values, index=df[letter_column].astype(str).str.upper()).to_dict()

# Streamlit interface
st.title("Text to Marcus Language")
text = st.text_area("Enter text:", height=200)  # Use text_area for multi-line input

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

    # Calculate image dimensions based on the number of lines
    width = (block_size + spacing) * max_chars_per_line
    height = line_height * num_lines

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
