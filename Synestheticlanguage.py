import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import numpy as np
import io

# Load the Excel file into a DataFrame
file_path = "Synesthesia (6).xlsx"  # Ensure this file is in the same directory in your repo
df = pd.read_excel(file_path)

# Column names in the Excel file
letter_column = 'Letter'
hex_column = 'HEX'

# Create the color dictionary, converting hex codes to lowercase for consistency
color_dict = pd.Series(df[hex_column].str.lower().values, index=df[letter_column].astype(str).str.upper()).to_dict()
reverse_color_dict = {v: k for k, v in color_dict.items()}

# Streamlit interface
st.title("Text to Marcus Language")
text = st.text_area("Enter text:", height=200)

# Generate Image Section
if st.button("Generate Image"):
    block_size = 150
    spacing = 20
    max_chars_per_line = 40
    line_height = block_size + spacing

    # Wrap text into lines
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        if len(" ".join(current_line + [word])) <= max_chars_per_line:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    num_lines = len(lines)
    max_line_length = max(len(line) for line in lines)
    width = (block_size + spacing) * max_line_length - spacing
    height = line_height * num_lines - spacing

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    y_offset = 0
    for line in lines:
        x_offset = 0
        for char in line:
            if char.isalnum():
                hex_color = color_dict.get(char.upper(), "#FFFFFF").lower()  # Ensure lowercase
                rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
                draw.ellipse(
                    [x_offset, y_offset, x_offset + block_size, y_offset + block_size],
                    fill=rgb_color
                )
            else:
                bbox = draw.textbbox((x_offset, y_offset), char, font=font)
                text_x = x_offset + (block_size - (bbox[2] - bbox[0])) // 2
                text_y = y_offset + (block_size - (bbox[3] - bbox[1])) // 2
                draw.text((text_x, text_y), char, fill="black", font=font)

            x_offset += block_size + spacing
        y_offset += line_height

    st.image(image, caption='Generated Synesthetic Image')

    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_buffer.seek(0)

    st.download_button(
        label="Download Image",
        data=img_buffer,
        file_name="output_image.png",
        mime="image/png"
    )

# Image Upload Section
uploaded_file = st.file_uploader("Upload an image with colors", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    st.image(image, caption='Uploaded Image')

    image_array = np.array(image)
    height, width, _ = image_array.shape

    block_size = 150
    spacing = 10
    detected_text = []
    y_offset = 0

    debug_output = []

    while y_offset + block_size <= height:
        x_offset = 0
        line_text = []
        while x_offset + block_size <= width:
            block = image_array[y_offset:y_offset + block_size, x_offset:x_offset + block_size]
            avg_color = block.reshape(-1, 3).mean(axis=0).astype(int)
            hex_color = f'#{avg_color[0]:02x}{avg_color[1]:02x}{avg_color[2]:02x}'.lower()  # Convert to lowercase

            matched_char = reverse_color_dict.get(hex_color, None)
            line_text.append(matched_char if matched_char else "?")

            debug_output.append(f"Detected color: {hex_color}, Matched char: {matched_char}")

            x_offset += block_size + spacing
        detected_text.append("".join(line_text))
        y_offset += block_size + spacing

    st.subheader("Debug Output for Detected Colors and Matches:")
    st.write("\n".join(debug_output))

    output_text = '\n'.join(detected_text)

    st.subheader("Detected Text:")
    st.write(output_text)
