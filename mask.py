import sys
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use model to process input
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Generate predictions
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    visualize_attentions(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
        result.attentions,
    )


def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with id `mask_token_id` in `inputs`,
    or `None` if not present.
    """
    input_ids = inputs["input_ids"][0]
    for index, token_id in enumerate(input_ids):
        if int(token_id) == mask_token_id:
            return index
    return None


def get_color_for_attention_score(attention_score):
    """
    Return a grayscale RGB tuple (r, g, b) for an attention score in [0, 1].
    """
    shade = round(float(attention_score) * 255)
    shade = max(0, min(255, shade))
    return (shade, shade, shade)


def visualize_attentions(tokens, attentions):
    """
    Generate attention diagrams for every layer and attention head.
    """
    num_layers = len(attentions)
    for layer_idx in range(num_layers):
        num_heads = len(attentions[layer_idx][0])
        for head_idx in range(num_heads):
            generate_diagram(
                layer_idx + 1,
                head_idx + 1,
                tokens,
                attentions[layer_idx][0][head_idx],
            )


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram for a particular layer and head, given the tokens
    and attention weights.
    """
    # Create new image
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw each token onto the image
    for i, token in enumerate(tokens):
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT,
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, (0, 0), token_image)

        _, _, width, _ = draw.textbbox(
            (0, 0), token, font=FONT
        )
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT,
        )

    # Draw each word's attention weights
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save image
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()