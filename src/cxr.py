import os
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_hub as tf_hub
import io
import png
from PIL import Image
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download
from sklearn.decomposition import PCA  # For dimensionality reduction

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Unable to create a python object for variable")

# Helper functions

def bert_tokenize(text):
    """Tokenizes input text and returns token IDs and padding masks."""
    preprocessor = tf_hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    out = preprocessor(tf.constant([text.lower()]))
    ids = out['input_word_ids'].numpy().astype(np.int32)
    masks = out['input_mask'].numpy().astype(np.float32)
    paddings = 1.0 - masks
    end_token_idx = ids == 102
    ids[end_token_idx] = 0
    paddings[end_token_idx] = 1.0
    ids = np.expand_dims(ids, axis=1)
    paddings = np.expand_dims(paddings, axis=1)
    assert ids.shape == (1, 1, 128)
    assert paddings.shape == (1, 1, 128)
    return ids, paddings


def png_to_tfexample(image_array: np.ndarray) -> tf.train.Example:
    """Creates a tf.train.Example from a NumPy array."""
    image = image_array.astype(np.float32)
    image -= image.min()

    if image_array.dtype == np.uint8:
        pixel_array = image.astype(np.uint8)
        bitdepth = 8
    else:
        max_val = image.max()
        if max_val > 0:
            image *= 65535 / max_val
        pixel_array = image.astype(np.uint16)
        bitdepth = 16

    if pixel_array.ndim != 2:
        raise ValueError(f'Array must be 2-D. Actual dimensions: {pixel_array.ndim}')

    output = io.BytesIO()
    png.Writer(width=pixel_array.shape[1], height=pixel_array.shape[0], greyscale=True, bitdepth=bitdepth).write(output, pixel_array.tolist())
    png_bytes = output.getvalue()

    example = tf.train.Example()
    features = example.features.feature
    features['image/encoded'].bytes_list.value.append(png_bytes)
    features['image/format'].bytes_list.value.append(b'png')

    return example


def download_models():
    """Download model files from HuggingFace."""
    snapshot_download(
        repo_id="google/cxr-foundation",
        local_dir="/hpc/group/kamaleswaranlab/cxr-foundation",  
        allow_patterns=['elixr-c-v2-pooled/*', 'pax-elixr-b-text/*']
    )


def load_models():
    """Load the ELIXR-C and QFormer models."""
    # Ensure using 'serve' tag for loading saved models
    elixrc_model = tf.saved_model.load("/hpc/group/kamaleswaranlab/cxr-foundation/elixr-c-v2-pooled")
    qformer_model = tf.saved_model.load("/hpc/group/kamaleswaranlab/cxr-foundation/pax-elixr-b-text")
    return elixrc_model, qformer_model


def process_image(image_path):
    """Process an image and generate embeddings."""
    img = Image.open(image_path).convert('L')
    serialized_img_tf_example = png_to_tfexample(np.array(img)).SerializeToString()
    
    # Run ELIXR-C model
    elixrc_model, qformer_model = load_models()
    elixrc_infer = elixrc_model.signatures['serving_default']
    elixrc_output = elixrc_infer(input_example=tf.constant([serialized_img_tf_example]))
    elixrc_embedding = elixrc_output['feature_maps_0'].numpy()

    # Run QFormer model with image embeddings
    qformer_input = {
        'image_feature': elixrc_embedding.tolist(),
        'ids': np.zeros((1, 1, 128), dtype=np.int32).tolist(),
        'paddings': np.zeros((1, 1, 128), dtype=np.float32).tolist(),
    }
    qformer_output = qformer_model.signatures['serving_default'](**qformer_input)
    elixrb_embeddings = qformer_output['all_contrastive_img_emb']

    return elixrc_embedding, elixrb_embeddings


def process_text(text_query):
    """Process a text input and generate embeddings."""
    tokens, paddings = bert_tokenize(text_query)
    
    # Run QFormer model with text input
    elixrc_embedding = np.zeros([1, 8, 8, 1376], dtype=np.float32)  
    qformer_input = {
        'image_feature': elixrc_embedding.tolist(),
        'ids': tokens.tolist(),
        'paddings': paddings.tolist(),
    }
    qformer_model = tf.saved_model.load("/hpc/group/kamaleswaranlab/cxr-foundation/pax-elixr-b-text")
    qformer_output = qformer_model.signatures['serving_default'](**qformer_input)
    text_embeddings = qformer_output['contrastive_txt_emb']

    return text_embeddings


def visualize_and_save_embeddings(embedding, output_path, title='Embedding Visualization'):
    """Visualize and save the generated embeddings to a PNG file."""
    # If the embedding is high-dimensional, reduce the dimensions (e.g., using PCA)
    if len(embedding.shape) > 2:  # Assuming it's a high-dimensional tensor
        pca = PCA(n_components=2)  # Reducing to 2D for visualization
        embedding_2d = pca.fit_transform(embedding.reshape(-1, embedding.shape[-1]))  # Reshape to 2D
        embedding_2d = embedding_2d.reshape(embedding.shape[0], embedding.shape[1], 2)  # Reshape back if needed
        embedding_to_show = embedding_2d  # This will be the reduced 2D representation for visualization
    else:
        embedding_to_show = embedding

    # Plot the embedding
    plt.imshow(embedding_to_show[0], cmap='gray')
    plt.colorbar()  # Show a colorbar to understand the value distribution
    plt.title(title)

    # Save the plot to the specified output path
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save as PNG with high resolution
    print(f"Embedding visualization saved to {output_path}")
    plt.close()  # Close the plot to avoid memory issues


# Main 
def main():
    # Download models (ensure they're available locally)
    download_models()

    input_type = 'image'  # Switch between 'image' or 'text' input
    image_path = '/hpc/group/kamaleswaranlab/cxr-foundation/JPCLN012.png'  # Update this path to your image file

    if input_type == 'image':
        # Process the image and generate embeddings
        elixrc_embedding, elixrb_embeddings = process_image(image_path)
        print("ELIXR-C - interim embedding shape:", elixrc_embedding.shape)
        print("ELIXR-B - embedding shape:", elixrb_embeddings.shape)
        
        # Visualize and save the embedding
        output_path = '/hpc/home/ak817/ondemand/data/sys/myjobs/projects/default/6/elixr_b_embedding.png'  
        visualize_and_save_embeddings(elixrb_embeddings, output_path, title='ELIXR-B Image Embedding')
    
    elif input_type == 'text':
        # Process the text and generate embeddings
        text_query = "Airspace opacity"  # Example query for text input
        text_embeddings = process_text(text_query)
        print("Text Embedding shape:", text_embeddings.shape)
        print("First 5 tokens:", text_embeddings[0][0:5])


if __name__ == "__main__":
    main()
