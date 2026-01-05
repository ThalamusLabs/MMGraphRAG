from PIL import Image
import io
import matplotlib.pyplot as plt
import base64

def load_data():
    with open('bus.jpg', 'rb') as f:
        image_bytes = f.read()

    return base64.b64encode(image_bytes).decode('utf-8')

def chunker(base64_image: str) -> list[bytes]:
    image_bytes = base64.b64decode(base64_image)
    # Decode the image
    img = Image.open(io.BytesIO(image_bytes))

    width, height = img.size
    half_width = width // 2
    half_height = height // 2

    # Create 4 chunks
    chunk1 = img.crop((0, 0, half_width, half_height))  # Top-left
    chunk2 = img.crop((half_width, 0, width, half_height))  # Top-right
    chunk3 = img.crop((0, half_height, half_width, height))  # Bottom-left
    chunk4 = img.crop((half_width, half_height, width, height))  

    result = []
    for chunk in [chunk1, chunk2, chunk3, chunk4]:
        # Convert chunk to bytes
        chunk_bytes = io.BytesIO()
        chunk.save(chunk_bytes, format='JPEG')
        chunk_bytes = chunk_bytes.getvalue()
        
        result.append(base64.b64encode(chunk_bytes).decode('utf-8'))

    return result



def main():
    img = load_data()
    result = chunker(img)
    print(result)


if __name__ == "__main__":
    main()