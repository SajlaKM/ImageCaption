import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import ViTModel, GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image
import zipfile
import os
import glob
from torch.utils.data import Dataset, DataLoader
from google.colab import drive


# Update dataset path to point to the correct subdirectory
IMAGE_DIR = os.path.join(extract_path, "Images")  # Updated path

# Check extracted files and subdirectories
print("Extracted directory structure:")
for root, dirs, files in os.walk(extract_path):
    print(f"📂 {root}")
    for file in files[:5]:  # Show first 5 files in each folder
        print(f"   📄 {file}")

# Ensure the directory has images
if not os.path.exists(IMAGE_DIR) or len(os.listdir(IMAGE_DIR)) == 0:
    raise ValueError(f"No images found in {IMAGE_DIR}. Check your dataset extraction.")

# Fine-tuned Vision Transformer (ViT) for feature extraction
class FineTunedViT(nn.Module):
    def __init__(self, vit_model_name="google/vit-base-patch16-224"):
        super(FineTunedViT, self).__init__()
        self.vit = ViTModel.from_pretrained(vit_model_name)

        # Remove the uninitialized pooling layer
        self.vit.pooler = None

        # Add a custom classifier for fine-tuning
        self.fc = nn.Linear(self.vit.config.hidden_size, 256)

    def forward(self, x):
        features = self.vit(x).last_hidden_state[:, 0, :]
        return self.fc(features)

# CPTR Model: Fine-tuned ViT as encoder, Transformer decoder
class CPTR(nn.Module):
    def __init__(self, vit_model_name="google/vit-base-patch16-224", gpt2_model_name="gpt2"):
        super(CPTR, self).__init__()
        self.vit = FineTunedViT(vit_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        self.decoder = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

        # Linear layer to match encoder-decoder dimensions
        self.linear = nn.Linear(256, self.decoder.config.n_embd)

    def forward(self, image, captions):
        vit_features = self.vit(image)
        vit_features = self.linear(vit_features).unsqueeze(1)

        caption_inputs = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
        input_ids = caption_inputs.input_ids

        outputs = self.decoder(input_ids, encoder_hidden_states=vit_features)
        return outputs.logits

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define Custom Dataset for Image Captioning
class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))

        if len(self.image_paths) == 0:
            print("No images found! Checking available file types...")
            print("Files:", os.listdir(image_dir))  # Debugging: Show all files
            raise ValueError("No image files found in the dataset directory!")

        print(f"Found {len(self.image_paths)} images.")  # Print found image count
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, image_path  # Labels (captions) are not provided here

# Create DataLoader
dataset = ImageCaptionDataset(IMAGE_DIR, transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load CPTR model with fine-tuned ViT
model = CPTR()

# Fine-tune the ViT model
optimizer = torch.optim.Adam(model.vit.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# Training Loop (Now Using Defined DataLoader)
for epoch in range(5):  # Adjust based on dataset size
    for images, image_paths in dataloader:
        optimizer.zero_grad()
        outputs = model.vit(images)
        loss = loss_fn(outputs, torch.zeros_like(outputs))  # Dummy loss as captions are not available
        loss.backward()
        optimizer.step()

# Load and preprocess image
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Generate caption
def generate_caption(model, image_path, max_length=20):
    model.eval()
    image = process_image(image_path)

    with torch.no_grad():
        vit_features = model.vit(image)
        vit_features = model.linear(vit_features).unsqueeze(1)

        generated_ids = model.decoder.generate(
            encoder_hidden_states=vit_features,
            max_length=max_length,
            pad_token_id=model.tokenizer.eos_token_id
        )
        caption = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return caption

# Generate captions for images in the dataset
dataset_images = [os.path.join(IMAGE_DIR, img) for img in os.listdir(IMAGE_DIR) if img.endswith(('.jpg', '.png'))]
for img_path in dataset_images[:5]:  # Show captions for first 5 images
    caption = generate_caption(model, img_path)
    print(f"Image: {img_path} \nGenerated Caption: {caption}\n")
