# ImageCaption
Image Captioning using Captioning Transformer
# Image Captioning using Vision Transformer (ViT) and CPTR

## Overview
This project implements **Image Captioning** using a **Vision Transformer (ViT) Encoder** and a **Transformer-based Decoder (CPTR)**. The model takes an image as input and generates a meaningful caption describing the image.

## Dataset
We use an image dataset stored in a ZIP file with corresponding captions in a `captions.txt` file.

### **Dataset Structure**
```
📂 /content/dataset
   📄 captions.txt
📂 /content/dataset/Images
   📄 649596742_5ba84ce946.jpg
   📄 854333409_38bc1da9dc.jpg
   📄 1087168168_70280d024a.jpg
   📄 3028969146_26929ae0e8.jpg
   📄 3385956569_a849218e34.jpg
```

## Model Architecture
1. **ViT Encoder**: Extracts deep features from images using Vision Transformer (ViT).
2. **Transformer Decoder (CPTR)**: Generates captions based on extracted features.

### **Pretrained Models Used**
- **ViT (google/vit-base-patch16-224)**: For feature extraction.
- **GPT-2 Decoder (gpt2)**: As a transformer-based language model.

## Installation & Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/image-captioning-vit.git
cd image-captioning-vit

# Install dependencies
pip install torch torchvision transformers pillow
```

## Usage
### **Data Preparation**
- Upload the dataset ZIP file to Google Drive.
- Extract it into `/content/dataset`.

### **Train the Model**
```python
python train.py
```

### **Generate Captions**
```python
python generate_caption.py --image_path path/to/image.jpg
```

## Training Details
- **Optimizer**: Adam (`lr=1e-4`)
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 5 (adjust based on dataset size)
- **Batch Size**: 4

## Results
Once trained, the model generates meaningful captions for images. Example:
```
Image: dog_running.jpg
Generated Caption: "A dog is running through a grassy field."
```

## Issues & Improvements
- 🔹 Fine-tune the ViT model on a larger dataset.
- 🔹 Use a more sophisticated decoder like BART or T5.
- 🔹 Implement beam search or top-k sampling for better captions.

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a Pull Request.

## License
This project is licensed under the MIT License.

## Acknowledgements
- Hugging Face Transformers
- PyTorch Community
- Original CPTR Paper

## Contact
For any questions, feel free to reach out:
- **Email**: sajcev101@gmail.com


