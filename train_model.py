import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # For potential future transformations

# --- 0. הגדרת מכשיר (Device) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. טעינת נתונים (JSON) ---
def load_annotations(json_path):
    """
    טוען את ה-annotations מקובץ JSON.
    Args:
        json_path (str): נתיב לקובץ ה-JSON.
    Returns:
        dict: מילון של annotations, או מילון ריק במקרה של שגיאה.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        return annotations
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return {}

# --- 2. Dataset ו-DataLoader של PyTorch ---
class ClockDataset(Dataset):
    """
    Dataset מותאם אישית עבור תמונות השעונים והקריאות שלהן.
    """
    def __init__(self, annotations_dict, img_dir, target_size=(128, 128), transform=None):
        self.annotations_dict = annotations_dict
        self.img_dir = img_dir
        self.target_size = target_size
        self.transform = transform # For torchvision transforms if needed later
        
        self.filenames = list(annotations_dict.keys())
        self.processed_data = self._preprocess_all()

    def _preprocess_all(self):
        data = []
        for filename in self.filenames:
            details = self.annotations_dict[filename]
            try:
                img_path = os.path.join(self.img_dir, filename)
                image = cv2.imread(img_path)

                if image is None:
                    print(f"Warning: Could not read image {img_path} for dataset. Skipping.")
                    continue

                points = details['points']
                if int(points[0][0])<int(points[1][0]):
                    x_min, y_min = int(points[0][0]), int(points[0][1])
                    x_max, y_max = int(points[1][0]), int(points[1][1])
                else:
                    x_max, y_max = int(points[0][0]), int(points[0][1])
                    x_min, y_min = int(points[1][0]), int(points[1][1])
                if x_min >= x_max or y_min >= y_max:
                    print(f"Warning: Invalid bbox points {points} for {img_path}. Skipping.")
                    continue
                
                h_img, w_img = image.shape[:2]
                x_min_c, y_min_c = max(0, x_min), max(0, y_min)
                x_max_c, y_max_c = min(w_img, x_max), min(h_img, y_max)

                if x_min_c >= x_max_c or y_min_c >= y_max_c:
                    print(f"Warning: Bbox {points} invalid after clamping for {img_path}. Skipping.")
                    continue
                
                cropped_image = image[y_min_c:y_max_c, x_min_c:x_max_c]
                if cropped_image.size == 0:
                    print(f"Warning: Cropped image empty for {img_path}. Skipping.")
                    continue

                resized_image = cv2.resize(cropped_image, self.target_size)
                normalized_image = resized_image / 255.0
                
                # Convert to PyTorch tensor: HWC to CHW
                # OpenCV reads images as BGR, PyTorch models usually expect RGB.
                # If your model is sensitive to color order, consider converting BGR to RGB here:
                # normalized_image_rgb = cv2.cvtColor(normalized_image.astype(np.float32), cv2.COLOR_BGR2RGB)
                # image_tensor = torch.tensor(normalized_image_rgb, dtype=torch.float32).permute(2, 0, 1)
                # For now, keeping BGR and permuting.
                image_tensor = torch.tensor(normalized_image, dtype=torch.float32).permute(2, 0, 1)


                readings_str = details['user_input']
                readings_float = [float(r.strip()) for r in readings_str]
                label_tensor = torch.tensor(readings_float, dtype=torch.float32)
                
                data.append({'image': image_tensor, 'label': label_tensor, 'filename': filename})

            except KeyError as e:
                print(f"Warning: Missing key {e} for {filename}. Skipping.")
            except ValueError as e:
                print(f"Warning: Value error for {filename}, readings: {details.get('user_input')}. Error: {e}. Skipping.")
            except Exception as e:
                print(f"Error processing {filename} for dataset: {e}. Skipping.")
        return data

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        sample = self.processed_data[idx]
        image = sample['image']
        label = sample['label']
        
        if self.transform: # Apply transforms if any (e.g., for data augmentation)
            image = self.transform(image)
            
        return image, label, sample['filename'] # Return filename for debugging/display

# --- 3. הגדרת מודל CNN ב-PyTorch ---
class ClockCNN(nn.Module):
    def __init__(self, num_outputs=2):
        super(ClockCNN, self).__init__()
        # Input: (Batch, 3, 128, 128)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding='same'), # (Batch, 32, 128, 128)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (Batch, 32, 64, 64)

            nn.Conv2d(32, 64, kernel_size=3, padding='same'), # (Batch, 64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (Batch, 64, 32, 32)

            nn.Conv2d(64, 128, kernel_size=3, padding='same'), # (Batch, 128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (Batch, 128, 16, 16)
            
            nn.Conv2d(128, 256, kernel_size=3, padding='same'), # (Batch, 256, 16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (Batch, 256, 8, 8)
        )
        
        # Calculate the flattened size after conv layers
        # For (128,128) input, after 4 MaxPool layers (stride 2), size becomes 128 / (2^4) = 128 / 16 = 8
        # So, flattened_size = 256 * 8 * 8
        self.flattened_size = 256 * 8 * 8 
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs) # פלט רגרסיה לינארי
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# --- הגדרות עיקריות ---
JSON_FILE_PATH = 'annotations.json'
IMAGES_DIR = 'Data/' 
MODEL_SAVE_PATH = 'clock_reader_pytorch_model.pth'

TARGET_IMAGE_SIZE = (128, 128)
NUM_OUTPUTS = 2
EPOCHS = 75 # ניתן לשנות
BATCH_SIZE = 32 # ניתן לשנות
LEARNING_RATE = 0.0005 # ניתן לשנות

# --- טעינת נתונים ו-DataLoaders ---
annotations_data = load_annotations(JSON_FILE_PATH)
if not annotations_data:
    print("Exiting: No annotations loaded.")
else:
    full_dataset = ClockDataset(annotations_data, IMAGES_DIR, TARGET_IMAGE_SIZE)
    
    if len(full_dataset) == 0:
        print("Exiting: Dataset is empty after processing. Check image paths and JSON content.")
    else:
        print(f"Successfully created dataset with {len(full_dataset)} samples.")

        # חלוקה לסט אימון, ולידציה ומבחן
        # 70% אימון, 15% ולידציה, 15% מבחן
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        # Ensure there's enough data for all splits
        if train_size == 0 or (val_size == 0 and test_size == 0) : # Need at least train, and one of val/test
             print(f"Warning: Not enough data for proper train/validation/test split. Total samples: {len(full_dataset)}")
             # Adjusting to use all available data for training if splits are too small
             train_dataset = full_dataset
             val_dataset = None # Or a very small subset if possible
             test_dataset = None
             if len(full_dataset) > 1: # Minimal split if possible
                 train_dataset, temp_dataset = torch.utils.data.random_split(full_dataset, [len(full_dataset)-1, 1], generator=torch.Generator().manual_seed(42))
                 val_dataset = temp_dataset # use the single sample for validation
             else: # Only one sample, use for training, no validation/test
                 train_dataset = full_dataset

        else:
            train_dataset, temp_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size + test_size], generator=torch.Generator().manual_seed(42))
            if val_size > 0 and test_size > 0 :
                 val_dataset, test_dataset = torch.utils.data.random_split(temp_dataset, [val_size, test_size], generator=torch.Generator().manual_seed(42))
            elif val_size > 0:
                 val_dataset = temp_dataset
                 test_dataset = None
            else: # test_size > 0
                 test_dataset = temp_dataset
                 val_dataset = None


        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if val_dataset and len(val_dataset) > 0 else None
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) if test_dataset and len(test_dataset) > 0 else None

        print(f"Training samples: {len(train_dataset)}")
        if val_loader: print(f"Validation samples: {len(val_dataset)}")
        else: print("No validation samples.")
        if test_loader: print(f"Test samples: {len(test_dataset)}")
        else: print("No test samples.")


        if len(train_dataset) == 0:
             print("Exiting: Training dataset is empty after splitting. Cannot train the model.")
        else:
            # --- בניית מודל, פונקציית הפסד ואופטימייזר ---
            model = ClockCNN(num_outputs=NUM_OUTPUTS).to(device)
            print(model) # הדפסת מבנה המודל

            criterion = nn.MSELoss() # פונקציית הפסד לרגרסיה
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

            # --- לולאת אימון ---
            train_losses = []
            val_losses = []
            train_maes = []
            val_maes = []

            print("\nStarting model training...")
            for epoch in range(EPOCHS):
                model.train() # מעבר למצב אימון
                running_loss = 0.0
                running_mae = 0.0
                
                for i, (inputs, labels, _) in enumerate(train_loader): # filenames are not used in training
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad() # איפוס גרדיאנטים
                    outputs = model(inputs) # קבלת פלט המודל
                    loss = criterion(outputs, labels) # חישוב הפסד
                    loss.backward() # חישוב גרדיאנטים
                    optimizer.step() # עדכון משקולות

                    running_loss += loss.item() * inputs.size(0)
                    running_mae += torch.abs(outputs - labels).sum().item() / NUM_OUTPUTS # MAE per reading

                epoch_loss = running_loss / len(train_loader.dataset)
                epoch_mae = running_mae / len(train_loader.dataset)
                train_losses.append(epoch_loss)
                train_maes.append(epoch_mae)

                # ולידציה (אם יש סט ולידציה)
                if val_loader:
                    model.eval() # מעבר למצב הערכה
                    val_running_loss = 0.0
                    val_running_mae = 0.0
                    with torch.no_grad(): # אין צורך לחשב גרדיאנטים בולידציה
                        for inputs_val, labels_val, _ in val_loader:
                            inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                            outputs_val = model(inputs_val)
                            loss_val = criterion(outputs_val, labels_val)
                            val_running_loss += loss_val.item() * inputs_val.size(0)
                            val_running_mae += torch.abs(outputs_val - labels_val).sum().item() / NUM_OUTPUTS
                    
                    epoch_val_loss = val_running_loss / len(val_loader.dataset)
                    epoch_val_mae = val_running_mae / len(val_loader.dataset)
                    val_losses.append(epoch_val_loss)
                    val_maes.append(epoch_val_mae)
                    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_loss:.4f}, Train MAE: {epoch_mae:.4f} | Val Loss: {epoch_val_loss:.4f}, Val MAE: {epoch_val_mae:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_loss:.4f}, Train MAE: {epoch_mae:.4f}")

            print("Model training finished.")

            # --- הערכת המודל על סט המבחן ---
            if test_loader:
                model.eval()
                test_running_loss = 0.0
                test_running_mae = 0.0
                with torch.no_grad():
                    for inputs_test, labels_test, _ in test_loader:
                        inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
                        outputs_test = model(inputs_test)
                        loss_test = criterion(outputs_test, labels_test)
                        test_running_loss += loss_test.item() * inputs_test.size(0)
                        test_running_mae += torch.abs(outputs_test - labels_test).sum().item() / NUM_OUTPUTS
                
                final_test_loss = test_running_loss / len(test_loader.dataset)
                final_test_mae = test_running_mae / len(test_loader.dataset)
                print(f"\nTest Results - Loss (MSE): {final_test_loss:.4f}, MAE: {final_test_mae:.4f}")
            else:
                print("\nNo test data to evaluate the final model.")


            # --- הצגת גרפים של תהליך האימון ---
            plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Training Loss (MSE)')
            if val_loader: plt.plot(val_losses, label='Validation Loss (MSE)')
            plt.title('Model Loss During Training (PyTorch)')
            plt.xlabel('Epochs')
            plt.ylabel('Loss (Mean Squared Error)')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(train_maes, label='Training MAE')
            if val_loader: plt.plot(val_maes, label='Validation MAE')
            plt.title('Model Mean Absolute Error During Training (PyTorch)')
            plt.xlabel('Epochs')
            plt.ylabel('Mean Absolute Error')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

            # --- שמירת המודל ---
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"\nTrained model saved to {MODEL_SAVE_PATH}")

            # --- דוגמה לניבוי על תמונה אקראית מסט המבחן (אם קיים) ---
            if test_loader and len(test_dataset) > 0:
                print("\n--- Example Prediction ---")
                # קבלת דגימה אקראית מסט המבחן
                # DataLoader אינו תומך באינדקס ישיר, אז ניקח את האצווה הראשונה ונבחר ממנה
                sample_batch_images, sample_batch_labels, sample_batch_filenames = next(iter(test_loader))
                
                random_idx_in_batch = np.random.randint(0, len(sample_batch_images))
                
                sample_image_tensor = sample_batch_images[random_idx_in_batch].unsqueeze(0).to(device) # הוספת מימד אצווה והעברה למכשיר
                actual_readings = sample_batch_labels[random_idx_in_batch].cpu().numpy() # המרה ל-NumPy להדפסה
                sample_filename = sample_batch_filenames[random_idx_in_batch]

                model.eval()
                with torch.no_grad():
                    predicted_readings_tensor = model(sample_image_tensor)
                predicted_readings = predicted_readings_tensor.cpu().numpy()[0]

                print(f"Filename: {sample_filename}")
                print(f"Actual Readings: {actual_readings}")
                print(f"Predicted Readings: {predicted_readings}")

                # הצגת התמונה
                # המרה מ-CHW (PyTorch) ל-HWC (Matplotlib) ומ-BGR ל-RGB
                display_image_np = sample_image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
                # אם התמונה עדיין מנורמלת (0-1), הכפל ב-255
                display_image_np = (display_image_np * 255).astype(np.uint8) if np.max(display_image_np) <= 1.0 else display_image_np
                # אם התמונה המקורית נטענה כ-BGR, והמרת ל-RGB לא נעשתה קודם, יש לעשות זאת כאן:
                # display_image_rgb = cv2.cvtColor(display_image_np, cv2.COLOR_BGR2RGB)
                # plt.imshow(display_image_rgb)
                # כרגע מניחים שהצבעים בסדר או שהמשתמש יתאים אם צריך
                plt.imshow(display_image_np)
                plt.title(f"File: {sample_filename}\nActual: {actual_readings[0]:.2f}, {actual_readings[1]:.2f}\nPredicted: {predicted_readings[0]:.2f}, {predicted_readings[1]:.2f}")
                plt.axis('off')
                plt.show()
            else:
                print("No test samples available for prediction example.")
