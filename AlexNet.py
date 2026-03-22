import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


PATH = "."
TRAIN_CSV_RAW = os.path.join(PATH, 'train.csv')
TEST_CSV_RAW = os.path.join(PATH, 'test.csv')
TRAIN_DIR = os.path.join(PATH, 'train/')
TEST_DIR = os.path.join(PATH, 'test/')
batch = 64
TRAIN_SPLIT = 0.8
num_epochs = 21

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du : {device}")


def is_valid_image(path):
    if not os.path.exists(path):
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False


def find_valid_image(root_dir, img_name):
    exts = ['.jpg', '.jpeg', '.png', '.webp']
    base, ext = os.path.splitext(str(img_name))
    if ext.lower() in exts:
        img_path = os.path.join(root_dir, img_name)
        if is_valid_image(img_path): return img_path
    for ext in exts:
        img_path = os.path.join(root_dir, base + ext)
        if is_valid_image(img_path): return img_path
    return None


transform_tr = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter( brightness=0.4,contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_tst = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



class CoinDataset(Dataset):
    def __init__(self, dataframe, transform=None, is_test=False, class_idx=None):
        self.data_frame = dataframe
        self.transform = transform
        self.is_test = is_test
        self.class_idx = class_idx

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx]['path']
        img_name = str(self.data_frame.iloc[idx]['Id'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, img_name
        else:
            label_str = self.data_frame.iloc[idx]['Class']
            label = torch.tensor(self.class_idx[label_str], dtype=torch.long)
            return image, label


def prepare_data():
    print("Filtrage train--")
    df = pd.read_csv(TRAIN_CSV_RAW)
    df["path"] = df["Id"].apply(lambda x: find_valid_image(TRAIN_DIR, x))
    df = df[df["path"].notna()].reset_index(drop=True)
    print("Filtrage test--")
    df_test = pd.read_csv(TEST_CSV_RAW)
    df_test["path"] = df_test["Id"].apply(lambda x: find_valid_image(TEST_DIR, x))
    df_test = df_test[df_test["path"].notna()].reset_index(drop=True)

    print(f"Tailles après filtrage - Train: {len(df)}, Test: {len(df_test)}")
    classes = df['Class'].unique()
    class_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return df, df_test, class_idx, len(classes)



class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    df_train, df_test, class_idx, num_classes = prepare_data()

    train_size = int(TRAIN_SPLIT * len(df_train))
    df_train_split = df_train.sample(n=train_size, random_state=42)
    df_val_split = df_train.drop(df_train_split.index)

    train_dataset = CoinDataset(df_train_split, transform=transform_tr, class_idx=class_idx)
    val_dataset = CoinDataset(df_val_split, transform=transform_tst, class_idx=class_idx)
    test_dataset = CoinDataset(df_test, transform=transform_tst, is_test=True, class_idx=class_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    model = AlexNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=0.0005)

    print("\nEntrainement.")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_train=0
        correct_train=0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Train"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {running_loss / len(train_loader):.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss / len(val_loader):.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

    model.eval()
    predictions = []
    image_ids = []

    idx_to_class = {v: k for k, v in class_idx.items()}
    with torch.no_grad():
        for images, img_names in tqdm(test_loader, desc="Test"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            pred_classes = [idx_to_class[idx.item()] for idx in predicted]
            predictions.extend(pred_classes)
            image_ids.extend(img_names)

    submission_df = pd.DataFrame({'image_id': image_ids, 'label': predictions})

    sample_sub_path = os.path.join(PATH, 'kaggle_submission.csv')
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
        submission_df.columns = sample_sub.columns

    out_file = os.path.join(PATH, 'kaggle_soummission.csv')
    submission_df.to_csv(out_file, index=False)
    print(f"Test csv genere")