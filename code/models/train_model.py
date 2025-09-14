import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import joblib
from pathlib import Path
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

# Optional import for training functionality
try:
    from datasets.mnist_preprocessing import preprocess_dataset
except ImportError:
    # This import is only needed for training, not for model inference
    preprocess_dataset = None


class SimpleCNN(nn.Module):
    """Improved CNN for MNIST digit classification with better regularization"""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout1 = nn.Dropout(0.25)  # after conv layers
        self.dropout2 = nn.Dropout(0.5)  # after first FC layer
        self.dropout3 = nn.Dropout(0.3)  # after second FC layer

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout1(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout1(x)

        x = self.global_avg_pool(x)
        x = x.view(-1, 256)

        x = F.relu(self.fc1(x))
        x = self.dropout2(x)

        x = F.relu(self.fc2(x))
        x = self.dropout3(x)

        x = self.fc3(x)

        return x


def train_model():
    """Train the CNN model on MNIST dataset"""

    if preprocess_dataset is None:
        raise ImportError(
            "Training functionality requires the datasets module. This is only available during training, not in the API container."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = preprocess_dataset()

    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    num_epochs = 15
    patience = 5  # early stopping patience
    best_accuracy = 0.0
    patience_counter = 0

    print("Starting training with early stopping...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                )

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
            print(f"  New best validation accuracy: {best_accuracy:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print("\nEvaluating on test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()

    test_accuracy = 100 * test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)

    model_path = Path("../../models")
    model_path.mkdir(exist_ok=True)

    torch.save(model.state_dict(), model_path / "mnist_cnn.pth")

    model_info = {
        "model_class": "SimpleCNN",
        "num_classes": 10,
        "input_shape": (1, 28, 28),
        "transform_mean": 0.1307,
        "transform_std": 0.3081,
    }

    joblib.dump(model_info, model_path / "model_info.pkl")

    print(f"\nModel saved to {model_path}")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Test loss: {avg_test_loss:.4f}")

    return model, test_accuracy


if __name__ == "__main__":
    model, accuracy = train_model()
