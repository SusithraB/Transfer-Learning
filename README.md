# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Develop an image classification model using transfer learning with the pre-trained VGG19 model.
</br>
</br>
</br>

## DESIGN STEPS
### STEP 1:
Import required libraries.Then dataset is loaded and define the training and testing dataset.
</br>

### STEP 2:
initialize the model,loss function,optimizer. CrossEntropyLoss for multi-class classification and Adam optimizer for efficient training.
</br>

### STEP 3:
Train the model with training dataset.
<br/>
### STEP 4:
Evaluate the model with testing dataset.
<br/>
### STEP 5:
Make Predictions on New Data.
<br/>

## PROGRAM
Include your code here
```
## Step 3: Train the Model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # Reshape labels to match the output shape
            labels = labels.unsqueeze(1).type(torch.float32) # Reshape labels to have shape [batch_size, 1] and cast to float32
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # Reshape labels for validation as well
                labels = labels.unsqueeze(1).type(torch.float32) # Reshape validation labels
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:SUSITHRA.B")
    print("Register Number:212223220113")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    



```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/6ee670e2-935c-4a56-874e-ca5a7b8549c1)


### Confusion Matrix
![image](https://github.com/user-attachments/assets/a6c133b8-0bbd-4861-b214-6870bc213dbf)

### Classification Report
![image](https://github.com/user-attachments/assets/d188341c-ae08-4d7b-924c-421c76e59d17)


### New Sample Prediction
![image](https://github.com/user-attachments/assets/575b79b0-1b5a-4eb9-828d-b6721ab31fbc)

## RESULT
Thus, the transfer Learning for classification using VGG-19 architecture has succesfully implemented.
