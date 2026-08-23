
"""
trainer.py

Generic training engine for deep learning
stock forecasting models.
"""

# =====================================================
# Import Required Libraries
# =====================================================

import os
import copy
import torch
import torch.nn as nn

from dataclasses import dataclass, field

from utils.metrics import evaluate_predictions


# =====================================================
# Training History
# =====================================================

@dataclass
class TrainingHistory:
    """
    Stores training history throughout optimization.
    """

    train_loss: list = field(default_factory=list)

    val_loss: list = field(default_factory=list)

    mae: list = field(default_factory=list)

    rmse: list = field(default_factory=list)

    mape: list = field(default_factory=list)

    r2: list = field(default_factory=list)

    learning_rate: list = field(default_factory=list)


# =====================================================
# Trainer Class
# =====================================================

class Trainer:
    """
    Generic deep learning trainer.

    Supports any PyTorch forecasting model.
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler=None,
        device="cpu",
        model_path="saved_models/best_model.pth"
    ):

        self.model = model.to(device)

        self.criterion = criterion

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.device = device

        self.model_path = model_path

        self.history = TrainingHistory()

        self.best_model = None

        self.best_loss = float("inf")

        self.best_epoch = 0

    # =====================================================
    # Train One Epoch
    # =====================================================

    def train_one_epoch(self, train_loader):
        """
        Train the model for one epoch.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.

        Returns
        -------
        float
            Average training loss.
        """

        # Put the model into training mode
        self.model.train()

        running_loss = 0.0

        # Iterate through each mini-batch
        for inputs, targets in train_loader:

            # Move data to the selected device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward propagation
            outputs = self.model(inputs)

            # Handle models that return additional outputs
            if isinstance(outputs, tuple):
                predictions = outputs[0]
            else:
                predictions = outputs

            # Compute loss
            loss = self.criterion(predictions, targets)

            # Backpropagation
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            # Update model parameters
            self.optimizer.step()

            # Accumulate batch loss
            running_loss += loss.item()

        # Average training loss
        epoch_loss = running_loss / len(train_loader)

        return epoch_loss

    # =====================================================
    # Validation
    # =====================================================

    def validate(self, val_loader):
        """
        Evaluate the model on the validation dataset.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader.

        Returns
        -------
        tuple
            Validation loss and evaluation metrics.
        """

        # Switch to evaluation mode
        self.model.eval()

        running_loss = 0.0

        y_true = []
        y_pred = []

        # Disable gradient computation
        with torch.no_grad():

            # Iterate through validation batches
            for inputs, targets in val_loader:

                # Move data to the selected device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward propagation
                outputs = self.model(inputs)

                # Handle models returning multiple outputs
                if isinstance(outputs, tuple):
                    predictions = outputs[0]
                else:
                    predictions = outputs

                # Compute validation loss
                loss = self.criterion(predictions, targets)

                running_loss += loss.item()

                # Store predictions and targets
                y_true.extend(
                    targets.cpu().numpy().flatten()
                )

                y_pred.extend(
                    predictions.cpu().numpy().flatten()
                )

        # Average validation loss
        val_loss = running_loss / len(val_loader)

        # Compute evaluation metrics
        metrics = evaluate_predictions(
            y_true,
            y_pred
        )

        return val_loss, metrics

    # =====================================================
    # Model Training (Fit)
    # =====================================================

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
        patience=10
    ):
        """
        Train the model for multiple epochs.

        Parameters
        ----------
        train_loader : DataLoader

        val_loader : DataLoader

        epochs : int

        patience : int
            Early stopping patience.

        Returns
        -------
        TrainingHistory
        """

        early_stop_counter = 0

        for epoch in range(epochs):

            # -----------------------------------------
            # Training
            # -----------------------------------------

            train_loss = self.train_one_epoch(
                train_loader
            )

            # -----------------------------------------
            # Validation
            # -----------------------------------------

            val_loss, metrics = self.validate(
                val_loader
            )

            # -----------------------------------------
            # Save Training History
            # -----------------------------------------

            self.history.train_loss.append(
                train_loss
            )

            self.history.val_loss.append(
                val_loss
            )

            self.history.mae.append(
                metrics["MAE"]
            )

            self.history.rmse.append(
                metrics["RMSE"]
            )

            self.history.mape.append(
                metrics["MAPE"]
            )

            self.history.r2.append(
                metrics["R2"]
            )

          

            # -----------------------------------------
            # Save Learning Rate
            # -----------------------------------------

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history.learning_rate.append(
                current_lr
            )

            # -----------------------------------------
            # Learning Rate Scheduler
            # -----------------------------------------

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # -----------------------------------------
            # Save Best Model
            # -----------------------------------------

            if val_loss < self.best_loss:

                self.best_loss = val_loss

                self.best_epoch = epoch + 1

                self.best_model = copy.deepcopy(
                    self.model.state_dict()
                )

                torch.save(
                    self.best_model,
                    self.model_path
                )

                early_stop_counter = 0

            else:

                early_stop_counter += 1

            # -----------------------------------------
            # Progress
            # -----------------------------------------

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"| Train Loss: {train_loss:.5f} "
                f"| Val Loss: {val_loss:.5f} "
                f"| MAE: {metrics['MAE']:.5f} "
                f"| RMSE: {metrics['RMSE']:.5f} "
                f"| LR: {current_lr:.6f}"
                
            )

            # -----------------------------------------
            # Early Stopping
            # -----------------------------------------

            if early_stop_counter >= patience:

                print("\nEarly stopping triggered.")

                print(f"Best Epoch           : {self.best_epoch}")

                print(f"Best Validation Loss : {self.best_loss:.6f}"
                )

                print(f"Training completed after {epoch+1} epochs."
                )

                break

        # ---------------------------------------------
        # Restore Best Model
        # ---------------------------------------------

        if os.path.exists(self.model_path):

            self.model.load_state_dict(
                torch.load(
                    self.model_path,
                    map_location=self.device
                )
             )

        return self.history
    
    # =====================================================
    # Prediction
    # =====================================================

    def predict(
        self,
        data_loader,
        return_attention=False
    ):
        """
        Generate predictions using the trained model.

        Parameters
        ----------
        data_loader : DataLoader
            Data loader for inference.

        return_attention : bool, optional
            If True, return attention weights for
            attention-based models.

        Returns
        -------
        numpy.ndarray
            Predicted values.

        If return_attention=True:
            Returns (predictions, attention_weights)
        """

        import numpy as np

        # Switch model to evaluation mode
        self.model.eval()

        predictions = []
        attention_weights = []

        # Disable gradient computation
        with torch.no_grad():

            for inputs, _ in data_loader:

                # Move inputs to device
                inputs = inputs.to(self.device)

                # Forward propagation
                outputs = self.model(inputs)

                # Handle models with attention
                if isinstance(outputs, tuple):

                    preds = outputs[0]
                    attention = outputs[1]

                    if return_attention:
                        attention_weights.append(
                            attention.cpu().numpy()
                        )

                else:

                    preds = outputs

                # Store predictions
                predictions.append(
                    preds.cpu().numpy()
                )

        # Concatenate all batches
        predictions = np.concatenate(
            predictions,
            axis=0
        ).ravel()

        # Return attention weights if requested
        if return_attention:

            attention_weights = np.concatenate(
                attention_weights,
                axis=0
            )

            return predictions, attention_weights

        return predictions

