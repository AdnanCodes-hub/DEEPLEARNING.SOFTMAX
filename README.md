============================================================
               Softmax Classification with TensorFlow
============================================================

This project demonstrates a basic implementation of a multi-class
classification model using TensorFlow and Keras. The model is trained
on a custom 2D dataset and uses the softmax activation function to 
classify data points into 10 distinct classes.

------------------------------------------------------------
                     📁 Project Structure
------------------------------------------------------------

Softmax-Classification/
├── dataset.txt              --> Custom dataset (2 features + 1 label)
├── Train.py                 --> Python script to train and visualize
├── softmax_model.keras      --> Saved trained model
└── README.txt               --> Project documentation (this file)

------------------------------------------------------------
                       📊 Dataset Format
------------------------------------------------------------

- File:     dataset.txt
- Format:   CSV (comma-separated)
- Shape:    (n_samples, 3)

Each row contains:
  - Feature 1 (float)
  - Feature 2 (float)
  - Label     (integer: 0–9)

Example:
  1.25, 3.62, 0
  2.13, 1.05, 4
  ...

------------------------------------------------------------
                        🚀 How to Run
------------------------------------------------------------

1. Install dependencies (Python 3.10+ recommended):

       pip install numpy matplotlib tensorflow

2. Run the script:

       python Train.py

This will:
 - Load and visualize the dataset.
 - Build and train the model.
 - Save the trained model as 'softmax_model.keras'.
 - Plot the decision boundary.

------------------------------------------------------------
                   🧠 Model Architecture
------------------------------------------------------------

Input Layer :  2 features
Hidden Layer:  10 neurons, ReLU activation
Output Layer:  10 neurons, Softmax activation (for 10 classes)

------------------------------------------------------------
                        📈 Output
------------------------------------------------------------

✔ Console output showing training accuracy per epoch  
✔ Visualization of dataset and decision boundaries  
✔ Trained model saved to: softmax_model.keras

------------------------------------------------------------
                     ⚠️ Common Issues
------------------------------------------------------------

• "Label value X outside of range [0, 3)"  
    ➤ This happens if your dataset contains labels like 9, but 
      your output layer only has 3 units.  
    ➤ Fix: Use `Dense(units=10)` for 10 classes.

• Keras input warning  
    ➤ Instead of input_shape in Dense, we use tf.keras.Input()

------------------------------------------------------------
                       📌 Author Info
------------------------------------------------------------

• Name      : Adnan Only (or your name)
• Purpose   : Machine Learning / Deep Learning Practice
• Tools     : Python, TensorFlow, NumPy, Matplotlib

------------------------------------------------------------
                        📝 License
------------------------------------------------------------

This project is open for learning, educational, and personal 
portfolio use. Feel free to modify and extend it as needed.
