## Category classification using string matching and machine learning model

### Approach 1 : Build a classifier with string matching i.e. without using any machine learning model

class MedicalClassifier:
    def __init__(self):
        # Define keywords for each category
        self.categories = {
            'Ambulance Charges': ['AMBULANCE', 'AMBULANCE SERVICE', 'BASIC LIFE SUPPORT (BLS) AMBULANCE', 'ACL'],
            'Anaesthesia': ['ANAESTHESIA', 'SPINAL ANESTHESIA', 'ANESTHETIST CHRGE', 'ISOFLURANE', 'PROFOL', 'SEDATION'],
            'Blood & Blood Products': ['BLOOD', 'PLASMA', 'PRBC', 'RED CELLS', 'BLOOD GROUPING', 'BLOOD COMPONENT'],
            'Doctor Fees': ['DOCTOR', 'CONSULTANT', 'NUTRITIONIST', 'SURGEON', 'DIETICIAN', 'MEDICAL HISTORY ASSESSMENT'],
            'Food & Beverages': ['FOOD', 'BEVERAGES', 'DIET', 'FRUIT SALAD', 'WATER MELON JUICE', 'MEAL'],
            'Medicines & Consumables': ['INJ', 'SYRINGE', 'STERILE', 'GLOVE', 'FACEMASK', 'TOWEL', 'PLASTER', 'DRESSING', 'BANDAGE', 'DIAPERS', 'IV CANNULA'],
            'Procedure Charges': ['ANGIOGRAM', 'PTCA', 'ANGIOPLASTY', 'CATARACT MICS', 'CHOLECYSTECTOMY', 'COLONOSCOPY'],
            'Room Rent': ['ROOM CHARGES', 'RENT', 'PRIVATE WARD', 'SINGLE ROOM', 'DOUBLE BED', 'GENERAL WARD', 'BED CHARGES', 'WARD CHARGES'],
            'Returns': ['Returns', 'Refund', 'Cancellation']
        }

    def classify_text(self, input_text):
        # Check for keywords in input_text and return the corresponding category
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword.lower() in input_text.lower():
                    return category
        # If no match found, return 'Uncategorized'
        return 'Uncategorized'


# Example Use Case
classifier = MedicalClassifier()
input_text = "coronary angiogram"
result = classifier.classify_text(input_text)
print("Expected Output:", "Procedure Charges")
print("Actual Output:", result)


### Approach 2 : Build a classifier using machine learning model

# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('bills.csv')

# Preprocess the data
X = data['description']  # Features
y = data['category']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the KNN classifier
k = 5  # We can choose the value of k based on our preference
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test_tfidf)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Function to predict the category of a given medical text
def predict_category(text):
    text_tfidf = vectorizer.transform([text])
    category = knn_classifier.predict(text_tfidf)
    return category[0]

# Example Use Case
input_text = "ANAESTHESIA INFLATABLE AIRCUSHION FACE"
predicted_category = predict_category(input_text)
print(f'Predicted Category: {predicted_category}')
