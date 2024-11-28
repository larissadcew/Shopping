import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    evidence = []
    labels = []
    
    # Define a conversão dos meses para números de 0 a 11
    month_to_num = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3,
        "May": 4, "June": 5, "Jul": 6, "Aug": 7,
        "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
    }
    
    # Abrir o arquivo CSV e pular o cabeçalho
    with open(filename,'r') as file:
        reader = csv.reader(file)
        next(reader)  

        for row in reader:
            evidence.append([
                int(row[0]),                     
                float(row[1]),                   
                int(row[2]),                     
                float(row[3]),                   
                int(row[4]),                     
                float(row[5]),                  
                float(row[6]),                   
                float(row[7]),                   
                float(row[8]),                   
                float(row[9]),                   
                month_to_num[row[10]],          
                int(row[11]),                    
                int(row[12]),                    
                int(row[13]),                   
                int(row[14]),                    
                1 if row[15] == "Returning_Visitor" else 0,  
                1 if row[16] == "TRUE" else 0    
            ])
            labels.append(1 if row[17] == "TRUE" else 0) 

    return evidence, labels
def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence,labels)
    return model

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # Initialize counts
    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0

    # Count each case based on actual and predicted labels
    for actual, predicted in zip(labels, predictions):
        if actual == 1 and predicted == 1:
            true_positive += 1
        elif actual == 1 and predicted == 0:
            false_negative += 1
        elif actual == 0 and predicted == 0:
            true_negative += 1
        elif actual == 0 and predicted == 1:
            false_positive += 1

    # Calculate sensitivity and specificity
    sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0

    return (sensitivity, specificity)

if __name__ == "__main__":
    main()
