import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split




# Now you can use clf for training and prediction



sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# Function to upload CSV file
def upload_file():
    st.markdown(
        """
        <div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>
            <h2 style='color: #333333;'>Upload Your CSV File 📁</h2>
            <p style='color: #666666;'>Upload your CSV file here. We'll take care of the rest!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        st.success("File uploaded successfully! 🎉")

        # Display the first few rows of the uploaded data
        df = pd.read_csv(uploaded_file)


        return df
    # Create a dictionary of classifier models




def explore_data(df):
    st.markdown("<h1 style='color:blue;'>Data Exploration</h1>", unsafe_allow_html=True)
    st.write(df.head())


# Assuming df is your DataFrame
    st.write("## Attrition Count Plot")
    sns.countplot(x='Attrition', data=df)
    st.pyplot()

    df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis="columns", inplace=True)

    categorical_col = []
    for column in df.columns:
        if df[column].dtype == object and len(df[column].unique()) <= 50:
            categorical_col.append(column)
        
            df['Attrition'] = df.Attrition.astype("category").cat.codes
    categorical_col.remove('Attrition')
    from sklearn.preprocessing import LabelEncoder

    label = LabelEncoder()
    for column in categorical_col:
        df[column] = label.fit_transform(df[column])

    from sklearn.model_selection import train_test_split

    X = df.drop('Attrition', axis=1)
    y = df.Attrition

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)





# Define function to print scores
    def print_score(clf, X_train, y_train, X_test, y_test, train=True):
        if train:
            pred = clf.predict(X_train)
            clf_report = classification_report(y_train, pred, output_dict=True)
            accuracy = accuracy_score(y_train, pred) * 100
            confusion = confusion_matrix(y_train, pred)
            st.write(" Train Result")
        else:
            pred = clf.predict(X_test)
            clf_report = classification_report(y_test, pred, output_dict=True)
            accuracy = accuracy_score(y_test, pred) * 100
            confusion = confusion_matrix(y_test, pred)
            st.write(" Test Result")
        
    # Convert classification report to DataFrame
        clf_report_df = pd.DataFrame(clf_report).transpose()
    
    # Display accuracy score
        st.write(f"Accuracy Score: {accuracy:.2f}%")

    # Display classification report
        st.write(" Classification Report")
        st.dataframe(clf_report_df)
    
    # Display confusion matrix
        st.write(r" Confusion Matrix")
        st.dataframe(pd.DataFrame(confusion))




# Let the user choose a classifier model
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}
    selected_model = st.selectbox('Select Classifier Model', list(models.keys()))

# Get the selected model
    tree_clf = models[selected_model]


# Train and evaluate the selected model
    st.markdown("<h1 style='color:blue;'>Model Training</h1>", unsafe_allow_html=True)
    tree_clf.fit(X_train, y_train)
    print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
    print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)

        
def main():
    st.markdown("<h1 style='color:blue;'>Machine Learning Pipeline</h1>", unsafe_allow_html=True)

    
    # Upload CSV file
    df = upload_file()
    # Create a dictionary of classifier models
 
    
    if df is not None:
        # Perform data exploration
        explore_data(df)
        


if __name__ == "__main__":
    main()
