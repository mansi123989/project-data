from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score






st.title("Data Explorer")
import pandas as pd
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Display the dataframe
    st.write(" Data Preview", df.sample(5))
    
    st.write("null values in your data ")
    st.write(df.isnull().sum() )

    option = st.radio(
        "How do you want to handle null values?",
        ("Do nothing", "Drop rows with nulls", "Fill nulls with 0", "Fill nulls with mean")
    )

    # Apply chosen method
    if option == "Drop rows with nulls":
        df_cleaned = df.dropna()
        st.success("Rows with null values dropped.")

    elif option == "Fill nulls with 0":
        df_cleaned = df.fillna(0)
        st.success("Null values filled with 0.")

    elif option == "Fill nulls with mean":
        df_cleaned = df.fillna(df.mean(numeric_only=True))
        st.success("Null values filled with column mean.")

    else:
        df_cleaned = df  

        # Let user pick a column to plot (categorical or object columns)
    col_options = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if col_options:
        selected_col = st.selectbox("Select a column to plot (e.g. Names)", col_options)

        # Count values and plot
        value_counts = df[selected_col].value_counts()

        st.write(f"### Bar Chart for '{selected_col}'")

        # Plot using matplotlib
        fig, ax = plt.subplots()
        value_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Count")
        ax.set_title(f"Frequency of {selected_col}")
        st.pyplot(fig)
  
    
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score


# Select target column
    target = st.selectbox("Select Target Column", df.columns)

    # Encode categorical features
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=['object', 'category']):
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

    # Detect classification target
    y = df_encoded[target]
    if y.dtype in [np.float64, np.int64] and y.nunique() > 10:
        st.warning("Detected continuous target — treating as classification by binning into 3 categories.")
        y = pd.cut(y, bins=3, labels=[0, 1, 2])
        df_encoded[target] = y

    # Correlation-based feature selection
    corr = df_encoded.corr()[target].abs().sort_values(ascending=False)
    selected_features = corr.drop(target)[corr > 0.1].index.tolist()
    st.write("### Selected Features Based on Correlation", selected_features)

    # Data prep
    X = df_encoded[selected_features]
    y = df_encoded[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define classification models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC()
    }

    results = {}

    st.write("### Model Evaluation (using Accuracy Score)")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)*100
        results[name] = acc/100

        st.markdown(f"#### {name} - Accuracy: `{acc:.2f}`%")

        # Plot actual vs predicted
        fig, ax = plt.subplots()
        ax.plot(y_test.values, label="Actual", marker='o')
        ax.plot(y_pred, label="Predicted", marker='x')
        ax.set_title(f"{name} - Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)

        st.markdown("---")  # adds space between graphs

    # Final summary chart
    st.write("### Accuracy Comparison")
    result_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])

    fig2, ax2 = plt.subplots()
    ax2.bar(result_df["Model"], result_df["Accuracy"], color="lightgreen")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Model Accuracy Comparison")
    plt.xticks(rotation=45)

    # Fix Y-axis range for better visual distinction
    min_acc = min(results.values())
    max_acc = max(results.values())
    margin = 0.05
    ax2.set_ylim(min_acc - margin, min(1.0, max_acc + margin))


    st.pyplot(fig2)
 
else:
    st.warning("No text/categorical columns found to plot.")       


    
