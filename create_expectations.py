import pandas as pd
import great_expectations as ge
from great_expectations.data_context import DataContext

# Initialize the data context
context = DataContext()

# Load the wine quality dataset
wine_df = pd.read_csv("data/winequality-red.csv", sep=';')
wine_ge = ge.dataset.PandasDataset(wine_df)

# Create basic expectations for data drift detection
expectations = [
    # Check for required columns
    wine_ge.expect_table_columns_to_match_ordered_list([
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol", "quality"
    ]),
    
    # Non-null expectations
    wine_ge.expect_column_values_to_not_be_null("fixed acidity"),
    wine_ge.expect_column_values_to_not_be_null("volatile acidity"),
    wine_ge.expect_column_values_to_not_be_null("alcohol"),
    wine_ge.expect_column_values_to_not_be_null("quality"),
    
    # Range expectations
    wine_ge.expect_column_values_to_be_between("fixed acidity", min_value=4.0, max_value=16.0),
    wine_ge.expect_column_values_to_be_between("volatile acidity", min_value=0.1, max_value=1.2),
    wine_ge.expect_column_values_to_be_between("pH", min_value=2.7, max_value=4.0),
    wine_ge.expect_column_values_to_be_between("alcohol", min_value=8.0, max_value=15.0),
    wine_ge.expect_column_values_to_be_between("quality", min_value=3, max_value=9),
    
    # Distribution expectations (for detecting drift)
    wine_ge.expect_column_mean_to_be_between("alcohol", min_value=10.0, max_value=11.0),
    wine_ge.expect_column_median_to_be_between("quality", min_value=5.0, max_value=6.0),
    wine_ge.expect_column_stdev_to_be_between("pH", min_value=0.1, max_value=0.3),
]

# Save all expectations to the suite
for expectation in expectations:
    suite = context.get_expectation_suite("wine_quality_suite")
    suite.add_expectation(expectation["expectation_config"])
    context.save_expectation_suite(suite)

print("Expectations added to suite 'wine_quality_suite'")

# Build and open Data Docs
context.build_data_docs()
print("Data Docs built. You can open them to see your expectations.")
