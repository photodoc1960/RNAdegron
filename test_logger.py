from Logger import CSVLogger
import os

# Test the CSV logger
print("Testing CSV logger...")

# Create test directory
os.makedirs('test_logs', exist_ok=True)

# Test logger
columns = ['epoch', 'train_loss', 'val_loss']
logger = CSVLogger(columns, 'test_logs/test.csv')

# Log some test data
test_data = [
    [1, 0.5, 0.6],
    [2, 0.4, 0.5],
    [3, 0.3, None],  # Test None value
    [4, 0.2, 0.4],
]

for data in test_data:
    logger.log(data)

# Read back and check
import pandas as pd
df = pd.read_csv('test_logs/test.csv')
print("Test CSV content:")
print(df)

# Clean up
os.remove('test_logs/test.csv')
os.rmdir('test_logs')

print("CSV logger test completed.")
