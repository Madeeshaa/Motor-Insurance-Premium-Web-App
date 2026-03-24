import sys
# Redirect stdout if needed but we want to see output
import premium_app_script as pas

print("Models loaded successfully!")
print("Testing prediction:")

try:
    pp = pas.predict_pure_premium(25, 5, 3, 15000, 1600)
    print(f"Predicted Pure Premium: {pp:.2f}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

print("Test Passed")
