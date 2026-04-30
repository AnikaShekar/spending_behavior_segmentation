import json
import joblib
import pandas as pd

#Load saved model artefacts
scaler = joblib.load('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/models/scaler.pkl')
pca    = joblib.load('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/models/pca.pkl')
kmeans = joblib.load('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/models/kmeans.pkl')

with open('D:/Notes & TB/6th SEM/1.AIML/miniproject/spending-segmentation/models/cluster_names.json', 'r') as f:
    cluster_names = {int(k): v for k, v in json.load(f).items()}

used_features = scaler.feature_names_in_.tolist()

def get_input(prompt, min_val, max_val, is_int=False):
    """
    Repeatedly prompts the user until a valid number within [min_val, max_val]
    is entered.
    """
    while True:
        try:
            raw = input(f'  {prompt}: ').strip()
            val = int(raw) if is_int else float(raw)
            if min_val <= val <= max_val:
                return val
            print(f'Please enter a value between {min_val} and {max_val}.')
        except ValueError:
            print('Invalid input. Please enter a number.')

#Collect customer values interactively
print('CUSTOMER SEGMENT PREDICTOR')
print('Enter customer details below\n')

print('BALANCE AND PAYMENTS')
balance = get_input('Balance Rs (0 - 20000)',  0,   20000)
balance_freq = get_input('Balance Frequency (0.0 - 1.0)', 0.0, 1.0)
payments = get_input('Payments Rs (0 - 50000)',  0,   50000)
minimum_payments = get_input('Minimum Payments Rs (0 - 30000)',  0,   30000)

print('\n  PURCHASES')
purchases = get_input('Purchases Rs (0 - 50000)',  0,   50000)
purchases_freq = get_input('Purchases Frequency (0.0 - 1.0)', 0.0, 1.0)
oneoff_freq = get_input('One-off Purchases Frequency (0.0 - 1.0)', 0.0, 1.0)
installments_freq = get_input('Installments Purchases Frequency (0.0 - 1.0)', 0.0, 1.0)

print('\nCREDIT AND CASH ADVANCE')
credit_limit = get_input('Credit Limit Rs (0 - 30000)', 0, 30000)
cash_advance = get_input('Cash Advance Rs (0 - 47000)', 0, 47000)
cash_adv_freq = get_input('Cash Advance Frequency (0.0 - 1.0)', 0.0, 1.0)
prc_full = get_input('Full Payment Ratio (0.0 - 1.0)', 0.0, 1.0)
tenure = get_input('Tenure in months (1 - 12)', 1, 12, is_int=True)

new_customer = {
    'BALANCE' : balance,
    'BALANCE_FREQUENCY' : balance_freq,
    'PURCHASES' : purchases,
    'PURCHASES_FREQUENCY' : purchases_freq,
    'ONEOFF_PURCHASES_FREQUENCY' : oneoff_freq,
    'PURCHASES_INSTALLMENTS_FREQUENCY' : installments_freq,
    'CASH_ADVANCE' : cash_advance,
    'CASH_ADVANCE_FREQUENCY' : cash_adv_freq,
    'CREDIT_LIMIT' : credit_limit,
    'PAYMENTS' : payments,
    'MINIMUM_PAYMENTS' : minimum_payments,
    'PRC_FULL_PAYMENT' : prc_full,
    'TENURE' : tenure,
}

new_df = pd.DataFrame([new_customer], columns=used_features)

new_scaled = scaler.transform(new_df)
new_pca = pca.transform(new_scaled)
predicted = kmeans.predict(new_pca)[0]

print(f"\nPredicted Segment: {cluster_names[predicted]}\n")