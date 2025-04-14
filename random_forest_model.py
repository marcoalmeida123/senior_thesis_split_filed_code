# === File: random_forest_model.py ===
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_random_forest(data):
    data['Target'] = (data['Return'].shift(-1) > 0).astype(int)
    features = ['Return', 'SMA_50', 'HMM_state']
    X = data[features]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model, X_test, y_test
