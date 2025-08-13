from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def validate_synthetic(real_data, synthetic_data):
    X = np.vstack([real_data, synthetic_data])
    y = np.array([0]*len(real_data) + [1]*len(synthetic_data))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = RandomForestClassifier().fit(X_train, y_train)
    return clf.score(X_test, y_test)  # Closer to 0.5 = better privacy