import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 訓練データとテストデータを結合して一括処理
    merged_df = pd.concat([train_df, test_df], axis=0, sort=False)
    
    # 欠損値処理 (Notebookのロジックを反映)
    merged_df = merged_df.drop(['PoolQC', 'MiscFeature', 'MasVnrArea', 'MasVnrType'], axis=1)
    
    # 数値列の補完
    num_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        if col != "SalePrice":
            merged_df[col] = merged_df[col].fillna(merged_df[col].mean())
            
    # カテゴリ列の補完
    obj_cols = merged_df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        merged_df[col] = merged_df[col].fillna(merged_df[col].mode()[0])

    # One-Hot Encoding
    merged_df = pd.get_dummies(merged_df)
    
    # 再分割
    train_preprocessed = merged_df[merged_df['SalePrice'].notnull()].copy()
    test_preprocessed = merged_df[merged_df['SalePrice'].isnull()].copy()
    
    # 正規化 (StandardScaler)
    scaler = StandardScaler()
    X_train = train_preprocessed.drop(['Id', 'SalePrice'], axis=1)
    y_train = train_preprocessed['SalePrice']
    X_test = test_preprocessed.drop(['Id', 'SalePrice'], axis=1)
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, test_df['Id'], train_df