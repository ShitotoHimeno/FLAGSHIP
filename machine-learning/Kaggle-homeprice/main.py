import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from src.datacleaning import load_and_preprocess
from src.svr import optimize_svr, train_final_model

def plot_analysis(df):
    # 画像のような散布図行列を作成
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
    print("可視化を実行中...")
    sns.pairplot(df[cols].dropna())
    plt.show()

def main():
    train_path = 'machine-learning\Kaggle-homeprice\data\train.csv'
    test_path = 'machine-learning\Kaggle-homeprice\data\test.csv'
    
    # 1. 前処理と生データの取得
    X_train, y_train, X_test, test_ids, raw_train = load_and_preprocess(train_path, test_path)
    
    # 2. 画像のような分析図を表示
    plot_analysis(raw_train)
    
    # 3. ハイパーパラメータ最適化
    print("最適化を開始します...")
    best_params = optimize_svr(X_train, y_train)
    print(f"最良パラメータ: {best_params}")
    
    # 4. 学習と予測
    model = train_final_model(X_train, y_train, best_params)
    predictions = model.predict(X_test)
    
    # 5. 提出ファイル作成
    submission = pd.DataFrame({'Id': test_ids, 'SalePrice': predictions})
    submission.to_csv('submission.csv', index=False)
    print("完了！ submission.csv を保存しました。")

if __name__ == "__main__":
    main()