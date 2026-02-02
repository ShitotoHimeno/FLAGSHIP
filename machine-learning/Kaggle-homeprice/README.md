# House Prices: SVR Prediction Project

このプロジェクトは、Kaggleの「House Prices: Advanced Regression Techniques」データセットを用い、SVR（サポートベクター回帰）とOptunaによるハイパーパラメータ最適化を組み合わせて住宅価格を予測するものです。

## 📁 フォルダ構成
```text
.
├── main.py                # 全体実行・フロー制御
├── README.md              # プロジェクト説明書
├── requirements.txt       # 依存ライブラリ一覧
├── data/                  # データ格納用フォルダ
│   ├── train.csv
│   └── test.csv
└── src/                   # ソースコード用フォルダ
    ├── datacleaning.py    # データ読込・欠損値処理・スケーリング
    └── svr.py             # SVRモデル定義・Optuna最適化

## 感想

自身で行った予測ではpublic scoreは0.14280。
前処理が大変なだけで綺麗にしたデータをライブラリにぶち込むだけで肩透かしを食らった。
また、同じようにxgboostで行った結果ではpublic scoreは0.13687。
タイプの違う予測手法に対して理解が深まったと思う。