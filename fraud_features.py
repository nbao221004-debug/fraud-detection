"""Hàm tạo đặc trưng dùng chung cho lúc train và lúc deploy.

Đặt trong một file riêng để joblib có thể nạp lại pipeline ở app.py.
"""
import pandas as pd

RAW_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
CATEGORICAL = ["type"]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Thêm các đặc trưng phát hiện bất thường số dư."""
    X = X.copy()
    # Số dư gốc lẽ ra phải giảm đúng bằng số tiền: lệch nghĩa là bất thường
    X["errorBalanceOrig"] = X["newbalanceOrig"] + X["amount"] - X["oldbalanceOrg"]
    # Số dư đích lẽ ra phải tăng đúng bằng số tiền
    X["errorBalanceDest"] = X["oldbalanceDest"] + X["amount"] - X["newbalanceDest"]
    # Tài khoản gốc bị rút sạch số dư
    X["origEmptied"] = ((X["newbalanceOrig"] == 0) & (X["oldbalanceOrg"] > 0)).astype(int)
    # Tài khoản đích trống cả trước và sau giao dịch
    X["destEmpty"] = ((X["oldbalanceDest"] == 0) & (X["newbalanceDest"] == 0)).astype(int)
    return X[RAW_COLUMNS + ["errorBalanceOrig", "errorBalanceDest", "origEmptied", "destEmpty"]]
