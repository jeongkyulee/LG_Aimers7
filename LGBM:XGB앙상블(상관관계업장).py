# =========================================================
# LG Aimers — 7일 수요예측 (룰 준수)
# 베이스: 스텝별 LGB/XGB 앙상블 (가중치는 Softmax(sMAPE) 기반 자동 산출)
# 추가(Step1): 화담숲주막/화담숲카페 "전용" LGB/XGB 모델 + 공용 앙상블과 블렌딩(Softmax(sMAPE))
# 주의: 외부 피처 추가 없음. 기존 기능 삭제 없음. 전용 모델은 target log1p 사용.
# =========================================================
import os, re, glob, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error  # (학습 중 early stopping은 MAE 그대로 사용)
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import xgboost as xgb
from workalendar.asia import SouthKorea

# ============ 설정 ============
LOOKBACK = 28
PREDICT_DAYS = 7
SEED = 42
np.random.seed(SEED)

SPECIAL_SHOPS = ["화담숲주막", "화담숲카페"]  # (원래 틀 유지용 — 아래에서 그룹으로 재정의)
TEMP_SOFTMAX = 0.6  # Softmax temperature (0.3~1.0 권장)

# ---- 그룹 정의 (요청 사항) ----
PAIR_1 = ["화담숲주막", "화담숲카페"]
PAIR_2 = ["카페테리아", "포레스트릿"]

# ============ sMAPE & Softmax 유틸 ============
EPS = 1e-8

def smape(y_true, y_pred, eps: float = EPS, multiply_100: bool = False) -> float:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    denom = np.abs(yt) + np.abs(yp) + eps
    val = np.mean(2.0 * np.abs(yp - yt) / denom)
    return (val * 100.0) if multiply_100 else val

def softmax_weights_from_smape(score_dict: dict, temperature: float = 1.0) -> dict:
    names = list(score_dict.keys())
    smapes = np.array([score_dict[n] for n in names], dtype=float)  # 낮을수록 좋음
    tau = max(float(temperature), 1e-6)
    logits = -smapes / tau
    logits -= logits.max()
    w = np.exp(logits)
    w = w / (w.sum() + EPS)
    return dict(zip(names, w))

# ============ 공통 함수/달력 ============
cal = SouthKorea()

def get_holidays(years):
    return set(pd.Timestamp(d[0]) for y in years for d in cal.holidays(y))

HOLI_TRAIN = get_holidays([2023, 2024])
HOLI_TEST  = get_holidays([2024, 2025])

def get_season(m:int)->int:
    if m in [3,4,5]: return 0  # 봄
    if m in [6,7,8]: return 1  # 여름
    if m in [9,10,11]: return 2 # 가을
    return 3                   # 겨울

def preprocess_base(df: pd.DataFrame, *, use_test_holidays: bool=False) -> pd.DataFrame:
    """기본 날짜/달력/주기/업장/메뉴 분리까지 한 번에."""
    df = df.copy()
    df['매출수량'] = df['매출수량'].clip(lower=0)
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    df[['영업장명','메뉴명']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)

    df['요일'] = df['영업일자'].dt.weekday
    df['월']   = df['영업일자'].dt.month
    df['주말'] = df['요일'].isin([5,6]).astype(int)

    holi = HOLI_TEST if use_test_holidays else HOLI_TRAIN
    df['공휴일'] = df['영업일자'].isin(holi).astype(int)
    df['계절']  = df['월'].apply(get_season)

    # 주기 인코딩
    df['요일_sin'] = np.sin(2*np.pi*df['요일']/7)
    df['요일_cos'] = np.cos(2*np.pi*df['요일']/7)
    df['월_sin']   = np.sin(2*np.pi*df['월']/12)
    df['월_cos']   = np.cos(2*np.pi*df['월']/12)
    df['연중일']   = df['영업일자'].dt.dayofyear
    df['doy_sin']  = np.sin(2*np.pi*df['연중일']/365.25)
    df['doy_cos']  = np.cos(2*np.pi*df['연중일']/365.25)

    # 전/후 공휴일
    cal_map = df[['영업일자','공휴일']].drop_duplicates().sort_values('영업일자')
    cal_map['전일_공휴일'] = cal_map['공휴일'].shift(1).fillna(0).astype(int)
    cal_map['익일_공휴일'] = cal_map['공휴일'].shift(-1).fillna(0).astype(int)
    df = df.merge(cal_map[['영업일자','전일_공휴일','익일_공휴일']], on='영업일자', how='left')

    # 일부 업장 성수기 플래그
    peak_summer_shops = {'느티나무 셀프BBQ'}
    peak_winter_shops = {'포레스트릿', '화담숲주막', '화담숲카페'}
    df['is_peak_summer_shop'] = df['영업장명'].isin(peak_summer_shops).astype(int)
    df['is_peak_winter_shop'] = df['영업장명'].isin(peak_winter_shops).astype(int)
    df['peak_summer'] = df['월'].isin([6,7,8]).astype(int) * df['is_peak_summer_shop']
    df['peak_winter'] = df['월'].isin([12,1,2]).astype(int) * df['is_peak_winter_shop']

    return df

# 앵커 메뉴 매핑
ANCHORS = {
    '느티나무 셀프BBQ': ['1인 수저세트', 'BBQ55'],
    '미라시아': ['브런치', '브런치(대인)', '미라시아 브런치'],
    '연회장': ['공깃밥', 'Cookie Platter'],
}

def is_anchor_menu(row):
    keys = ANCHORS.get(row['영업장명'], [])
    if not keys: return 0
    name = str(row['메뉴명'])
    return int(any(k in name for k in keys))

def consec_runs_of_zero(prev_series: pd.Series) -> pd.Series:
    out = np.zeros(len(prev_series), dtype=int)
    cnt = 0
    for i, v in enumerate(prev_series.fillna(-1).values):
        if v == 0: cnt += 1
        else: cnt = 0
        out[i] = cnt
    return pd.Series(out, index=prev_series.index)

def create_features(df: pd.DataFrame, le_menu: LabelEncoder, le_shop: LabelEncoder) -> pd.DataFrame:
    df = df.sort_values(['영업장명','메뉴명','영업일자']).copy()

    # 라벨 인코딩
    if '메뉴ID' not in df.columns:
        df['메뉴ID'] = le_menu.transform(df['영업장명_메뉴명'])
    if '영업장ID' not in df.columns:
        df['영업장ID'] = le_shop.transform(df['영업장명'])

    # ---- 기본 lag/rolling ----
    for lag in [1,2,7,14,28]:
        df[f'lag_{lag}'] = df.groupby('메뉴ID')['매출수량'].shift(lag)

    for win in [3,7,14]:
        g = df.groupby('메뉴ID')['매출수량'].shift(1).rolling(win)
        df[f'roll_mean_{win}'] = g.mean()
        df[f'roll_std_{win}']  = g.std()

    df['ewm_7'] = df.groupby('메뉴ID')['매출수량'].shift(1).ewm(span=7, adjust=False).mean()

    # ---- 스파이크 & 연속 0 ----
    prev = df.groupby('메뉴ID')['매출수량'].shift(1)
    q90  = prev.groupby(df['메뉴ID']).transform(lambda s: s.rolling(28, min_periods=3).quantile(0.90))
    df['spike_prev'] = (prev > q90).astype(int)
    df['run_zero']   = prev.groupby(df['메뉴ID']).transform(consec_runs_of_zero)

    # ---- 업장 총수요 ----
    df['shop_total'] = df.groupby(['영업장ID','영업일자'])['매출수량'].transform('sum')
    grp_shop = df.groupby('영업장ID', sort=False)
    df['shop_total_shift1'] = grp_shop['shop_total'].shift(1)
    for win in [7,14,28]:
        df[f'shop_roll_mean_{win}'] = grp_shop['shop_total_shift1'].transform(lambda s: s.rolling(win, min_periods=1).mean())
        df[f'shop_roll_std_{win}']  = grp_shop['shop_total_shift1'].transform(lambda s: s.rolling(win, min_periods=2).std())

    # 메뉴 점유율
    df['share_7']  = df['roll_mean_7']  / (df['shop_roll_mean_7']  + 1e-6)
    df['share_14'] = df['roll_mean_14'] / (df['shop_roll_mean_14'] + 1e-6)
    df['share_28'] = df['roll_mean_14'] / (df['shop_roll_mean_28'] + 1e-6)

    # ---- 앵커 메뉴 ----
    if 'is_anchor' not in df.columns:
        df['is_anchor'] = df.apply(is_anchor_menu, axis=1)
    df['anchor_sales'] = df['매출수량'] * df['is_anchor']
    df['anchor_total'] = df.groupby(['영업장ID','영업일자'])['anchor_sales'].transform('sum')
    df['anchor_total_shift1'] = grp_shop['anchor_total'].shift(1)
    for win in [7,14]:
        df[f'anchor_roll_{win}'] = grp_shop['anchor_total_shift1'].transform(lambda s: s.rolling(win, min_periods=1).mean())

    # ---- 신규 추가: 최근 활동성 (주막/카페 대응) ----
    df['recent_activity'] = prev.groupby(df['메뉴ID']).transform(lambda s: s.rolling(7, min_periods=1).apply(lambda r: (r>0).any(), raw=True))
    df['active_ratio']    = prev.groupby(df['메뉴ID']).transform(lambda s: (s.rolling(28, min_periods=1).apply(lambda r: (r>0).sum(), raw=True))/28.0)

    # ---- 보강: 주막/카페 대응 피처 ----
    df['is_open_7d'] = prev.groupby(df['메뉴ID']).transform(
        lambda s: s.rolling(7, min_periods=1).max()
    )
    df['open_ratio_7d'] = prev.groupby(df['메뉴ID']).transform(
        lambda s: s.rolling(7, min_periods=1).apply(lambda r: (r>0).sum(), raw=True) / 7.0
    )
    df['last_open_gap'] = prev.groupby(df['메뉴ID']).transform(
        lambda s: s[::-1].groupby((s!=0)[::-1].cumsum()).cumcount()[::-1]
    )

    return df

# ============ 데이터 로드 & 전처리 ============
train = pd.read_csv("/Users/jeong-kyu/Documents/LG_Aimers_7기/open/train/train.csv")
train = preprocess_base(train, use_test_holidays=False)

# 인코더 학습
le_menu = LabelEncoder().fit(train['영업장명_메뉴명'])
le_shop = LabelEncoder().fit(train['영업장명'])

# 피처 생성
train_feat = create_features(train, le_menu, le_shop).dropna().reset_index(drop=True)

# ============ 검증 분할(최근 60일) ============
cutoff = train_feat['영업일자'].max() - pd.Timedelta(days=60)
train_df = train_feat[train_feat['영업일자'] <= cutoff].copy()
valid_df = train_feat[train_feat['영업일자'] >  cutoff].copy()

EXCLUDE = {
    '영업일자','영업장명_메뉴명','영업장명','메뉴명','매출수량','target',
    'anchor_sales'  # 내부 파생(누수 아님이지만 모델엔 불필요)
}
features = [c for c in train_feat.columns if c not in EXCLUDE]
print(f"사용 피처 수: {len(features)}")

# ============ 공용 모델 학습 + step별 가중 (Softmax over sMAPE) ============
models_lgb, models_xgb = {}, {}
step_weights = {}   # step -> (wl, wx)  [LGB, XGB 가중치]

for step_ahead in range(1, PREDICT_DAYS+1):
    # 타깃: step ahead
    tr = train_df.copy()
    tr['target'] = tr.groupby('메뉴ID')['매출수량'].shift(-step_ahead)
    tr = tr.dropna()
    X_train, y_train = tr[features], tr['target']

    va = valid_df.copy()
    va['target'] = va.groupby('메뉴ID')['매출수량'].shift(-step_ahead)
    va = va.dropna()
    X_val, y_val = va[features], va['target']

    # LGBM
    lgbm = LGBMRegressor(
        objective='regression',
        learning_rate=0.03,
        num_leaves=64,
        n_estimators=2000,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED
    )
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        callbacks=[early_stopping(100), log_evaluation(200)]
    )
    models_lgb[step_ahead] = lgbm
    pred_lgb_va = lgbm.predict(X_val, num_iteration=getattr(lgbm, "best_iteration_", None))
    smape_l = smape(y_val, pred_lgb_va)

    # XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    params_xgb = dict(
        objective='reg:squarederror',
        eval_metric='mae',
        eta=0.03, max_depth=8,
        subsample=0.8, colsample_bytree=0.8,
        seed=SEED
    )
    xgbm = xgb.train(
        params_xgb, dtrain,
        num_boost_round=2000,
        evals=[(dtrain,'train'),(dval,'valid')],
        early_stopping_rounds=100,
        verbose_eval=200
    )
    models_xgb[step_ahead] = xgbm
    best_iter = getattr(xgbm, 'best_iteration', None)
    if best_iter is None:
        best_iter = getattr(xgbm, 'num_boosted_rounds', lambda: 2000)()
    pred_xgb_va = xgbm.predict(dval, iteration_range=(0, best_iter))
    smape_x = smape(y_val, pred_xgb_va)

    # step별 가중치: Softmax over sMAPE (낮을수록 좋음 → -sMAPE를 로짓으로)
    w = softmax_weights_from_smape({"lgb": smape_l, "xgb": smape_x}, temperature=TEMP_SOFTMAX)
    wl, wx = float(w["lgb"]), float(w["xgb"])
    step_weights[step_ahead] = (wl, wx)
    print(f"[step={step_ahead}] sMAPE LGB={smape_l:.4f} | XGB={smape_x:.4f} -> wl={wl:.3f}, wx={wx:.3f}")

print("✅ 모든 Horizon 공용 모델 학습 완료")

# ================= 공용 업장별 Validation sMAPE 분석 =================
valid_metrics = []
valid_preds_common = {}  # (step) -> DataFrame(index=va.index, columns=['pred_common'])

for step_ahead in range(1, PREDICT_DAYS+1):
    va = valid_df.copy()
    va['target'] = va.groupby('메뉴ID')['매출수량'].shift(-step_ahead)
    va = va.dropna()
    X_val, y_val = va[features], va['target']

    preds_lgb = models_lgb[step_ahead].predict(
        X_val, num_iteration=getattr(models_lgb[step_ahead], "best_iteration_", None)
    )
    xgbm = models_xgb[step_ahead]
    best_iter = getattr(xgbm, 'best_iteration', None)
    if best_iter is None:
        best_iter = getattr(xgbm, 'num_boosted_rounds', lambda: 2000)()
    preds_xgb = xgbm.predict(xgb.DMatrix(X_val), iteration_range=(0, best_iter))

    wl, wx = step_weights.get(step_ahead, (0.5,0.5))
    preds_common = wl*preds_lgb + wx*preds_xgb

    va = va.copy()
    va['pred_common'] = preds_common
    valid_preds_common[step_ahead] = va[['영업장명','메뉴ID','target','pred_common']].reset_index(drop=True)

    shop_smape = va.groupby('영업장명').apply(lambda g: smape(g['target'], g['pred_common']))
    for shop, s in shop_smape.items():
        valid_metrics.append({'step': step_ahead, 'shop': shop, 'smape': s})

valid_metrics_df = pd.DataFrame(valid_metrics)
print(valid_metrics_df.pivot(index="shop", columns="step", values="smape").round(4))

# ============ (추가) 전용/그룹 모델 학습 ============

# 1) 그룹 사전 구성: "PAIR_1", "PAIR_2", + 나머지 업장은 각자 단독 그룹
GROUPS = {
    "PAIR_1": set(PAIR_1),
    "PAIR_2": set(PAIR_2),
}
_all_shops = set(train_df['영업장명'].unique().tolist())
_grouped = GROUPS["PAIR_1"] | GROUPS["PAIR_2"]
for s in sorted(_all_shops - _grouped):
    GROUPS[s] = {s}  # 단독 그룹

# (원래 틀 변수명 재사용) SPECIAL_SHOPS를 "그룹 키" 리스트로 대체
SPECIAL_SHOPS = list(GROUPS.keys())

shop_models_lgb = {g:{} for g in SPECIAL_SHOPS}
shop_models_xgb = {g:{} for g in SPECIAL_SHOPS}
shop_step_blend = {g:{} for g in SPECIAL_SHOPS}        # 공용 vs 전용(그룹) 블렌딩 가중
shop_internal_weights = {g:{} for g in SPECIAL_SHOPS}  # 전용 내부 LGB/XGB Softmax 가중 (wl_in, wx_in)

for group_key in SPECIAL_SHOPS:
    members = GROUPS[group_key]
    print(f"\n🔧 전용(그룹) 모델 학습: {group_key} | 멤버: {sorted(members)}")
    # 그룹 데이터만 필터링
    tr_s = train_df[train_df['영업장명'].isin(members)].copy()
    va_s = valid_df[valid_df['영업장명'].isin(members)].copy()

    for step_ahead in range(1, PREDICT_DAYS+1):
        # 타깃 준비 (log1p)
        tr_s_ = tr_s.copy()
        tr_s_['target'] = tr_s_.groupby('메뉴ID')['매출수량'].shift(-step_ahead)
        tr_s_ = tr_s_.dropna()
        if len(tr_s_) == 0:
            continue
        X_tr_s, y_tr_s = tr_s_[features], np.log1p(tr_s_['target'].clip(lower=0))

        va_s_ = va_s.copy()
        va_s_['target'] = va_s_.groupby('메뉴ID')['매출수량'].shift(-step_ahead)
        va_s_ = va_s_.dropna()
        if len(va_s_) == 0:
            continue
        X_va_s, y_va_s = va_s_[features], np.log1p(va_s_['target'].clip(lower=0))
        y_va_true = va_s_['target'].values  # sMAPE는 원척도로 계산

        # LGBM (전용/그룹)
        lgbm_s = LGBMRegressor(
            objective='regression',
            learning_rate=0.03,
            num_leaves=64,
            n_estimators=2000,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED
        )
        lgbm_s.fit(
            X_tr_s, y_tr_s,
            eval_set=[(X_va_s, y_va_s)],
            eval_metric='mae',
            callbacks=[early_stopping(100), log_evaluation(200)]
        )
        shop_models_lgb[group_key][step_ahead] = lgbm_s

        # XGB (전용/그룹)
        dtr_s = xgb.DMatrix(X_tr_s, label=y_tr_s)
        dva_s = xgb.DMatrix(X_va_s, label=y_va_s)
        params_xgb_s = dict(
            objective='reg:squarederror',
            eval_metric='mae',
            eta=0.03, max_depth=8,
            subsample=0.8, colsample_bytree=0.8,
            seed=SEED
        )
        xgbm_s = xgb.train(
            params_xgb_s, dtr_s,
            num_boost_round=2000,
            evals=[(dtr_s,'train'),(dva_s,'valid')],
            early_stopping_rounds=100,
            verbose_eval=200
        )
        shop_models_xgb[group_key][step_ahead] = xgbm_s

        # ---- 전용(그룹) 내부 LGB/XGB Softmax 가중 ----
        best_iter_s = getattr(xgbm_s, 'best_iteration', None)
        if best_iter_s is None:
            best_iter_s = getattr(xgbm_s, 'num_boosted_rounds', lambda: 2000)()

        preds_lgb_s_val_log = lgbm_s.predict(X_va_s, num_iteration=getattr(lgbm_s, "best_iteration_", None))
        preds_xgb_s_val_log = xgbm_s.predict(dva_s, iteration_range=(0, best_iter_s))
        pred_lgb_val = np.expm1(preds_lgb_s_val_log)  # 원척도
        pred_xgb_val = np.expm1(preds_xgb_s_val_log)  # 원척도

        s_lgb = smape(y_va_true, pred_lgb_val)
        s_xgb = smape(y_va_true, pred_xgb_val)
        w_in = softmax_weights_from_smape({"lgb": s_lgb, "xgb": s_xgb}, temperature=TEMP_SOFTMAX)
        wl_in, wx_in = float(w_in["lgb"]), float(w_in["xgb"])
        shop_internal_weights[group_key][step_ahead] = (wl_in, wx_in)

        preds_shop_val = wl_in*pred_lgb_val + wx_in*pred_xgb_val

        # ---- 공용 vs 전용(그룹) 블렌딩 가중 (전용 비중) ----
        va_common = valid_preds_common[step_ahead]
        common_for_group = va_common[va_common['영업장명'].isin(members)].reset_index(drop=True)

        n = min(len(common_for_group), len(preds_shop_val))
        if n == 0:
            continue
        tgt = common_for_group['target'].values[:n]
        pred_common = common_for_group['pred_common'].values[:n]
        pred_shop = preds_shop_val[:n]

        s_common = smape(tgt, pred_common)
        s_shop   = smape(tgt, pred_shop)
        w2 = softmax_weights_from_smape({"shop": s_shop, "common": s_common}, temperature=TEMP_SOFTMAX)
        w_shop = float(w2["shop"])  # 전용(그룹) 비중
        shop_step_blend[group_key][step_ahead] = w_shop

        print(f"[{group_key} step={step_ahead}] 내부 sMAPE LGB={s_lgb:.4f} XGB={s_xgb:.4f} -> wl_in={wl_in:.3f}, wx_in={wx_in:.3f} | "
              f"공용vs전용 sMAPE common={s_common:.4f} shop={s_shop:.4f} -> w_shop={w_shop:.3f}")

print("✅ 전용(그룹) 모델 학습/내부가중/블렌딩 가중 산출 완료")

# ============ 예측 함수 ============
def predict_for_test(test_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    # 동일 전처리 & 인코딩
    test_df = preprocess_base(test_df, use_test_holidays=True)
    # 각 (영업장명_메뉴명) 최근 28일만 사용 (룰 준수)
    test_df = test_df.sort_values(['영업장명_메뉴명','영업일자']).groupby('영업장명_메뉴명').tail(LOOKBACK)

    test_df['메뉴ID']   = le_menu.transform(test_df['영업장명_메뉴명'])
    test_df['영업장ID'] = le_shop.transform(test_df['영업장명'])

    df_feat = create_features(test_df, le_menu, le_shop).fillna(0)

    results = []
    df_pred = df_feat.copy()

    for step_ahead in range(1, PREDICT_DAYS+1):
        X_test = df_pred[features].fillna(0)

        # 공용 예측
        preds_lgb = models_lgb[step_ahead].predict(
            X_test, num_iteration=getattr(models_lgb[step_ahead], "best_iteration_", None)
        )
        xgbm = models_xgb[step_ahead]
        best_iter = getattr(xgbm, 'best_iteration', None)
        if best_iter is None:
            best_iter = getattr(xgbm, 'num_boosted_rounds', lambda: 2000)()
        preds_xgb = xgbm.predict(xgb.DMatrix(X_test), iteration_range=(0, best_iter))

        wl, wx = step_weights.get(step_ahead, (0.5, 0.5))
        preds_common = wl*preds_lgb + wx*preds_xgb

        # 전용(그룹) 예측 + 공용과 블렌딩
        preds_final = preds_common.copy()
        for group_key, members in GROUPS.items():
            mask = df_pred['영업장명'].isin(members)
            if not mask.any():
                continue
            has_lgb = (group_key in shop_models_lgb) and (step_ahead in shop_models_lgb[group_key])
            has_xgb = (group_key in shop_models_xgb) and (step_ahead in shop_models_xgb[group_key])
            if not (has_lgb and has_xgb):
                continue

            wl_in, wx_in = shop_internal_weights.get(group_key, {}).get(step_ahead, (0.5, 0.5))

            X_test_s = X_test[mask]
            lgbm_s = shop_models_lgb[group_key][step_ahead]
            xgbm_s = shop_models_xgb[group_key][step_ahead]
            best_iter_s = getattr(xgbm_s, 'best_iteration', None)
            if best_iter_s is None:
                best_iter_s = getattr(xgbm_s, 'num_boosted_rounds', lambda: 2000)()

            preds_lgb_s_log = lgbm_s.predict(X_test_s, num_iteration=getattr(lgbm_s, "best_iteration_", None))
            preds_xgb_s_log = xgbm_s.predict(xgb.DMatrix(X_test_s), iteration_range=(0, best_iter_s))
            preds_shop = wl_in*np.expm1(preds_lgb_s_log) + wx_in*np.expm1(preds_xgb_s_log)

            # 공용 vs 전용(그룹) 블렌딩
            w_shop = shop_step_blend.get(group_key, {}).get(step_ahead, 0.5)
            preds_final[mask] = w_shop*preds_shop + (1-w_shop)*preds_common[mask]

        # 다음 step을 위해 "예측을 lag_{step}"에 반영
        df_pred[f'lag_{step_ahead}'] = preds_final

        # 각 메뉴ID의 마지막 행 값 → 제출용
        for mid, g in df_pred.groupby('메뉴ID'):
            pred_val = float(preds_final[g.index][-1])
            results.append({
                '영업일자': f"{prefix}+{step_ahead}일",
                '영업장명_메뉴명': le_menu.inverse_transform([int(mid)])[0],
                '매출수량': float(max(pred_val, 0.0))
            })

    return pd.DataFrame(results)

# ============ 전체 TEST 예측/저장 ============
all_preds = []
for path in sorted(glob.glob("/Users/jeong-kyu/Documents/LG_Aimers_7기/open/test/TEST_*.csv")):
    prefix = re.search(r'(TEST_\d+)', os.path.basename(path)).group(1)
    test_df = pd.read_csv(path)
    pred_df = predict_for_test(test_df, prefix)
    all_preds.append(pred_df)

full_pred_df = pd.concat(all_preds, ignore_index=True)

# ---- 주막/카페 보정 (훈련 평균 vs 예측 평균 스케일링) ----
train_means = train.groupby("영업장명")["매출수량"].mean()
pred_means  = full_pred_df.groupby(full_pred_df["영업장명_메뉴명"].str.split("_").str[0])["매출수량"].mean()

for shop in ["화담숲주막", "화담숲카페"]:
    if shop in train_means.index and shop in pred_means.index:
        scale = train_means[shop] / (pred_means[shop] + 1e-6)
        full_pred_df.loc[full_pred_df["영업장명_메뉴명"].str.startswith(shop), "매출수량"] *= scale
        print(f"[보정] {shop} 스케일 {scale:.3f} 적용")

submission = full_pred_df.pivot(index="영업일자",
                                columns="영업장명_메뉴명",
                                values="매출수량").reset_index()

out_path = "/Users/jeong-kyu/Documents/LG_Aimers_7기/open/submission_plus_features_smape4(유형업장).csv"
submission.to_csv(out_path, index=False, encoding="utf-8-sig")
print("✅ 제출 파일 저장 완료:", out_path)
