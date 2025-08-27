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

SPECIAL_SHOPS = ["화담숲주막", "화담숲카페"]  # 전용 모델을 학습할 업장들

# [UPD: 6) step별 Softmax temperature 스케줄]
# 1~2일은 균등화(온도↑), 후반은 성능 좋은 모델에 집중(온도↓)
TEMP_SOFTMAX_SCHEDULE = {1:1.1, 2:0.9, 3:0.7, 4:0.6, 5:0.5, 6:0.45, 7:0.4}
def temp_for_step(k:int) -> float:
    return float(TEMP_SOFTMAX_SCHEDULE.get(k, 0.6))

# [UPD: 5) 글로벌 shop 정규화 보정 스위치 & 클램프]
APPLY_GLOBAL_SHOP_NORMALIZATION = True
GLOBAL_NORM_MIN = 0.70
GLOBAL_NORM_MAX = 1.30

# [UPD: 4) 롤링 검증용 윈도 길이(일)]
ROLLING_VAL_WINDOWS = [60, 45, 30]

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

# [UPD: 1) Feature 강화 — longer lags / roll_28 / slope / interaction]
def _rolling_slope(arr: np.ndarray) -> float:
    # 선형회귀 기울기 (x=0..n-1)
    if len(arr) < 5 or np.all(np.isnan(arr)):
        return np.nan
    x = np.arange(len(arr))
    y = np.asarray(arr, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 5: return np.nan
    x = x[mask]; y = y[mask]
    if len(x) < 2: return np.nan
    # slope = cov(x,y)/var(x)
    vx = np.var(x)
    if vx == 0: return 0.0
    return float(np.cov(x, y, bias=True)[0,1] / vx)

def create_features(df: pd.DataFrame, le_menu: LabelEncoder, le_shop: LabelEncoder) -> pd.DataFrame:
    df = df.sort_values(['영업장명','메뉴명','영업일자']).copy()

    # 라벨 인코딩
    if '메뉴ID' not in df.columns:
        df['메뉴ID'] = le_menu.transform(df['영업장명_메뉴명'])
    if '영업장ID' not in df.columns:
        df['영업장ID'] = le_shop.transform(df['영업장명'])

    # ---- 기본/확장 lag ----
    for lag in [1,2,7,14,28,35,56]:   # [UPD] 35, 56 추가
        df[f'lag_{lag}'] = df.groupby('메뉴ID')['매출수량'].shift(lag)

    # ---- rolling ----
    for win in [3,7,14,28]:           # [UPD] roll_mean_28 추가
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
    df['share_28'] = df['roll_mean_28'] / (df['shop_roll_mean_28'] + 1e-6)  # [UPD] 수정

    # ---- 앵커 메뉴 ----
    if 'is_anchor' not in df.columns:
        df['is_anchor'] = df.apply(is_anchor_menu, axis=1)
    df['anchor_sales'] = df['매출수량'] * df['is_anchor']
    df['anchor_total'] = df.groupby(['영업장ID','영업일자'])['anchor_sales'].transform('sum')
    df['anchor_total_shift1'] = grp_shop['anchor_total'].shift(1)
    for win in [7,14]:
        df[f'anchor_roll_{win}'] = grp_shop['anchor_total_shift1'].transform(lambda s: s.rolling(win, min_periods=1).mean())

    # ---- 최근 활동성 ----
    df['recent_activity'] = prev.groupby(df['메뉴ID']).transform(lambda s: s.rolling(7, min_periods=1).apply(lambda r: (r>0).any(), raw=True))
    df['active_ratio']    = prev.groupby(df['메뉴ID']).transform(lambda s: (s.rolling(28, min_periods=1).apply(lambda r: (r>0).sum(), raw=True))/28.0)

    # ---- 오픈/휴점 패턴 보강 ----
    df['is_open_7d'] = prev.groupby(df['메뉴ID']).transform(lambda s: s.rolling(7, min_periods=1).max())
    df['open_ratio_7d'] = prev.groupby(df['메뉴ID']).transform(lambda s: s.rolling(7, min_periods=1).apply(lambda r: (r>0).sum(), raw=True) / 7.0)
    df['last_open_gap'] = prev.groupby(df['메뉴ID']).transform(lambda s: s[::-1].groupby((s!=0)[::-1].cumsum()).cumcount()[::-1])

    # ---- [UPD] 추세(slope) 피처: 7/14일
    df['slope_7']  = prev.groupby(df['메뉴ID']).transform(lambda s: s.rolling(7, min_periods=5).apply(_rolling_slope, raw=True))
    df['slope_14'] = prev.groupby(df['메뉴ID']).transform(lambda s: s.rolling(14, min_periods=7).apply(_rolling_slope, raw=True))

    # ---- [UPD] 인터랙션: 주말/공휴일/성수기 × 주요 lag
    for base in ['lag_1','lag_7','lag_14']:
        if base in df.columns:
            df[f'{base}_x_weekend']  = df[base] * df['주말']
            df[f'{base}_x_holiday']  = df[base] * df['공휴일']
            df[f'{base}_x_psummer']  = df[base] * df['peak_summer']
            df[f'{base}_x_pwinter']  = df[base] * df['peak_winter']

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

    # [UPD: 3) 공용 모델도 log1p로 학습 (학습/ES는 로그스케일, 평가는 expm1 후 sMAPE)]
    y_train_log = np.log1p(y_train.clip(lower=0))
    y_val_log   = np.log1p(y_val.clip(lower=0))

    # LGBM
    lgbm = LGBMRegressor(
        objective='regression',
        learning_rate=0.03,
        num_leaves=64,
        n_estimators=2500,          # [UPD] 약간 증대
        subsample=0.8,
        colsample_bytree=0.75,      # [UPD] 살짝 줄여 규제 강화
        min_data_in_leaf=40,        # [UPD] 규제
        reg_lambda=1.0,             # [UPD] L2
        random_state=SEED
    )
    lgbm.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        eval_metric='mae',          # 로그스케일 MAE로 ES
        callbacks=[early_stopping(120), log_evaluation(250)]
    )
    models_lgb[step_ahead] = lgbm
    pred_lgb_va_log = lgbm.predict(X_val, num_iteration=getattr(lgbm, "best_iteration_", None))
    pred_lgb_va = np.expm1(pred_lgb_va_log)
    smape_l = smape(y_val, pred_lgb_va)

    # XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train_log)
    dval   = xgb.DMatrix(X_val,   label=y_val_log)
    params_xgb = dict(
        objective='reg:squarederror',
        eval_metric='mae',    # 로그스케일 MAE
        eta=0.03, max_depth=8,
        subsample=0.8, colsample_bytree=0.75,  # [UPD]
        gamma=0.0, reg_lambda=1.0,             # [UPD] 규제
        seed=SEED
    )
    xgbm = xgb.train(
        params_xgb, dtrain,
        num_boost_round=2500,                 # [UPD]
        evals=[(dtrain,'train'),(dval,'valid')],
        early_stopping_rounds=150,            # [UPD]
        verbose_eval=250
    )
    models_xgb[step_ahead] = xgbm
    best_iter = getattr(xgbm, 'best_iteration', None)
    if best_iter is None:
        best_iter = getattr(xgbm, 'num_boosted_rounds', lambda: 2500)()
    pred_xgb_va = np.expm1(xgbm.predict(dval, iteration_range=(0, best_iter)))
    smape_x = smape(y_val, pred_xgb_va)

    # [UPD: 4+6) 롤링 검증 기반 + step별 템퍼러처로 가중 산정]
    # 기본 단일 검증 sMAPE도 포함하되, 롤링 윈도우 평균으로 보정
    smapes_l, smapes_x = [smape_l], [smape_x]
    for win in ROLLING_VAL_WINDOWS:
        va_win = valid_df[valid_df['영업일자'] > (valid_df['영업일자'].max() - pd.Timedelta(days=win))].copy()
        va_win['target'] = va_win.groupby('메뉴ID')['매출수량'].shift(-step_ahead)
        va_win = va_win.dropna()
        if len(va_win) == 0: 
            continue
        X_w, y_w = va_win[features], va_win['target']
        # 예측 (로그→선형)
        lgb_p = np.expm1(models_lgb[step_ahead].predict(X_w, num_iteration=getattr(models_lgb[step_ahead], "best_iteration_", None)))
        xgb_p = np.expm1(models_xgb[step_ahead].predict(xgb.DMatrix(X_w), iteration_range=(0, best_iter)))
        smapes_l.append(smape(y_w, lgb_p))
        smapes_x.append(smape(y_w, xgb_p))

    sm_l = float(np.mean(smapes_l))
    sm_x = float(np.mean(smapes_x))

    w = softmax_weights_from_smape(
        {"lgb": sm_l, "xgb": sm_x},
        temperature=temp_for_step(step_ahead)
    )
    wl, wx = float(w["lgb"]), float(w["xgb"])
    step_weights[step_ahead] = (wl, wx)
    print(f"[step={step_ahead}] sMAPE LGB={sm_l:.4f} | XGB={sm_x:.4f} (avg over folds) -> wl={wl:.3f}, wx={wx:.3f}")

print("✅ 모든 Horizon 공용 모델 학습 완료")

# ================= 공용 업장별 Validation sMAPE 분석 =================
valid_metrics = []
valid_preds_common = {}  # (step) -> DataFrame(index=va.index, columns=['pred_common'])

for step_ahead in range(1, PREDICT_DAYS+1):
    va = valid_df.copy()
    va['target'] = va.groupby('메뉴ID')['매출수량'].shift(-step_ahead)
    va = va.dropna()
    X_val, y_val = va[features], va['target']

    preds_lgb = np.expm1(models_lgb[step_ahead].predict(
        X_val, num_iteration=getattr(models_lgb[step_ahead], "best_iteration_", None)
    ))
    xgbm = models_xgb[step_ahead]
    best_iter = getattr(xgbm, 'best_iteration', None)
    if best_iter is None:
        best_iter = getattr(xgbm, 'num_boosted_rounds', lambda: 2500)()
    preds_xgb = np.expm1(xgbm.predict(xgb.DMatrix(X_val), iteration_range=(0, best_iter)))

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

# ============ (추가) 전용 모델 학습: SPECIAL_SHOPS ============
shop_models_lgb = {s:{} for s in SPECIAL_SHOPS}
shop_models_xgb = {s:{} for s in SPECIAL_SHOPS}
shop_step_blend = {s:{} for s in SPECIAL_SHOPS}   # 공용 vs 전용 블렌딩 가중(Softmax(sMAPE) 기반, 전용 가중 반환)

for shop in SPECIAL_SHOPS:
    print(f"\n🔧 전용 모델 학습: {shop}")
    # shop 데이터만 필터링
    tr_s = train_df[train_df['영업장명']==shop].copy()
    va_s = valid_df[valid_df['영업장명']==shop].copy()

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
        X_va_s, y_va_s = va_s_[features], np.log1p(va_s_['target'].clip(lower=0))

        # LGBM (전용)
        lgbm_s = LGBMRegressor(
            objective='regression',
            learning_rate=0.03,
            num_leaves=64,
            n_estimators=2500,         # [UPD]
            subsample=0.8,
            colsample_bytree=0.75,     # [UPD]
            min_data_in_leaf=40,       # [UPD]
            reg_lambda=1.0,            # [UPD]
            random_state=SEED
        )
        lgbm_s.fit(
            X_tr_s, y_tr_s,
            eval_set=[(X_va_s, y_va_s)],
            eval_metric='mae',
            callbacks=[early_stopping(120), log_evaluation(250)]
        )
        shop_models_lgb[shop][step_ahead] = lgbm_s

        # XGB (전용)
        dtr_s = xgb.DMatrix(X_tr_s, label=y_tr_s)
        dva_s = xgb.DMatrix(X_va_s, label=y_va_s)
        params_xgb_s = dict(
            objective='reg:squarederror',
            eval_metric='mae',
            eta=0.03, max_depth=8,
            subsample=0.8, colsample_bytree=0.75,  # [UPD]
            gamma=0.0, reg_lambda=1.0,             # [UPD]
            seed=SEED
        )
        xgbm_s = xgb.train(
            params_xgb_s, dtr_s,
            num_boost_round=2500,                 # [UPD]
            evals=[(dtr_s,'train'),(dva_s,'valid')],
            early_stopping_rounds=150,            # [UPD]
            verbose_eval=250
        )
        shop_models_xgb[shop][step_ahead] = xgbm_s

        # 전용 내부 앙상블(여긴 0.5/0.5 유지; 필요시 softmax로 확장 가능)
        best_iter_s = getattr(xgbm_s, 'best_iteration', None)
        if best_iter_s is None:
            best_iter_s = getattr(xgbm_s, 'num_boosted_rounds', lambda: 2500)()
        preds_lgb_s_val_log = lgbm_s.predict(X_va_s, num_iteration=getattr(lgbm_s, "best_iteration_", None))
        preds_xgb_s_val_log = xgbm_s.predict(dva_s, iteration_range=(0, best_iter_s))
        preds_shop_val = 0.5*np.expm1(preds_lgb_s_val_log) + 0.5*np.expm1(preds_xgb_s_val_log)

        # 공용 예측 대비 전용 예측의 블렌딩 가중 산출 (Softmax over sMAPE, step별 temp)
        va_common = valid_preds_common[step_ahead]
        common_for_shop = va_common[va_common['영업장명']==shop].reset_index(drop=True)

        n = min(len(common_for_shop), len(preds_shop_val))
        if n == 0:
            continue
        tgt = common_for_shop['target'].values[:n]
        pred_common = common_for_shop['pred_common'].values[:n]
        pred_shop = preds_shop_val[:n]

        s_common = smape(tgt, pred_common)
        s_shop   = smape(tgt, pred_shop)
        w2 = softmax_weights_from_smape(
            {"shop": s_shop, "common": s_common},
            temperature=temp_for_step(step_ahead)
        )
        w_shop = float(w2["shop"])  # 전용 가중
        shop_step_blend[shop][step_ahead] = w_shop
        print(f"[{shop} step={step_ahead}] sMAPE common={s_common:.4f} | shop={s_shop:.4f} -> w_shop(전용)={w_shop:.3f}")

print("✅ 전용 모델 학습/블렌딩 가중 산출 완료")

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

        # 공용 예측 (로그→선형)
        preds_lgb = np.expm1(models_lgb[step_ahead].predict(
            X_test, num_iteration=getattr(models_lgb[step_ahead], "best_iteration_", None)
        ))
        xgbm = models_xgb[step_ahead]
        best_iter = getattr(xgbm, 'best_iteration', None)
        if best_iter is None:
            best_iter = getattr(xgbm, 'num_boosted_rounds', lambda: 2500)()
        preds_xgb = np.expm1(xgbm.predict(xgb.DMatrix(X_test), iteration_range=(0, best_iter)))

        wl, wx = step_weights.get(step_ahead, (0.5, 0.5))
        preds_common = wl*preds_lgb + wx*preds_xgb

        # 전용 예측(주막/카페 행만) + 공용과 블렌딩
        preds_final = preds_common.copy()
        for shop in SPECIAL_SHOPS:
            mask = df_pred['영업장명'] == shop
            if not mask.any():
                continue
            if (shop not in shop_models_lgb) or (step_ahead not in shop_models_lgb[shop]):
                continue
            X_test_s = X_test[mask]
            # 전용 예측(log→exp)
            lgbm_s = shop_models_lgb[shop][step_ahead]
            xgbm_s = shop_models_xgb[shop][step_ahead]
            best_iter_s = getattr(xgbm_s, 'best_iteration', None)
            if best_iter_s is None:
                best_iter_s = getattr(xgbm_s, 'num_boosted_rounds', lambda: 2500)()
            preds_lgb_s_log = lgbm_s.predict(X_test_s, num_iteration=getattr(lgbm_s, "best_iteration_", None))
            preds_xgb_s_log = xgbm_s.predict(xgb.DMatrix(X_test_s), iteration_range=(0, best_iter_s))
            preds_shop = 0.5*np.expm1(preds_lgb_s_log) + 0.5*np.expm1(preds_xgb_s_log)

            w_shop = shop_step_blend.get(shop, {}).get(step_ahead, 0.5)
            preds_final[mask] = w_shop*preds_shop + (1-w_shop)*preds_common[mask]

        # 다음 step을 위해 "예측을 lag_{step}"에 반영
        df_pred[f'lag_{step_ahead}'] = preds_final

        # 각 메뉴ID의 마지막 행 값 → 제출용
        for mid, g in df_pred.groupby('메뉴ID'):
            pred_val = float(preds_final[g.index][-1])
            results.append({
                '영업일자': f"{prefix}+{step_ahead}일",
                '영업장명_메뉴명': le_menu.inverse_transform([int(mid)])[0],
                '매출수량': max(int(round(pred_val)), 0)
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

# ---- [UPD: 5) 글로벌 shop 정규화 (주막/카페 포함 전 업장, 보정 폭 클램프) ----
if APPLY_GLOBAL_SHOP_NORMALIZATION:
    train_means = train.groupby("영업장명")["매출수량"].mean()
    pred_means  = full_pred_df.groupby(full_pred_df["영업장명_메뉴명"].str.split("_").str[0])["매출수량"].mean()

    for shop, tr_mean in train_means.items():
        if shop in pred_means.index and tr_mean > 0:
            scale = float(tr_mean / (pred_means[shop] + 1e-6))
            # 과보정 방지
            scale = max(GLOBAL_NORM_MIN, min(GLOBAL_NORM_MAX, scale))
            full_pred_df.loc[full_pred_df["영업장명_메뉴명"].str.startswith(shop), "매출수량"] *= scale
            print(f"[글로벌 보정] {shop} 스케일 {scale:.3f} 적용")
else:
    # 기존 두 업장만 보정(원래 코드 유지)
    train_means = train.groupby("영업장명")["매출수량"].mean()
    pred_means  = full_pred_df.groupby(full_pred_df["영업장명_메뉴명"].str.split("_").str[0])["매출수량"].mean()
    for shop in ["화담숲주막", "화담숲카페"]:
        if shop in train_means.index and shop in pred_means.index:
            scale = train_means[shop] / (pred_means[shop] + 1e-6)
            full_pred_df.loc[full_pred_df["영업장명_메뉴명"].str.startswith(shop), "매출수량"] *= scale
            print(f"[보정] {shop} 스케일 {scale:.3f} 적용")

# 제출 Pivot
submission = full_pred_df.pivot(index="영업일자",
                                columns="영업장명_메뉴명",
                                values="매출수량").reset_index()

out_path = "/Users/jeong-kyu/Documents/LG_Aimers_7기/open/submission_plus_features_smape2(기법변경).csv"
submission.to_csv(out_path, index=False, encoding="utf-8-sig")
print("✅ 제출 파일 저장 완료:", out_path)
