import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import optuna
import plotly.express as px 
from tqdm import tqdm
import matplotlib.pyplot as plt


desired_pareto = [ ]

def plot_save(pareto,name):
    """
    Plot the Pareto front and save it as an image.
    """
    # Extract DPD and accuracy values from the Pareto front
    dpd_vals = [res['dpd'] for res in pareto]
    acc_vals = [res['accuracy'] for res in pareto]
    alpha_vals = [res['alpha'] for res in pareto]

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'DPD': dpd_vals,
        'Accuracy': acc_vals,
        'alpha': alpha_vals
    })

    # Create a scatter plot using Plotly
    # fig = px.scatter(
    #     plot_df,
    #     x='DPD',
    #     y='Accuracy',
    #     color='alpha',
    #     color_continuous_scale='Viridis',
    #     size_max=15,
    #     hover_data={'alpha': True, 'DPD': True, 'Accuracy': True},
    #     title='Pareto Front: DPD vs. Accuracy'
    # )
    # fig.update_traces(marker=dict(line_width=0))
    
    # # Save the plot as an image
    # fig.write_image(name + ".png")
    plt.figure()
    plt.scatter(dpd_vals, acc_vals, c=alpha_vals, cmap="viridis")
    plt.colorbar(label="alpha")
    plt.xlabel("DPD"); plt.ylabel("Accuracy")
    plt.savefig(name + ".png", dpi=150)
    plt.close()

def find_pareto_front(global_seed: int):
    """
    Runs a multi‐objective Optuna study over accuracy (maximize)
    and demographic parity difference (minimize) for an XGBoost model
    with a custom fairness objective. Returns three lists:
    (dpd_vals, acc_vals, alpha_vals) of Pareto‐optimal solutions.
    """
    # 1) reproducibility
    random.seed(global_seed)
    np.random.seed(global_seed)

    # 2) load data
    df = pd.read_csv('german_credit_data_cleaned.csv')
    y = df['Risk_good'].values
    X_full = df.drop(columns=['Risk_good'])
    sensitive = df['Sex_male'].values

    # 3) custom XGBoost objective factory
    def make_custom_obj(sens_array, alpha=0.5):
        def custom_obj(preds, dmatrix):
            labels = dmatrix.get_label()
            p = 1.0 / (1.0 + np.exp(-preds))
            grad_log = p - labels
            hess_log = p * (1 - p)
            mask_p = (sens_array == 1)
            mask_u = ~mask_p
            n_p, n_u = mask_p.sum(), mask_u.sum()
            # group means
            m_p = p[mask_p].mean() if n_p>0 else 0
            m_u = p[mask_u].mean() if n_u>0 else 0
            sign = np.sign(m_p - m_u)
            d = p * (1 - p)
            grad_fair = np.zeros_like(preds)
            if n_p>0:
                grad_fair[mask_p] =  sign*(1/n_p)*d[mask_p]
            if n_u>0:
                grad_fair[mask_u] = -sign*(1/n_u)*d[mask_u]
            grad = alpha*grad_log + (1-alpha)*grad_fair
            hess = alpha*hess_log + (1-alpha)*0.0
            return grad, hess
        return custom_obj

    # 4) Optuna objective builder
    def create_objective(alpha, seed):
        def objective(trial):
            md = trial.suggest_int('max_depth', 3, 7)
            eta = trial.suggest_float('eta', 0.01, 0.3)
            subs = trial.suggest_float('subsample', 0.6, 1.0)
            col = trial.suggest_float('colsample_bytree', 0.6, 1.0)
            kf = KFold(n_splits=3, shuffle=True, random_state=seed)
            accs, dpds = [], []
            for tr, va in kf.split(X_full):
                Xtr = X_full.iloc[tr].drop(columns=['Sex_male']).values
                ytr = y[tr]
                str_ = sensitive[tr]
                Xva = X_full.iloc[va].drop(columns=['Sex_male']).values
                yva = y[va]
                svr = sensitive[va]
                obj = make_custom_obj(str_, alpha)
                dtr = xgb.DMatrix(Xtr, label=ytr)
                dva = xgb.DMatrix(Xva, label=yva)
                params = dict(max_depth=md, eta=eta,
                              subsample=subs, colsample_bytree=col,
                              verbosity=0, seed=seed+trial.number)
                bst = xgb.train(params, dtr, num_boost_round=100,
                                obj=obj,
                                evals=[(dtr,'train'),(dva,'val')],
                                early_stopping_rounds=10,
                                verbose_eval=False)
                pred = (bst.predict(dva) > 0.5).astype(int)
                accs.append(accuracy_score(yva, pred))
                dpds.append(abs(pred[svr==1].mean() - pred[svr==0].mean()))
            mean_acc, mean_dpd = np.mean(accs), np.mean(dpds)
            trial.set_user_attr('accuracy', mean_acc)
            trial.set_user_attr('dpd', mean_dpd)
            trial.set_user_attr('params', {'alpha':alpha,'max_depth':md,
                                          'eta':eta,'subsample':subs,
                                          'colsample_bytree':col})
            return mean_acc, mean_dpd
        return objective

    # 5) helper funcs
    def test_loss(a, dpd, acc):
        return (1-a)*(1-dpd) + a*acc

    def is_pareto(acc, dpd):
        n = len(acc)
        mask = [True]*n
        for i in range(n):
            for j in range(n):
                if (acc[j]>=acc[i] and dpd[j]<=dpd[i]) and (acc[j]>acc[i] or dpd[j]<dpd[i]):
                    mask[i] = False
                    break
        return mask

    def normalise(v, mn, mx):
        return (v - mn)/(mx - mn) if mx>mn else 0

    # 6) multi‐objective loop
    results = []
    count_alpha = 30
    trials_per_alpha = 100
    sampler = optuna.samplers.NSGAIISampler(seed=global_seed)

    # endpoints α=0 (minimize dpd) and α=1 (maximize acc)
    for alpha in (0.0, 1.0):
        study = optuna.create_study(directions=['maximize','minimize'],
                                    sampler=sampler)
        study.optimize(create_objective(alpha, global_seed),
                       n_trials=trials_per_alpha)
        # pick best by test_loss
        best_val, best_tr = -1, None
        for t in study.best_trials:
            val = test_loss(alpha, t.values[1], t.values[0])
            if val>best_val:
                best_val, best_tr = val, t
        results.append({
            'accuracy': best_tr.values[0],
            'dpd':      best_tr.values[1],
            'alpha':    alpha,
            'params':   best_tr.user_attrs['params']
        })

    # iteratively fill in-between
    max_acc = max(r['accuracy'] for r in results)
    min_acc = min(r['accuracy'] for r in results)
    max_dpd = max(r['dpd']      for r in results)
    min_dpd = min(r['dpd']      for r in results)

    for _ in range(count_alpha):
        # sort by α
        results.sort(key=lambda x: x['alpha'])
        max_dist, pick_alpha = -1, None
        for i in range(len(results)-1):
            d1 = normalise(results[i]['accuracy'], min_acc, max_acc)
            d2 = normalise(results[i+1]['accuracy'], min_acc, max_acc)
            e1 = normalise(results[i]['dpd'],      min_dpd, max_dpd)
            e2 = normalise(results[i+1]['dpd'],      min_dpd, max_dpd)
            dist = (d1-d2)**2 + (e1-e2)**2
            if dist>max_dist:
                max_dist = dist
                pick_alpha = 0.5*(results[i]['alpha'] + results[i+1]['alpha'])
        # run new study
        study = optuna.create_study(directions=['maximize','minimize'],
                                    sampler=sampler)
        study.optimize(create_objective(pick_alpha, global_seed),
                       n_trials=trials_per_alpha)
        best_val, best_tr = -1, None
        for t in study.best_trials:
            val = test_loss(pick_alpha, t.values[1], t.values[0])
            if val>best_val:
                best_val, best_tr = val, t
        results.append({
            'accuracy': best_tr.values[0],
            'dpd':      best_tr.values[1],
            'alpha':    pick_alpha,
            'params':   best_tr.user_attrs['params']
        })
        # update ranges
        max_acc = max(max_acc, best_tr.values[0])
        min_acc = min(min_acc, best_tr.values[0])
        max_dpd = max(max_dpd, best_tr.values[1])
        min_dpd = min(min_dpd, best_tr.values[1])

    # 7) extract Pareto frontier
    acc_list = [r['accuracy'] for r in results]
    dpd_list = [r['dpd']      for r in results]
    mask = is_pareto(acc_list, dpd_list)
    pareto = [r for r, m in zip(results, mask) if m]

    def get_area(results):
        results.sort(key=lambda x: x['dpd'])
        max_acc = max([res['accuracy'] for res in results])
        max_dpd = max([res['dpd'] for res in results])
        min_acc = min([res['accuracy'] for res in results])
        min_dpd = min([res['dpd'] for res in results])
        for i in range(len(results)):
            results[i]['accuracy'] = normalise(results[i]['accuracy'],min_acc,max_acc)
            results[i]['dpd'] = normalise(results[i]['dpd'],min_dpd,max_dpd)
        area = 0
        for i in range(len(results)-1):
            length = results[i]['accuracy']
            width = (results[i+1]['dpd'] - results[i]['dpd'])
            triangle_h = results[i+1]['accuracy'] - results[i]['accuracy']
            triangle_b = results[i+1]['dpd'] - results[i]['dpd']
            triangle_area = 0.5 * triangle_h * triangle_b
            area += length * width
            area += triangle_area
        return area
    
    val = get_area(pareto)
    desired_pareto.append((val, pareto,seed))



seeds = [i for i in range(1, 51)]

for seed in tqdm(seeds, desc="Processing seeds"):
    find_pareto_front(seed)


desired_pareto.sort(key=lambda x: x[0], reverse=True)
count = 0
for x in desired_pareto:
    hyperarea = x[0]
    seed_used = x[2]
    print (f"HyperArea: {hyperarea}")
    count += 1
    plot_save(x[1], f"pareto_front_{seed_used}_{count}")
    # if count == 5:
    #     break
