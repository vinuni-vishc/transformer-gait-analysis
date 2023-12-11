import os
import numpy as np
import torch
import shutil
import pandas as pd
import statsmodels.regression.linear_model as sm

from GaitDataset import GaitDataset
from config import *

def test(dataset:GaitDataset, num_epochs, model, criterion, optimizer):
    print("===================TESTING===================")

    X, y = dataset.all_dataset

    datasplit_df = pd.read_csv(DATA_PATH + '/train_test_valid_id_split.csv')
    datasplit_df['videoid'] = datasplit_df['videoid'].apply(lambda x: int(x))
    video_ids = [t[0] for t in dataset.video_segments if t[0] in dataset.datasplit_ids['all']]

    if dataset.doublesided:
        video_ids = video_ids + video_ids
        predictions_df = pd.DataFrame(video_ids,columns=['videoid'])
        side_col = ['L' if i < len(predictions_df)/2 else 'R' for i in range(len(predictions_df))]
        predictions_df[dataset.target_metric] = y
        predictions_df['side'] = side_col
        predictions_df = predictions_df.merge(right=datasplit_df[['videoid','dataset']],on=['videoid'],how='left')
    else:
        predictions_df = pd.DataFrame(video_ids,columns=['videoid'])
        predictions_df[dataset.target_metric] = y
        predictions_df = predictions_df.merge(right=datasplit_df[['videoid','dataset']],on=['videoid'],how='left')

    for epoch in range(num_epochs):
        predictions = []
        test_model = model
        checkpoint = torch.load(f"./checkpoints/{dataset.target_metric}/epoch_{epoch+1}.pth")
        test_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for i in range(0, X.size()[0], 30000):
            with torch.no_grad():
                test_model.eval()
                preds = test_model(X[i: i+30000])
                batch_size = preds.size()[0]
                preds = preds * dataset.target_range + dataset.target_min
                predictions.extend(np.array(preds.cpu()))
        predictions_df[f"{dataset.target_metric}_pred_{epoch + 1}"] = predictions
    
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1} completed")
    
    if dataset.doublesided:
        preds_by_id = predictions_df.groupby(['videoid','side','dataset'],as_index=False).mean()
    else:
        preds_by_id = predictions_df.groupby(['videoid','dataset'],as_index=False).mean()
    
    preds_by_id['const'] = 1

    #Fit a least square model to remove bias for videos with larger number of segments.
    def get_corrected_model(iteration):
        lm = sm.OLS(preds_by_id[preds_by_id['dataset'] == 'train'][dataset.target_metric].values,
                            preds_by_id[preds_by_id['dataset'] == 'train'][['%s_pred_%s' % (dataset.target_metric,iteration),"const"]].values).fit()

        preds_by_id['%s_pred_%s_corrected' % (dataset.target_metric,iteration)] = lm.predict(preds_by_id[['%s_pred_%s' % (dataset.target_metric,iteration),"const"]])
        preds_by_id['error2'] = np.square(preds_by_id[dataset.target_metric] - preds_by_id['%s_pred_%s_corrected' % (dataset.target_metric,iteration)])
        rmses = np.sqrt(preds_by_id.groupby('dataset')['error2'].mean())
        return lm, rmses

    train_rmse = []
    val_rmse = []
    for i in range(1, num_epochs + 1):
        _, rmses = get_corrected_model(i)
        train_rmse.append(rmses.loc['train'])
        val_rmse.append(rmses.loc['validation'])

    ### Best epoch is the one with lowest RMSE for the validation set.
    best_epoch = np.argmin(val_rmse) + 1
    print(f'Best epoch: {best_epoch}')
    print(f"Best validation RMSE: {np.min(val_rmse)}")

    if not os.path.exists('./predictions'):
        os.makedirs('./predictions')
    preds_by_id.to_csv(f'./predictions/{dataset.target_metric}_predictions.csv')

    if not os.path.exists('./best_weights'):
        os.makedirs('./best_weights')
    shutil.copy2(f'./checkpoints/{dataset.target_metric}/epoch_{best_epoch}.pth', f'./best_weights/{dataset.target_metric}_best_epoch.pth')

    test_df = preds_by_id[preds_by_id['dataset'] == 'test']
    ground_truth = test_df[f'{dataset.target_metric}']
    best_epoch_preds = test_df[f'{dataset.target_metric}_pred_{epoch}_corrected']

    preds_df = {"Ground Truth": ground_truth, "Predictions": best_epoch_preds}
    data = pd.DataFrame(preds_df)

    print(f"Epoch {best_epoch}'s correlation: {data.corr()['Predictions']['Ground Truth']}")
    print(f"Epoch {best_epoch}'s test RMSE: {np.sqrt(np.square(ground_truth - best_epoch_preds).mean())}")
    print(f"Epoch {best_epoch}'s test MAE: {np.abs(ground_truth - best_epoch_preds).mean()}")