import torch.optim as optim


################################################################################
# The commonly used interact
################################################################################

def interact_datetime_train(period: int):
    return {
        "fname_log": "os.path.join($(workspace_root), 'train/logs/" +
                     "{}.log'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))",
        "summary_root": "os.path.join($(workspace_root), 'train/summary/')",
        "period": period,
    }


def interact_datetime_evaluate():
    return {
        "fname_log": "os.path.join($(workspace_root), 'evaluate/logs/" +
                     "{}.log'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))",
    }
