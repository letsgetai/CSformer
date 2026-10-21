from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar,Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'solar': Dataset_Solar,
}


def data_provider(args, flag):
    """Build a dataset and loader for ``train``, ``val``, ``test`` or ``pred``.

    The validation loader has two explicit protocols:

    ``legacy``
        ``shuffle=True, drop_last=True``.  This is the historical loader and the
        protocol used for the released single-seed results.  Every epoch sees a
        different random subset and the incomplete final batch is discarded.
    ``full``
        ``shuffle=False, drop_last=False``.  Every validation sample is evaluated
        in a deterministic order, so early-stopping decisions are comparable
        across epochs.

    Training keeps the historical shuffled, drop-last loader.  Testing and
    prediction always keep every sample (``shuffle=False, drop_last=False``).
    """
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    val_protocol = getattr(args, 'val_protocol', 'legacy')
    if val_protocol not in ('legacy', 'full'):
        raise ValueError("val_protocol must be 'legacy' or 'full', got {!r}".format(val_protocol))

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    elif flag == 'val' and val_protocol == 'full':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
