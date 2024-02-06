from data.MyDataset import PTCRDataset, MyDataset, PLATEDataset, MetaEmbDataset


def create_dataset(data_dir, args, mode):
    if args.model_name == 'PLATE':
        if mode == 'val' or mode == 'test':
            return PLATEDataset(data_dir=data_dir, max_length=args.maxlen,
                             mode=mode, neg_num=args.num_test_neg_item, device=args.device)
        else:
            return PLATEDataset(data_dir=data_dir, max_length=args.maxlen,
                            mode=mode, device=args.device)
    elif args.model_name == 'MetaEmb':
        if mode == 'val' or mode == 'test':
            return MetaEmbDataset(data_dir=data_dir, max_length=args.maxlen,
                              mode=mode, neg_num=args.num_test_neg_item, device=args.device)
        else:
            return MetaEmbDataset(data_dir=data_dir, max_length=args.maxlen,
                            mode=mode, device=args.device, K=args.K)
    elif args.model_name  == "DSSM_SASRec_PTCR" or "PTCR" in args.model_name:
        if mode == 'val' or mode == 'test':
            return PTCRDataset(data_dir=data_dir, max_length=args.maxlen,
                             mode=mode, neg_num=args.num_test_neg_item, device=args.device)
        else:
            return PTCRDataset(data_dir=data_dir, max_length=args.maxlen,
                           mode=mode, device=args.device)
    else:
        if mode == 'val' or mode == 'test':
            return MyDataset(data_dir=data_dir, max_length=args.maxlen,
                             mode=mode, neg_num=args.num_test_neg_item, device=args.device)
        else:
            return MyDataset(data_dir=data_dir, max_length=args.maxlen,
                             mode=mode,  device=args.device)
