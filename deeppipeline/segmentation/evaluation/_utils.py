import torch
import os
import glob

from deeppipeline.segmentation.models import init_model


def load_fold(args, fold_id):
    snapshot_name = glob.glob(os.path.join(args.snapshots_root, args.snapshot, f'fold_{fold_id}*.pth'))
    if len(snapshot_name) == 0:
        raise ValueError("Snapshot can't be found")
    snapshot_name = snapshot_name[0]

    net = init_model(ignore_data_parallel=True)
    snp = torch.load(snapshot_name)
    if isinstance(snp, dict):
        net_snp = snp['model']
    else:
        net_snp = snp
    net.load_state_dict(net_snp)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net).to('cuda')
    net.eval()
    return net

def run_oof_binary(session_backup, read_img, read_mask, img_group_id_colname=None):
    metadata = session_backup[f'metadata'][0]
    if img_group_id_colname is not None:
        for group_name, _ in metadata.groupby(by=img_group_id_colname):
            os.makedirs(os.path.join(args.snapshots_root, args.snapshot, 'oof_inference', group_name), exist_ok=True)
    else:
        os.makedirs(os.path.join(args.snapshots_root, args.snapshot, 'oof_inference'), exist_ok=True)

    for fold_id, _, val_set in session_backup['cv_split'][0]:
        print(colored('====> ', 'green') + f'Loading fold [{fold_id}]')
        net = load_fold(args, fold_id)

        if args.tta:
            raise NotImplementedError('TTA is not yet supported')

        val_dataset = SegmentationDataset(split=val_set,
                                          trf=session_backup['val_trf'][0],
                                          read_img=read_gs_ocv,
                                          read_mask=read_gs_binary_mask_ocv,
                                          img_group_id_colname=img_group_id_colname)

        val_loader = DataLoader(val_dataset, batch_size=args.bs,
                                num_workers=args.n_threads,
                                sampler=SequentialSampler(val_dataset))

        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader), desc=f'Predicting fold {fold_id}:'):
                img = batch['img']
                if img_group_id_colname is not None:
                    group_ids = batch['group_id']
                else:
                    group_ids = None
                fnames = batch['fname']
                predicts = torch.sigmoid(net(img)).mul(255).to('cpu').numpy().astype(np.uint8)

                for idx, fname in enumerate(fnames):
                    pred_mask = predicts[idx].squeeze()
                    if img_group_id_colname is not None:
                        cv2.imwrite(os.path.join(args.snapshots_root,
                                                args.snapshot,
                                                'oof_inference', group_ids[idx], fname), pred_mask)
                    else:
                        cv2.imwrite(os.path.join(args.snapshots_root,
                                                args.snapshot,
                                                'oof_inference', fname), pred_mask)
