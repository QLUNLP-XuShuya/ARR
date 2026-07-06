import os
from pprint import pprint
from configs.config import parser
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks
from models.ARR import ARR
from lightning.pytorch import seed_everything
import lightning.pytorch as pl


def _set_eval_flags(model, on=False):
    state = {}
    for attr in ['use_alignment']:
        if hasattr(model, attr):
            state[attr] = getattr(model, attr)
            setattr(model, attr, on)
    return state


def _restore_flags(model, state):
    for k, v in state.items():
        setattr(model, k, v)


def train(args):
    if args.local_rank == 0:
        print("=" * 60)
        print(f"use_alignment={getattr(args, 'use_alignment', False)} | "
              f"use_cross_attention={getattr(args, 'use_cross_attention', False)} | "
              f"use_cot={getattr(args, 'use_cot', False)} | "
              f"grad_ckpt={getattr(args, 'gradient_checkpointing', True)}")
        print(f"batch_size={args.batch_size}, accumulate={args.accumulate_grad_batches}")
        print("=" * 60)

    dm = DataModule(args)
    callbacks = add_callbacks(args)

    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks["callbacks"],
        logger=callbacks["loggers"],
        gradient_clip_val=getattr(args, 'gradient_clip_val', 1.0),
        gradient_clip_algorithm=getattr(args, 'gradient_clip_algorithm', 'norm'),
    )

    if args.ckpt_file is not None:
        print(f"Loading checkpoint from {args.ckpt_file}")
        model = ARR.load_from_checkpoint(args.ckpt_file, strict=False, args=args)
    else:
        model = ARR(args)

    if args.local_rank == 0:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel: total={total:,} | trainable={trainable:,} ({trainable/total:.2%})")

    if args.test:
        if args.local_rank == 0:
            print("\nStarting Testing...")
        prev = _set_eval_flags(model, on=False)
        try:
            trainer.test(model, datamodule=dm)
        finally:
            _restore_flags(model, prev)
    elif args.validate:
        if args.local_rank == 0:
            print("\nStarting Validation...")
        prev = _set_eval_flags(model, on=False)
        try:
            trainer.validate(model, datamodule=dm)
        finally:
            _restore_flags(model, prev)
    else:
        if args.local_rank == 0:
            print("\nStarting Training...")
        trainer.fit(model, datamodule=dm)


def _ensure(args, name, val):
    if not hasattr(args, name) or getattr(args, name) is None:
        setattr(args, name, val)


def main():
    args = parser.parse_args()
    if not hasattr(args, 'local_rank') or args.local_rank < 0:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    _ensure(args, 'use_alignment', False)
    _ensure(args, 'align_loss_weight', 0.1)
    _ensure(args, 'align_temperature', 0.07)
    _ensure(args, 'max_phrases_per_report', 15)
    _ensure(args, 'gradient_clip_val', 1.0)
    _ensure(args, 'gradient_clip_algorithm', 'norm')
    _ensure(args, 'gradient_checkpointing', True)
    _ensure(args, 'use_cot', False)
    _ensure(args, 'cot_findings_loss_weight', 0.3)
    _ensure(args, 'cot_reasoning_loss_weight', 0.3)
    _ensure(args, 'cot_report_loss_weight', 1.0)
    _ensure(args, 'cot_split_strategy', 'heuristic')
    _ensure(args, 'cot_split_ratio', 0.7)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    if args.local_rank == 0:
        print("\n" + "=" * 60)
        print("ARR Configuration")
        print("=" * 60)
        pprint(vars(args))
        print("=" * 60 + "\n")

    seed_everything(42, workers=True)
    train(args)


if __name__ == '__main__':
    main()
