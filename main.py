import os, sys
import argparse, datetime, glob 
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy

from util import get_default_cfgs, get_parser, instantiate_from_config

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")+1]
    else:
        if opt.name:
            name = "_"+opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = ""
        nowname = now+name+opt.postfix
        output_dir = opt.train_url if opt.train_url is not None else "/data/s3/logs"
        logdir = os.path.join(output_dir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)
    
    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())  

        #trainer_config['limit_val_batches'] = 1
        if config.model.accumulate_grad_batches is not None:
            trainer_config['accumulate_grad_batches'] = config.model.accumulate_grad_batches
        if config.model.max_steps is not None:
            trainer_config['max_steps'] = config.model.max_steps
        if config.model.max_epochs is not None:
            trainer_config['max_epochs'] = config.model.max_epochs
        if opt.world_size > 1:
            trainer_config['num_nodes'] = opt.world_size
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            cpu = True
        else:
            print(f"Running on GPUs {trainer_config['gpus']}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # data and model
        data = instantiate_from_config(config.data)
        model = instantiate_from_config(config.model)
        if 'SparseGPT' in config.model.target:
            model = model.half()

        # trainer and callbacks
        trainer_kwargs = dict()
        logger_cfg, callbacks_cfg = get_default_cfgs(
            lightning_config, opt, nowname, now,
            logdir, ckptdir, cfgdir, config)
        trainer_kwargs['logger'] = instantiate_from_config(logger_cfg)
        trainer_kwargs['callbacks'] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        trainer_kwargs['enable_checkpointing'] = True
        if opt.deepspeed > 0:
            trainer_kwargs['strategy'] = DeepSpeedStrategy(
                stage=opt.deepspeed, offload_optimizer=opt.deepspeed>1,
                offload_parameters= opt.deepspeed==3,
                pin_memory= opt.deepspeed==3,
                allgather_bucket_size=5e8, reduce_bucket_size=5e8, 
                logging_batch_size_per_gpu=config.data.params.batch_size)
        else:
            flag = config.model.find_unused_parameters 
            flag = True if flag is not None else False
            trainer_kwargs['strategy'] = DDPStrategy(find_unused_parameters=flag)
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        
        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            if isinstance(lightning_config.trainer.gpus, int):
                ngpu = lightning_config.trainer.gpus
            else:
                ngpu = len(lightning_config.trainer.gpus.strip(',').split(','))
        else:
            ngpu = 1
        accumulate_grad_batches = lightning_config.trainer.get('accumulate_grad_batches')
        #print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        if accumulate_grad_batches is not None:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        else:
            model.learning_rate = base_lr
        model.scheduler_params = config.model.scheduler_params
        model.optimizer_params = config.model.optimizer_params
        #print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr) * 4".format(
        #    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            pass
            """ 
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)
            """    

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
